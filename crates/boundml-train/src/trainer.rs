//! The training loop: plan and budget-check first, then allocate, stream minibatches,
//! update with SGD/AdamW, validate per epoch and checkpoint atomically.
//!
//! Determinism: given the same config, data files, binary and CPU, an uninterrupted
//! run and a run that is stopped and resumed from any checkpoint produce bitwise
//! identical parameters. Results may differ across CPUs, compilers or builds.

use crate::checkpoint::{
    CHECKPOINT_FORMAT, CHECKPOINT_VERSION, Checkpoint, CheckpointState, EpochRecord, Progress,
};
use crate::config::TrainConfig;
use crate::error::{Result, TrainError};
use crate::metrics::{LabelStats, MetricAcc};
use boundml_artifact::DataRef;
use boundml_artifact::fsutil::{sha256_file, sha256_hex};
use boundml_core::{Executor, Graph, MemoryReport, Mode, Optimizer, Plan, Rng, Task};
use boundml_data::{BatchReader, CsvSource, Preprocessor, RawRow, Schema};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum TrainEvent {
    Planned(Box<MemoryReport>),
    Resumed { epoch: u32, global_step: u64 },
    Epoch(EpochRecord),
    Checkpoint { global_step: u64 },
    Stopped { global_step: u64 },
}

pub struct TrainOptions<'a> {
    pub checkpoint_dir: PathBuf,
    pub resume: bool,
    /// Stop (after checkpointing) once the global step count reaches this value.
    pub max_steps: Option<u64>,
    pub should_stop: &'a (dyn Fn() -> bool + Sync),
    pub on_event: &'a mut dyn FnMut(&TrainEvent),
}

#[derive(Debug, Clone)]
pub struct TrainOutcome {
    pub completed: bool,
    pub global_step: u64,
    pub history: Vec<EpochRecord>,
    pub report: MemoryReport,
}

pub fn epoch_rng(seed: u64, epoch: u32) -> Rng {
    Rng::seed(seed ^ 0xD1B5_4A32_D192_ED03u64.wrapping_mul(u64::from(epoch) + 1))
}

pub fn config_sha256(cfg: &TrainConfig) -> String {
    sha256_hex(&serde_json::to_vec(cfg).unwrap_or_default())
}

/// Training-mode memory report (no tensor allocation).
pub fn plan_training(cfg: &TrainConfig) -> Result<(Graph, Plan, MemoryReport)> {
    let graph = cfg.dense_spec().graph()?;
    let plan = Plan::new(
        &graph,
        Mode::Training,
        cfg.training.batch_size,
        cfg.training.planner,
    )?;
    let validation = validation_plan(cfg, &graph)?;
    let report = plan
        .report(&graph, 1, cfg.training.optimizer.state_slots())
        .with_training_extras(
            validation.as_ref().map_or(0, Plan::arena_bytes),
            staging_bytes(cfg),
            data_loader_bytes(cfg),
        );
    Ok((graph, plan, report))
}

/// The separate inference plan used to evaluate the validation file each epoch.
fn validation_plan(cfg: &TrainConfig, graph: &Graph) -> Result<Option<Plan>> {
    match cfg.data.valid {
        Some(_) => Ok(Some(Plan::new(
            graph,
            Mode::Inference,
            cfg.training.batch_size,
            cfg.training.planner,
        )?)),
        None => Ok(None),
    }
}

/// Tensor staging buffers owned by the training loop alongside the arenas: the
/// batch buffers (features f32, labels f32, line numbers u64) and, with a
/// validation file, the evaluation buffers (the same plus outputs).
pub fn staging_bytes(cfg: &TrainConfig) -> u64 {
    let bs = cfg.training.batch_size as u64;
    let nf = cfg.data.features.len() as u64;
    let no = cfg.schema().task().output_cols() as u64;
    let batch = bs * (nf * 4 + 4 + 8);
    let eval = if cfg.data.valid.is_some() {
        bs * (nf * 4 + 4 + 8 + no * 4)
    } else {
        0
    };
    batch + eval
}

/// Data-loader buffers (reported, not budgeted): the training shuffle buffer of
/// `shuffle_buffer` preprocessed rows, plus one row for the validation reader.
pub fn data_loader_bytes(cfg: &TrainConfig) -> u64 {
    let nf = cfg.data.features.len() as u64;
    let row = nf * 4 + 4 + 8;
    let valid = if cfg.data.valid.is_some() { row } else { 0 };
    cfg.data.shuffle_buffer as u64 * row + valid
}

/// Inference-mode memory report for the deploy settings.
pub fn plan_inference(cfg: &TrainConfig) -> Result<MemoryReport> {
    let graph = cfg.dense_spec().graph()?;
    let plan = Plan::new(
        &graph,
        Mode::Inference,
        cfg.deploy.max_batch,
        cfg.deploy.planner,
    )?;
    Ok(plan.report(&graph, cfg.deploy.sessions, 0))
}

fn label_stats(path: &Path, schema: &Schema) -> Result<LabelStats> {
    let mut src = CsvSource::open(path, schema, true)?;
    let mut stats = LabelStats::new(&schema.task());
    let mut row = RawRow::default();
    while src.read_row(&mut row)? {
        stats.add(row.label.unwrap_or(0.0));
    }
    Ok(stats)
}

struct Run {
    cfg: TrainConfig,
    schema: Schema,
    pre: Preprocessor,
    graph: Arc<Graph>,
    labels: LabelStats,
    data: Vec<DataRef>,
    train_path: PathBuf,
    valid_path: Option<PathBuf>,
    history: Vec<EpochRecord>,
}

pub fn train(cfg: &TrainConfig, base: &Path, opts: TrainOptions<'_>) -> Result<TrainOutcome> {
    cfg.validate()?;
    let (graph, plan, report) = plan_training(cfg)?;
    if let Some(budget) = cfg.train_budget()? {
        report.check_budget(budget)?;
    }
    (opts.on_event)(&TrainEvent::Planned(Box::new(report.clone())));
    let schema = cfg.schema();
    let train_path = base.join(&cfg.data.train);
    let valid_path = cfg.data.valid.as_ref().map(|v| base.join(v));
    let (train_hash, _) = sha256_file(&train_path)?;
    let cfg_hash = config_sha256(cfg);

    let (pre, labels, mut params, mut opt, mut progress, history) = if opts.resume {
        let ck = Checkpoint::load(&opts.checkpoint_dir)?;
        if ck.state.config_sha256 != cfg_hash {
            return Err(TrainError::Checkpoint(
                "configuration differs from the checkpoint's; resume requires the identical config"
                    .into(),
            ));
        }
        let saved = ck.state.data.iter().find(|d| d.role == "train");
        if saved.map(|d| d.sha256.as_str()) != Some(train_hash.as_str()) {
            return Err(TrainError::DataChanged(format!(
                "{} hash differs",
                train_path.display()
            )));
        }
        // The validation file affects recorded metrics, so it must be unchanged too.
        let saved_valid = ck
            .state
            .data
            .iter()
            .find(|d| d.role == "valid")
            .map(|d| d.sha256.clone());
        let current_valid = match &valid_path {
            Some(v) => Some(sha256_file(v)?.0),
            None => None,
        };
        if saved_valid != current_valid {
            return Err(TrainError::DataChanged(
                "the validation file differs from the one recorded in the checkpoint".into(),
            ));
        }
        let mut opt = Optimizer::new(cfg.training.optimizer.clone(), &graph)?;
        opt.restore(ck.optimizer)?;
        (opts.on_event)(&TrainEvent::Resumed {
            epoch: ck.state.progress.epoch,
            global_step: ck.state.progress.global_step,
        });
        let s = ck.state;
        (
            s.preprocessing,
            s.label_stats,
            ck.params,
            opt,
            s.progress,
            s.history,
        )
    } else {
        let pre = Preprocessor::fit(&train_path, &schema, cfg.data.normalize)?;
        let labels = label_stats(&train_path, &schema)?;
        let spec = cfg.dense_spec();
        let mut params = spec.init_params(&graph, &mut Rng::seed(cfg.seed));
        init_output_bias(&graph, &mut params, &labels, &schema.task());
        let opt = Optimizer::new(cfg.training.optimizer.clone(), &graph)?;
        let progress = Progress {
            epoch: 0,
            batches_in_epoch: 0,
            global_step: 0,
            reader: None,
            epoch_loss_sum: 0.0,
            epoch_rows: 0,
            completed: false,
        };
        (pre, labels, params, opt, progress, Vec::new())
    };

    let mut data = vec![DataRef {
        role: "train".into(),
        file_name: file_name(&train_path),
        sha256: train_hash,
        rows: pre.rows_fitted,
    }];
    if let Some(v) = &valid_path {
        data.push(DataRef {
            role: "valid".into(),
            file_name: file_name(v),
            sha256: sha256_file(v)?.0,
            rows: 0,
        });
    }
    let graph = Arc::new(graph);
    let mut run = Run {
        cfg: cfg.clone(),
        schema,
        pre,
        graph: Arc::clone(&graph),
        labels,
        data,
        train_path,
        valid_path,
        history,
    };
    if progress.completed {
        return Ok(TrainOutcome {
            completed: true,
            global_step: progress.global_step,
            history: run.history,
            report,
        });
    }

    // Large tensor storage is allocated only now, after planning and budget checks.
    let mut exec = Executor::from_plan(Arc::clone(&graph), plan);
    let bs = cfg.training.batch_size;
    let nf = run.schema.features.len();
    let (mut x, mut y, mut lines) = (vec![0.0f32; bs * nf], vec![0.0f32; bs], vec![0u64; bs]);
    // The validation arena is part of the budgeted plan (see `plan_training`).
    let mut valid_exec =
        validation_plan(cfg, &graph)?.map(|p| Executor::from_plan(Arc::clone(&graph), p));

    while progress.epoch < cfg.training.epochs {
        let mut reader = match &progress.reader {
            Some(pos) => BatchReader::resume(
                &run.train_path,
                &run.schema,
                &run.pre,
                cfg.data.shuffle_buffer,
                pos,
            )?,
            None => BatchReader::open(
                &run.train_path,
                &run.schema,
                &run.pre,
                cfg.data.shuffle_buffer,
                epoch_rng(cfg.seed, progress.epoch),
                true,
            )?,
        };
        loop {
            let n = reader.next_batch(bs, &mut x, &mut y, &mut lines)?;
            if n == 0 {
                break;
            }
            let loss = exec.forward_backward(&params, &x[..n * nf], &y[..n], n)?;
            opt.step(&mut params, exec.param_grads())?;
            progress.global_step += 1;
            progress.batches_in_epoch += 1;
            progress.epoch_loss_sum += f64::from(loss) * n as f64;
            progress.epoch_rows += n as u64;
            progress.reader = Some(reader.position());
            let stop =
                (opts.should_stop)() || opts.max_steps.is_some_and(|m| progress.global_step >= m);
            let periodic = cfg.training.checkpoint_every_steps > 0
                && progress.global_step % cfg.training.checkpoint_every_steps == 0;
            if stop || periodic {
                save(&run, &opts.checkpoint_dir, &params, &opt, &progress)?;
                (opts.on_event)(&TrainEvent::Checkpoint {
                    global_step: progress.global_step,
                });
            }
            if stop {
                (opts.on_event)(&TrainEvent::Stopped {
                    global_step: progress.global_step,
                });
                return Ok(TrainOutcome {
                    completed: false,
                    global_step: progress.global_step,
                    history: run.history,
                    report,
                });
            }
        }
        let valid = match (&run.valid_path, valid_exec.as_mut()) {
            (Some(p), Some(ex)) => {
                evaluate_preprocessed(ex, &params, p, &run.schema, &run.pre, bs)?
            }
            _ => BTreeMap::new(),
        };
        let record = EpochRecord {
            epoch: progress.epoch,
            train_loss: progress.epoch_loss_sum / progress.epoch_rows.max(1) as f64,
            valid,
        };
        (opts.on_event)(&TrainEvent::Epoch(record.clone()));
        run.history.push(record);
        progress.epoch += 1;
        progress.batches_in_epoch = 0;
        progress.reader = None;
        progress.epoch_loss_sum = 0.0;
        progress.epoch_rows = 0;
        progress.completed = progress.epoch >= cfg.training.epochs;
        save(&run, &opts.checkpoint_dir, &params, &opt, &progress)?;
        (opts.on_event)(&TrainEvent::Checkpoint {
            global_step: progress.global_step,
        });
    }
    Ok(TrainOutcome {
        completed: true,
        global_step: progress.global_step,
        history: run.history,
        report,
    })
}

/// Start the output layer at the constant baseline: bias = training label mean
/// (regression), log-odds of the positive class (binary) or log class priors
/// (multiclass). Weights keep their random initialization. Classes with zero
/// training rows get a floor probability of 1e-3 before the log.
pub fn init_output_bias(graph: &Graph, params: &mut [f32], labels: &LabelStats, task: &Task) {
    let Some(last) = graph.params().len().checked_sub(1) else {
        return;
    };
    if graph.params()[last].rows != 1 {
        return;
    }
    let o = graph.param_offset(last);
    let len = graph.param_len(last);
    let prior = labels.constant_output(task);
    match task {
        Task::Regression => params[o] = prior[0],
        Task::Binary => {
            let p = prior[0].clamp(1e-3, 1.0 - 1e-3);
            params[o] = (p / (1.0 - p)).ln();
        }
        Task::Multiclass { .. } => {
            for (b, p) in params[o..o + len].iter_mut().zip(&prior) {
                *b = p.max(1e-3).ln();
            }
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn save(run: &Run, dir: &Path, params: &[f32], opt: &Optimizer, progress: &Progress) -> Result<()> {
    let mut ck = Checkpoint {
        state: CheckpointState {
            format: CHECKPOINT_FORMAT.into(),
            format_version: CHECKPOINT_VERSION,
            runtime: boundml_core::VERSION.into(),
            config: run.cfg.clone(),
            config_sha256: config_sha256(&run.cfg),
            schema: run.schema.clone(),
            preprocessing: run.pre.clone(),
            spec: run.cfg.dense_spec(),
            label_stats: run.labels.clone(),
            optimizer_step: opt.steps(),
            progress: progress.clone(),
            history: run.history.clone(),
            data: run.data.clone(),
            tensors_sha256: String::new(),
        },
        params: params.to_vec(),
        optimizer: opt.state(),
    };
    ck.save(dir, &run.graph)
}

/// Evaluate a model on a CSV using a fitted preprocessor (file order, bounded memory).
pub fn evaluate_preprocessed(
    exec: &mut Executor,
    params: &[f32],
    path: &Path,
    schema: &Schema,
    pre: &Preprocessor,
    batch: usize,
) -> Result<BTreeMap<String, f64>> {
    let task = schema.task();
    let nf = schema.features.len();
    let no = task.output_cols();
    let mut reader = BatchReader::open(path, schema, pre, 1, Rng::seed(0), true)?;
    let (mut x, mut y, mut lines) = (
        vec![0.0f32; batch * nf],
        vec![0.0f32; batch],
        vec![0u64; batch],
    );
    let mut out = vec![0.0f32; batch * no];
    let mut acc = MetricAcc::new(task);
    loop {
        let n = reader.next_batch(batch, &mut x, &mut y, &mut lines)?;
        if n == 0 {
            break;
        }
        exec.infer(params, &x[..n * nf], n, &mut out[..n * no])?;
        for r in 0..n {
            acc.add(&out[r * no..(r + 1) * no], y[r]);
        }
    }
    Ok(acc.finish())
}

/// Predicts `n` raw rows (schema order) into an output buffer.
pub type PredictFn<'a> = dyn FnMut(&[Option<f64>], usize, &mut [f32]) -> Result<()> + 'a;

/// Evaluate model outputs and the constant baseline on the same rows.
pub fn evaluate_with_baseline(
    predict: &mut PredictFn<'_>,
    path: &Path,
    schema: &Schema,
    labels: &LabelStats,
    batch: usize,
) -> Result<(BTreeMap<String, f64>, BTreeMap<String, f64>)> {
    let task: Task = schema.task();
    let nf = schema.features.len();
    let no = task.output_cols();
    let mut src = CsvSource::open(path, schema, true)?;
    let mut raw = vec![None; batch * nf];
    let mut y = vec![0.0f32; batch];
    let mut out = vec![0.0f32; batch * no];
    let constant = labels.constant_output(&task);
    let (mut acc, mut base) = (MetricAcc::new(task), MetricAcc::new(task));
    let mut row = RawRow::default();
    loop {
        let mut n = 0;
        while n < batch && src.read_row(&mut row)? {
            raw[n * nf..(n + 1) * nf].copy_from_slice(&row.features);
            y[n] = row.label.unwrap_or(0.0);
            n += 1;
        }
        if n == 0 {
            break;
        }
        predict(&raw[..n * nf], n, &mut out[..n * no])?;
        for r in 0..n {
            acc.add(&out[r * no..(r + 1) * no], y[r]);
            base.add(&constant, y[r]);
        }
        if n < batch {
            break;
        }
    }
    Ok((acc.finish(), base.finish()))
}
