//! Command implementations. Each returns structured JSON plus human-readable text;
//! `main.rs` chooses which to print. Errors are strings beginning with a `BML-` code.

use crate::exit;
use crate::serve::{AppState, ServeConfig};
use boundml_artifact::{CompareOptions, Package, Verdict, compare};
use boundml_core::{MemoryReport, Mode, PlannerKind};
use boundml_data::{CsvSource, RawRow};
use boundml_train::config::parse_bytes;
use boundml_train::{TrainConfig, TrainEvent, TrainOptions};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct Outcome {
    pub json: Value,
    pub text: String,
    pub exit: i32,
}

pub type CmdResult = Result<Outcome, String>;

fn ok(json: Value, text: String) -> CmdResult {
    Ok(Outcome {
        json,
        text,
        exit: exit::OK,
    })
}

fn s<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn mib(b: u64) -> String {
    if b >= 1 << 20 {
        format!("{:.2} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

pub fn report_text(r: &MemoryReport) -> String {
    let mut t = String::new();
    let _ = writeln!(
        t,
        "  mode {:?}, planner {:?}, max batch {}, sessions {}",
        r.mode, r.planner, r.max_batch, r.sessions
    );
    let rows = [
        ("shared weights", r.shared_weights_bytes),
        ("inputs/targets", r.input_bytes),
        ("outputs", r.output_bytes),
        ("activations", r.activation_bytes),
        ("retained for backward", r.retained_for_backward_bytes),
        ("activation gradients", r.gradient_bytes),
        ("parameter gradients", r.param_grad_bytes),
        ("optimizer state", r.optimizer_state_bytes),
        ("kernel scratch", r.kernel_scratch_bytes),
        ("alignment padding", r.alignment_padding_bytes),
        ("sum of buffers (no reuse)", r.sum_of_buffers_bytes),
        ("arena per session (after reuse)", r.arena_bytes_per_session),
        ("TOTAL planned tensor bytes", r.total_planned_bytes),
    ];
    for (k, v) in rows {
        let _ = writeln!(t, "  {k:<34} {:>12}  ({v} B)", mib(v));
    }
    let _ = writeln!(t, "  not covered: {}", r.excluded.join("; "));
    t
}

pub fn version(long: &str, revision: &str, target: &str) -> CmdResult {
    let pkg = boundml_artifact::manifest::PACKAGE_FORMAT_VERSION;
    let ck = boundml_train::checkpoint::CHECKPOINT_VERSION;
    ok(
        json!({"version": boundml_core::VERSION, "revision": revision, "target": target,
            "package_format_version": pkg, "checkpoint_format_version": ck}),
        format!("boundml {long}\npackage format v{pkg}, checkpoint format v{ck}"),
    )
}

pub fn init(dir: &Path) -> CmdResult {
    let files = crate::init::init(dir)?;
    let text = format!(
        "created example in {}:\n  {}\nnext: boundml validate {}",
        dir.display(),
        files.join("\n  "),
        files[0]
    );
    ok(json!({"created": files}), text)
}

pub fn split(csv: &Path, out: &Path, valid: f64, test: f64, seed: u64) -> CmdResult {
    let c = boundml_data::split_csv(csv, out, valid, test, seed).map_err(s)?;
    ok(
        json!(c),
        format!(
            "train {} / valid {} / test {} rows written to {}",
            c.train,
            c.valid,
            c.test,
            out.display()
        ),
    )
}

pub fn validate(config: &Path) -> CmdResult {
    let (cfg, base) = TrainConfig::load(config).map_err(s)?;
    let schema = cfg.schema();
    let mut files = vec![];
    for (role, p) in std::iter::once(("train", Some(&cfg.data.train)))
        .chain(std::iter::once(("valid", cfg.data.valid.as_ref())))
    {
        if let Some(p) = p {
            let path = base.join(p);
            CsvSource::open(&path, &schema, true).map_err(s)?;
            files.push(json!({"role": role, "path": path.display().to_string(), "header": "ok"}));
        }
    }
    let (_, _, train) = boundml_train::plan_training(&cfg).map_err(s)?;
    if let Some(b) = cfg.train_budget().map_err(s)? {
        train.check_budget(b).map_err(s)?;
    }
    let infer = boundml_train::plan_inference(&cfg).map_err(s)?;
    if let Some(b) = cfg.deploy_budget().map_err(s)? {
        infer.check_budget(b).map_err(s)?;
    }
    let text = format!(
        "{}: valid\n  task {:?}, {} features, data headers ok ({} file(s))\n  training plan {} (budget {}), inference plan {} (budget {})",
        config.display(),
        schema.task(),
        schema.features.len(),
        files.len(),
        mib(train.total_planned_bytes),
        cfg.training.tensor_budget.as_deref().unwrap_or("none"),
        mib(infer.total_planned_bytes),
        cfg.deploy.tensor_budget.as_deref().unwrap_or("none"),
    );
    ok(
        json!({"valid": true, "task": schema.task(), "files": files,
        "train_planned_bytes": train.total_planned_bytes, "inference_planned_bytes": infer.total_planned_bytes}),
        text,
    )
}

pub struct PlanArgs<'a> {
    pub config: &'a Path,
    pub mode: Mode,
    pub budget: Option<&'a str>,
    pub batch: Option<usize>,
    pub planner: Option<PlannerKind>,
    pub sessions: Option<usize>,
}

pub fn plan(a: PlanArgs<'_>) -> CmdResult {
    let (mut cfg, _) = TrainConfig::load(a.config).map_err(s)?;
    if let Some(p) = a.planner {
        cfg.training.planner = p;
        cfg.deploy.planner = p;
    }
    if let Some(n) = a.sessions {
        cfg.deploy.sessions = n;
    }
    let (report, cfg_budget) = match a.mode {
        Mode::Training => {
            if let Some(b) = a.batch {
                cfg.training.batch_size = b;
            }
            (
                boundml_train::plan_training(&cfg).map_err(s)?.2,
                cfg.train_budget().map_err(s)?,
            )
        }
        Mode::Inference => {
            if let Some(b) = a.batch {
                cfg.deploy.max_batch = b;
            }
            (
                boundml_train::plan_inference(&cfg).map_err(s)?,
                cfg.deploy_budget().map_err(s)?,
            )
        }
    };
    let budget = match a.budget {
        Some(b) => Some(parse_bytes(b).map_err(s)?),
        None => cfg_budget,
    };
    let mut text = format!(
        "tensor memory plan for {}\n{}",
        a.config.display(),
        report_text(&report)
    );
    if let Some(b) = budget {
        report.check_budget(b).map_err(s)?;
        let _ = write!(text, "  within budget of {} ({b} B)", mib(b));
    }
    ok(
        json!({"report": report, "budget_bytes": budget, "within_budget": budget.map(|b| report.total_planned_bytes <= b)}),
        text,
    )
}

pub fn train(
    config: &Path,
    checkpoint: &Path,
    resume: bool,
    max_steps: Option<u64>,
    quiet: bool,
) -> CmdResult {
    let (cfg, base) = TrainConfig::load(config).map_err(s)?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    // A second Ctrl-C (or a handler already installed) falls back to default behaviour.
    let _ = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst));
    let should_stop = move || stop.load(Ordering::SeqCst);
    let started = Instant::now();
    let mut on_event = |e: &TrainEvent| {
        if quiet {
            return;
        }
        match e {
            TrainEvent::Planned(r) => eprintln!(
                "planned training tensors: {} total",
                mib(r.total_planned_bytes)
            ),
            TrainEvent::Resumed { epoch, global_step } => {
                eprintln!("resumed at epoch {epoch}, step {global_step}")
            }
            TrainEvent::Epoch(r) => {
                let v: Vec<String> = r
                    .valid
                    .iter()
                    .filter(|(k, _)| *k != "rows")
                    .map(|(k, v)| format!("{k}={v:.4}"))
                    .collect();
                eprintln!(
                    "epoch {:>3}  train_loss={:.5}  {}",
                    r.epoch + 1,
                    r.train_loss,
                    v.join(" ")
                );
            }
            TrainEvent::Checkpoint { .. } => {}
            TrainEvent::Stopped { global_step } => {
                eprintln!("stopped at step {global_step}; checkpoint saved")
            }
        }
    };
    let out = boundml_train::train(
        &cfg,
        &base,
        TrainOptions {
            checkpoint_dir: checkpoint.to_path_buf(),
            resume,
            max_steps,
            should_stop: &should_stop,
            on_event: &mut on_event,
        },
    )
    .map_err(s)?;
    let secs = started.elapsed().as_secs_f64();
    let last = out.history.last();
    let text = format!(
        "{} after {} steps in {secs:.2}s; checkpoint: {}{}",
        if out.completed {
            "training complete"
        } else {
            "training stopped"
        },
        out.global_step,
        checkpoint.display(),
        last.map(|l| format!(
            "\nfinal epoch: train_loss={:.5}{}",
            l.train_loss,
            fmt_metrics(&l.valid)
        ))
        .unwrap_or_default()
    );
    ok(
        json!({"completed": out.completed, "global_step": out.global_step, "seconds": secs, "history": out.history,
        "planned_tensor_bytes": out.report.total_planned_bytes}),
        text,
    )
}

pub fn evaluate(path: &Path, data: &Path) -> CmdResult {
    let e = boundml_train::evaluate(path, data).map_err(s)?;
    let mut text = format!("metrics on {}:\n", data.display());
    for (k, v) in &e.model {
        let b = e
            .baseline
            .as_ref()
            .and_then(|b| b.get(k))
            .map(|b| format!("   (constant baseline {b:.4})"))
            .unwrap_or_default();
        let _ = writeln!(text, "  {k:<10} {v:>10.4}{b}");
    }
    ok(
        json!({"model": e.model, "constant_baseline": e.baseline}),
        text,
    )
}

pub fn pack(checkpoint: &Path, out: &Path, name: Option<&str>) -> CmdResult {
    let pkg = boundml_train::pack(checkpoint, out, name).map_err(s)?;
    let m = pkg.manifest();
    let n = m.test_vectors.as_ref().map_or(0, |t| t.inputs.len());
    ok(
        json!({"package": out.display().to_string(), "name": m.name, "test_vectors": n}),
        format!(
            "packed '{}' -> {} ({n} synthetic self-test vectors, verified after write)",
            m.name,
            out.display()
        ),
    )
}

pub fn inspect(pkg: &Path) -> CmdResult {
    let p = Package::load(pkg).map_err(s)?;
    let m = p.manifest();
    let report = p.memory_report(m.bounds.sessions).map_err(s)?;
    let mut text = String::new();
    let _ = writeln!(
        text,
        "{} — {} v{} (min runtime {})",
        m.name, m.format, m.format_version, m.min_runtime_version
    );
    let _ = writeln!(
        text,
        "task: {:?}; outputs: {}",
        m.task,
        m.output_names().join(", ")
    );
    let _ = writeln!(
        text,
        "features ({}): {}",
        m.schema.features.len(),
        m.schema.features.join(", ")
    );
    let _ = writeln!(
        text,
        "label: {}; missing: {:?}; extra columns: {:?}; normalize: {:?}",
        m.schema.label, m.schema.missing, m.schema.extra_columns, m.preprocessing.normalize
    );
    let _ = writeln!(
        text,
        "graph: {} nodes, {} parameters",
        m.graph.nodes.len(),
        p.model().graph().param_count()
    );
    let _ = writeln!(
        text,
        "bounds: max_batch {}, sessions {}, budget {:?}",
        m.bounds.max_batch, m.bounds.sessions, m.bounds.tensor_budget_bytes
    );
    let _ = writeln!(
        text,
        "provenance: source {}, created by {}, seed {:?}",
        m.provenance.source, m.provenance.created_by, m.provenance.seed
    );
    for (k, v) in &m.provenance.metrics {
        if k.starts_with("valid_") || k == "final_train_loss" {
            let _ = writeln!(text, "  {k} = {v:.5}");
        }
    }
    let _ = writeln!(
        text,
        "test vectors: {} ({})",
        m.test_vectors.as_ref().map_or(0, |t| t.inputs.len()),
        m.test_vectors
            .as_ref()
            .map_or("none", |t| t.origin.as_str())
    );
    let _ = write!(text, "planned inference memory:\n{}", report_text(&report));
    ok(json!({"manifest": m, "memory": report}), text)
}

pub fn verify(pkg: &Path) -> CmdResult {
    let p = Package::load(pkg).map_err(s)?;
    let r = p.verify().map_err(s)?;
    let text = if r.vectors == 0 {
        "integrity, version and consistency checks passed; no test vectors are bundled, so no outputs were checked".to_string()
    } else {
        format!(
            "integrity, version and consistency checks passed; self-test: {} vector(s), max abs error {:e} -> {}",
            r.vectors,
            r.max_abs_error,
            if r.passed { "PASS" } else { "FAIL" }
        )
    };
    Ok(Outcome {
        json: json!(r),
        text,
        exit: if r.passed {
            exit::OK
        } else {
            exit::CHECK_FAILED
        },
    })
}

pub fn predict(pkg: &Path, data: &Path, out: &Path, batch: usize, preview: usize) -> CmdResult {
    let p = Arc::new(Package::load(pkg).map_err(s)?);
    let r = p.verify().map_err(s)?;
    if !r.passed {
        return Err(format!(
            "BML-A007 self-test failed: package {} fails its test vectors",
            pkg.display()
        ));
    }
    let batch = batch.min(p.manifest().bounds.max_batch).max(1);
    let mut pred = p.predictor(batch).map_err(s)?;
    let m = p.manifest();
    let mut src = CsvSource::open(data, &m.schema, false).map_err(s)?;
    // Refuse to overwrite the input itself; then write to an exclusively reserved,
    // uniquely named temporary file next to `out` and rename only on success. A
    // failure removes just that temporary file and never touches the input or any
    // pre-existing file.
    if boundml_data::same_file(out, data) {
        return Err(format!(
            "BML-A001 I/O error on {}: --out must not be the input file",
            out.display()
        ));
    }
    let mut reserved = boundml_data::ReservedFile::next_to(out)
        .map_err(|e| format!("BML-A001 I/O error on {}: {e}", out.display()))?;
    let tmp = reserved.path().to_path_buf();
    let mut w = csv::Writer::from_writer(reserved.take_file());
    let names = m.output_names();
    let mut header = vec!["line".to_string()];
    header.extend(names.iter().cloned());
    if m.schema.classes().is_some() {
        header.push("predicted".into());
    }
    w.write_record(&header).map_err(s)?;
    let (nf, no) = (pred.n_features(), pred.output_cols());
    let mut raw = vec![None; batch * nf];
    let mut lines = vec![0u64; batch];
    let mut outv = vec![0.0f32; batch * no];
    let mut row = RawRow::default();
    let mut total = 0u64;
    let mut shown: Vec<Vec<String>> = Vec::new();
    let started = Instant::now();
    loop {
        let mut n = 0;
        while n < batch && src.read_row(&mut row).map_err(s)? {
            raw[n * nf..(n + 1) * nf].copy_from_slice(&row.features);
            lines[n] = row.line;
            n += 1;
        }
        if n == 0 {
            break;
        }
        pred.predict_raw(&raw[..n * nf], n, &mut outv[..n * no])
            .map_err(|e| format!("{e} (batch starting at line {})", lines[0]))?;
        for r in 0..n {
            let o = &outv[r * no..(r + 1) * no];
            let mut rec = vec![lines[r].to_string()];
            rec.extend(o.iter().map(|v| v.to_string()));
            if let Some(c) = m.schema.classes() {
                rec.push(c[boundml_artifact::compare::decision(&m.task, o)].clone());
            }
            if shown.len() < preview {
                shown.push(rec.clone());
            }
            w.write_record(&rec).map_err(s)?;
        }
        total += n as u64;
        if n < batch {
            break;
        }
    }
    w.flush()
        .map_err(|e| format!("BML-A001 I/O error on {}: {e}", tmp.display()))?;
    drop(w);
    reserved
        .commit(out)
        .map_err(|e| format!("BML-A001 I/O error on {}: {e}", out.display()))?;
    let secs = started.elapsed().as_secs_f64();
    let mut text = format!(
        "wrote {total} predictions to {} in {secs:.3}s",
        out.display()
    );
    if !shown.is_empty() {
        // Right-aligned columns: header first, then the first `preview` rows.
        let rows: Vec<Vec<String>> = std::iter::once(header.clone())
            .chain(
                shown
                    .iter()
                    .map(|r| r.iter().map(|v| fmt_cell(v)).collect()),
            )
            .collect();
        let widths: Vec<usize> = (0..header.len())
            .map(|c| rows.iter().map(|r| r[c].len()).max().unwrap_or(0))
            .collect();
        for r in &rows {
            let cells: Vec<String> = r
                .iter()
                .zip(&widths)
                .map(|(v, w)| format!("{v:>w$}"))
                .collect();
            let _ = write!(text, "\n  {}", cells.join("  "));
        }
    }
    ok(
        json!({"rows": total, "out": out.display().to_string(), "seconds": secs, "preview": shown}),
        text,
    )
}

/// `  valid: rmse=0.1913 r2=0.9954 ...` (rows omitted), or empty without validation.
fn fmt_metrics(m: &std::collections::BTreeMap<String, f64>) -> String {
    let parts: Vec<String> = m
        .iter()
        .filter(|(k, _)| k.as_str() != "rows")
        .map(|(k, v)| format!("{k}={v:.4}"))
        .collect();
    if parts.is_empty() {
        String::new()
    } else {
        format!("  valid: {}", parts.join(" "))
    }
}

/// Shorten long float strings for the human-readable preview table.
fn fmt_cell(v: &str) -> String {
    match v.parse::<f64>() {
        Ok(x) if v.contains('.') => format!("{x:.4}"),
        _ => v.to_string(),
    }
}

pub fn compare_cmd(current: &Path, candidate: &Path, opts: CompareOptions) -> CmdResult {
    let a = Arc::new(Package::load(current).map_err(s)?);
    let b = Arc::new(Package::load(candidate).map_err(s)?);
    let r = compare(&a, &b, &opts).map_err(s)?;
    let mut text = format!("verdict: {:?}\n", r.verdict);
    for c in &r.changes {
        let _ = writeln!(text, "  [{:?}] {}: {}", c.severity, c.area, c.message);
    }
    let _ = writeln!(
        text,
        "  planned tensor bytes: {} -> {}",
        r.memory.current_total_planned_bytes, r.memory.candidate_total_planned_bytes
    );
    if let Some(n) = &r.numeric {
        let _ = writeln!(
            text,
            "  on {} rows: max |diff| {:e}, mean |diff| {:e}{}",
            n.rows,
            n.max_abs_diff,
            n.mean_abs_diff,
            n.decision_agreement
                .map(|a| format!(", decision agreement {:.2}%", a * 100.0))
                .unwrap_or_default()
        );
        if let (Some(name), Some(x), Some(y)) =
            (&n.metric_name, n.current_metric, n.candidate_metric)
        {
            let _ = writeln!(text, "  {name}: current {x:.4}, candidate {y:.4}");
        }
    }
    let code = if r.verdict == Verdict::Breaking {
        exit::CHECK_FAILED
    } else {
        exit::OK
    };
    Ok(Outcome {
        json: json!(r),
        text,
        exit: code,
    })
}

pub fn import(bundle: &Path, out: &Path, max_batch: usize) -> CmdResult {
    let r = crate::import::import(bundle, out, max_batch)?;
    ok(
        json!({"package": out.display().to_string(), "reference_vectors": r.vectors, "max_abs_error": r.max_abs_error}),
        format!(
            "imported {} -> {}; matches source-framework outputs on {} reference rows (max abs error {:e})",
            bundle.display(),
            out.display(),
            r.vectors,
            r.max_abs_error
        ),
    )
}

pub fn serve(pkg: &Path, bind: &str, cfg: ServeConfig) -> CmdResult {
    let p = Arc::new(Package::load(pkg).map_err(s)?);
    let r = p.verify().map_err(s)?;
    if !r.passed {
        return Err(format!(
            "BML-A007 self-test failed: {} fails its test vectors",
            pkg.display()
        ));
    }
    let state = AppState::new(Arc::clone(&p), cfg.clone())
        .map_err(|e| format!("BML-S005 invalid server configuration: {e}"))?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(cfg.workers.max(1))
        .enable_all()
        .build()
        .map_err(s)?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(bind).await.map_err(|e| format!("BML-S006 cannot bind {bind}: {e}"))?;
        let addr = listener.local_addr().map_err(s)?;
        eprintln!("serving '{}' on http://{addr} (workers {}, queue {}, max rows {}, timeout {:?}); Ctrl-C to stop",
            p.manifest().name, cfg.workers, cfg.queue, cfg.max_rows, cfg.timeout);
        let st = Arc::clone(&state);
        crate::serve::run(state, listener, async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("shutdown requested; draining in-flight requests");
        })
        .await
        .map_err(s)?;
        ok(json!({"served": st.served.load(Ordering::Relaxed), "rejected_overload": st.rejected_overload.load(Ordering::Relaxed)}),
            format!("stopped; served {} request(s), rejected {} for overload", st.served.load(Ordering::Relaxed), st.rejected_overload.load(Ordering::Relaxed)))
    })
}

/// Default cap on rows per HTTP request when `--max-rows` is not given.
pub const DEFAULT_MAX_ROWS: usize = 256;

pub fn serve_config(
    workers: Option<usize>,
    queue: usize,
    max_body: &str,
    max_rows: usize,
    timeout_ms: u64,
    pkg_sessions: usize,
) -> Result<ServeConfig, String> {
    Ok(ServeConfig {
        workers: workers.unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism().map_or(1, usize::from);
            pkg_sessions.min(cpus)
        }),
        queue,
        max_body_bytes: usize::try_from(parse_bytes(max_body).map_err(s)?).map_err(s)?,
        max_rows,
        timeout: Duration::from_millis(timeout_ms),
        simulate_compute_delay: Duration::ZERO,
    })
}
