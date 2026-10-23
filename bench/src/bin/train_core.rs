//! BoundML core training on in-memory data, the like-for-like counterpart of
//! `bench/py_train.py`. It uses the same
//! config, standardization, initialization, output-bias init, AdamW and a seeded
//! per-epoch shuffle, but no CSV streaming or per-epoch validation (which `boundml
//! train` does). Only the training loop is timed.
//!
//! For the controlled comparison, `train_core` writes `<control.json>` before timing:
//! the standardized train/test tensors, the initial parameters (after output-bias
//! initialization) and every epoch's row permutation. `py_train.py --control` loads
//! it, so both trainers start from the same weights and visit the same batches in the
//! same order.
//!
//! Usage: `train_core <config.toml> <test.csv> [<control.json>]`

use boundml_core::{Executor, InferenceSession, Mode, Model, Optimizer, Rng, Task};
use boundml_data::{BatchReader, Normalize, Preprocessor};
use boundml_train::TrainConfig;
use boundml_train::metrics::{LabelStats, MetricAcc};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn load(path: &Path, cfg: &TrainConfig, pre: &Preprocessor) -> (Vec<f32>, Vec<f32>) {
    let schema = cfg.schema();
    let nf = schema.features.len();
    let mut r = BatchReader::open(path, &schema, pre, 1, Rng::seed(0), true).unwrap();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    let (mut bx, mut by, mut bl) = (vec![0.0; 256 * nf], vec![0.0; 256], vec![0u64; 256]);
    loop {
        let n = r.next_batch(256, &mut bx, &mut by, &mut bl).unwrap();
        if n == 0 {
            break;
        }
        x.extend_from_slice(&bx[..n * nf]);
        y.extend_from_slice(&by[..n]);
    }
    (x, y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (cfg, base) = TrainConfig::load(Path::new(&a[1]))?;
    let schema = cfg.schema();
    let task: Task = schema.task();
    let nf = schema.features.len();
    let train_path = base.join(&cfg.data.train);
    let pre = Preprocessor::fit(&train_path, &schema, Normalize::Standardize)?;
    let (xtr, ytr) = load(&train_path, &cfg, &pre);
    let (xte, yte) = load(Path::new(&a[2]), &cfg, &pre);
    let spec = cfg.dense_spec();
    let graph = Arc::new(spec.graph()?);
    let mut params = spec.init_params(&graph, &mut Rng::seed(cfg.seed));
    let mut labels = LabelStats::new(&task);
    ytr.iter().for_each(|y| labels.add(*y));
    boundml_train::trainer::init_output_bias(&graph, &mut params, &labels, &task);
    let n = ytr.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = Rng::seed(cfg.seed);
    let perms: Vec<Vec<usize>> = (0..cfg.training.epochs)
        .map(|_| {
            for i in (1..n).rev() {
                idx.swap(i, rng.below(i as u64 + 1) as usize);
            }
            idx.clone()
        })
        .collect();
    if let Some(control) = a.get(3) {
        let init: serde_json::Map<String, serde_json::Value> = graph
            .params()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let at = graph.param_offset(i);
                let values = &params[at..at + graph.param_len(i)];
                (
                    p.name.clone(),
                    json!({"rows": p.rows, "cols": p.cols, "values": values}),
                )
            })
            .collect();
        let doc = json!({"n_features": nf, "outputs": task.output_cols(),
            "x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte,
            "init": init, "perms": perms, "batch_size": cfg.training.batch_size,
            "note": "weights are [in, out] row-major; biases [1, out]"});
        std::fs::write(control, serde_json::to_vec(&doc)?)?;
    }
    let bs = cfg.training.batch_size;
    let mut ex = Executor::new(Arc::clone(&graph), Mode::Training, bs, cfg.training.planner)?;
    let mut opt = Optimizer::new(cfg.training.optimizer.clone(), &graph)?;
    let (mut bx, mut by) = (vec![0.0f32; bs * nf], vec![0.0f32; bs]);
    let started = Instant::now();
    for perm in &perms {
        for chunk in perm.chunks(bs) {
            let m = chunk.len();
            for (k, &r) in chunk.iter().enumerate() {
                bx[k * nf..(k + 1) * nf].copy_from_slice(&xtr[r * nf..(r + 1) * nf]);
                by[k] = ytr[r];
            }
            ex.forward_backward(&params, &bx[..m * nf], &by[..m], m)?;
            opt.step(&mut params, ex.param_grads())?;
        }
    }
    let secs = started.elapsed().as_secs_f64();
    let model = Arc::new(Model::new((*graph).clone(), params)?);
    let mut s = InferenceSession::new(model, 1)?;
    let no = task.output_cols();
    let mut out = vec![0.0f32; no];
    let mut acc = MetricAcc::new(task);
    for (r, y) in yte.iter().enumerate() {
        s.predict_into(&xte[r * nf..(r + 1) * nf], 1, &mut out)?;
        acc.add(&out, *y);
    }
    let m = acc.finish();
    let metric = if task == Task::Regression {
        json!({"rmse": m["rmse"]})
    } else {
        json!({"accuracy": m["accuracy"]})
    };
    println!(
        "{}",
        json!({"runtime": "boundml-core-in-memory", "threads": 1, "train_seconds": secs, "test": metric})
    );
    Ok(())
}
