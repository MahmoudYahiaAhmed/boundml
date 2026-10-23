//! Turn a training checkpoint into an inference package, and evaluate either.

use crate::checkpoint::Checkpoint;
use crate::error::{Result, TrainError};
use crate::metrics::LabelStats;
use crate::trainer::{config_sha256, evaluate_with_baseline};
use boundml_artifact::{Bounds, Package, PackageSpec, Provenance};
use boundml_core::{InferenceSession, Model, Rng, Task};
use boundml_data::{MissingPolicy, Preprocessor};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

pub const SYNTHETIC_VECTORS: usize = 16;

/// Synthetic raw inputs spanning ±2 training standard deviations around the mean,
/// plus one row with all features missing when the policy imputes.
pub fn synthetic_inputs(pre: &Preprocessor, seed: u64) -> Vec<Vec<Option<f64>>> {
    let mut rng = Rng::seed(seed ^ 0x5EED_7E57);
    let mut rows: Vec<Vec<Option<f64>>> = (0..SYNTHETIC_VECTORS)
        .map(|_| {
            pre.features
                .iter()
                .map(|f| Some(f.mean + f.std * f64::from(rng.uniform(-2.0, 2.0))))
                .collect()
        })
        .collect();
    if pre.missing == MissingPolicy::ImputeMean {
        rows.push(vec![None; pre.features.len()]);
    }
    rows
}

pub fn pack(checkpoint_dir: &Path, out: &Path, name: Option<&str>) -> Result<Package> {
    let ck = Checkpoint::load(checkpoint_dir)?;
    let s = &ck.state;
    let cfg = &s.config;
    let mut metrics = BTreeMap::new();
    if let Some(last) = s.history.last() {
        metrics.insert("final_train_loss".into(), last.train_loss);
        for (k, v) in &last.valid {
            metrics.insert(format!("valid_{k}"), *v);
        }
    }
    metrics.insert("train_label_mean".into(), s.label_stats.mean);
    metrics.insert("train_rows".into(), s.label_stats.rows as f64);
    for (i, c) in s.label_stats.class_counts.iter().enumerate() {
        metrics.insert(format!("train_class_count.{i}"), *c as f64);
    }
    let notes = (!s.progress.completed).then(|| {
        format!(
            "packed from an incomplete run (epoch {} of {})",
            s.progress.epoch, cfg.training.epochs
        )
    });
    let spec = PackageSpec {
        name: name.unwrap_or(&cfg.name).to_string(),
        schema: s.schema.clone(),
        preprocessing: s.preprocessing.clone(),
        graph: s.spec.graph_def()?,
        weights: ck.params.clone(),
        bounds: Bounds {
            max_batch: cfg.deploy.max_batch,
            tensor_budget_bytes: cfg.deploy_budget()?,
            sessions: cfg.deploy.sessions,
            planner: cfg.deploy.planner,
        },
        provenance: Provenance {
            source: "boundml-train".into(),
            seed: Some(cfg.seed),
            training_config_sha256: Some(config_sha256(cfg)),
            data: s.data.clone(),
            metrics,
            notes,
            ..Default::default()
        },
        test_inputs: synthetic_inputs(&s.preprocessing, cfg.seed),
        test_expected: None,
        test_origin: "synthetic".into(),
        test_atol: 1e-6,
        test_rtol: 1e-5,
    };
    Ok(Package::save(out, spec)?)
}

/// Reconstruct label statistics recorded in a package's provenance (for baselines).
pub fn package_label_stats(pkg: &Package) -> Option<LabelStats> {
    let m = &pkg.manifest().provenance.metrics;
    let rows = *m.get("train_rows")? as u64;
    let k = match pkg.manifest().task {
        Task::Regression => 0,
        Task::Binary => 2,
        Task::Multiclass { classes } => classes,
    };
    let class_counts = (0..k)
        .map(|i| m.get(&format!("train_class_count.{i}")).map(|v| *v as u64))
        .collect::<Option<Vec<u64>>>()?;
    Some(LabelStats {
        rows,
        mean: *m.get("train_label_mean")?,
        class_counts,
    })
}

pub struct Evaluation {
    pub model: BTreeMap<String, f64>,
    pub baseline: Option<BTreeMap<String, f64>>,
}

const EVAL_BATCH: usize = 256;

/// Evaluate a checkpoint directory or a package directory on a labelled CSV.
pub fn evaluate(path: &Path, data: &Path) -> Result<Evaluation> {
    if path.join("manifest.json").exists() {
        let pkg = Arc::new(Package::load(path)?);
        let batch = EVAL_BATCH.min(pkg.manifest().bounds.max_batch);
        let mut p = pkg.predictor(batch)?;
        let labels = package_label_stats(&pkg);
        let schema = pkg.manifest().schema.clone();
        let stats = labels
            .clone()
            .unwrap_or_else(|| LabelStats::new(&schema.task()));
        let (model, base) = evaluate_with_baseline(
            &mut |raw, n, out| p.predict_raw(raw, n, out).map_err(TrainError::from),
            data,
            &schema,
            &stats,
            batch,
        )?;
        return Ok(Evaluation {
            model,
            baseline: labels.map(|_| base),
        });
    }
    let ck = Checkpoint::load(path)?;
    let s = &ck.state;
    let model = Arc::new(Model::new(s.spec.graph()?, ck.params.clone())?);
    let mut session = InferenceSession::new(model, EVAL_BATCH)?;
    let nf = s.schema.features.len();
    let mut x = vec![0.0f32; EVAL_BATCH * nf];
    let pre = &s.preprocessing;
    let (model, base) = evaluate_with_baseline(
        &mut |raw, n, out| {
            for r in 0..n {
                pre.apply(&raw[r * nf..(r + 1) * nf], &mut x[r * nf..(r + 1) * nf])
                    .map_err(|i| {
                        TrainError::Config(format!(
                            "row {r}: feature '{}' missing or out of range",
                            pre.features[i].name
                        ))
                    })?;
            }
            session.predict_into(&x[..n * nf], n, out)?;
            Ok(())
        },
        data,
        &s.schema,
        &s.label_stats,
        EVAL_BATCH,
    )?;
    Ok(Evaluation {
        model,
        baseline: Some(base),
    })
}
