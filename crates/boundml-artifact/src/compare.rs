//! Compare a candidate package with the current one before an update.
//!
//! Classification of changes:
//! * **breaking** — clients or deployments that work with the current package may
//!   fail or silently change meaning: task/label/class changes, removed features,
//!   output width change, lower `max_batch`, candidate exceeding the current tensor
//!   budget, stricter missing/extra-column policy, or numerical drift above an
//!   explicit threshold.
//! * **warning** — accepted by the same clients but worth review (new features that
//!   would be treated as missing, refitted preprocessing, architecture change).
//! * **info** — reordering of features (inputs are mapped by name) etc.

use crate::error::{ArtifactError, Result};
use crate::package::Package;
use boundml_core::Task;
use boundml_data::{CsvSource, ExtraColumns, MissingPolicy, RawRow};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct CompareOptions {
    /// CSV evaluated by both packages (each with its own fitted preprocessing).
    pub data: Option<PathBuf>,
    /// Maximum tolerated absolute output difference; exceeding it is breaking.
    pub max_abs_diff: Option<f32>,
    /// Optional cap on rows read from `data`.
    pub max_rows: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Breaking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Identical,
    Compatible,
    Breaking,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Change {
    pub area: String,
    pub severity: Severity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryDiff {
    pub current_total_planned_bytes: u64,
    pub candidate_total_planned_bytes: u64,
    pub current_arena_bytes: u64,
    pub candidate_arena_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericDiff {
    pub rows: u64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f64,
    /// Fraction of rows with the same predicted class (classification only).
    pub decision_agreement: Option<f64>,
    pub metric_name: Option<String>,
    pub current_metric: Option<f64>,
    pub candidate_metric: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareReport {
    pub verdict: Verdict,
    pub changes: Vec<Change>,
    pub memory: MemoryDiff,
    pub numeric: Option<NumericDiff>,
}

pub fn compare(
    current: &Arc<Package>,
    candidate: &Arc<Package>,
    opts: &CompareOptions,
) -> Result<CompareReport> {
    let (a, b) = (current.manifest(), candidate.manifest());
    let mut changes = Vec::new();
    let mut add = |area: &str, severity: Severity, message: String| {
        changes.push(Change {
            area: area.into(),
            severity,
            message,
        })
    };

    // Task and labels.
    if a.task != b.task {
        add(
            "task",
            Severity::Breaking,
            format!("task changed from {:?} to {:?}", a.task, b.task),
        );
    }
    if a.schema.label != b.schema.label {
        add(
            "schema",
            Severity::Breaking,
            format!(
                "label column renamed '{}' -> '{}'",
                a.schema.label, b.schema.label
            ),
        );
    }
    if a.schema.label_kind != b.schema.label_kind {
        add(
            "schema",
            Severity::Breaking,
            "label classes or label kind changed (output meaning differs)".into(),
        );
    }
    // Features.
    let removed: Vec<&String> = a
        .schema
        .features
        .iter()
        .filter(|f| !b.schema.features.contains(f))
        .collect();
    let added: Vec<&String> = b
        .schema
        .features
        .iter()
        .filter(|f| !a.schema.features.contains(f))
        .collect();
    if !removed.is_empty() {
        add(
            "schema",
            Severity::Warning,
            format!("candidate no longer uses feature(s): {removed:?}"),
        );
    }
    if !added.is_empty() {
        let sev = if b.schema.missing == MissingPolicy::Reject {
            Severity::Breaking
        } else {
            Severity::Warning
        };
        add(
            "schema",
            sev,
            format!(
                "candidate requires new feature(s) {added:?}; current clients do not send them (missing policy: {:?})",
                b.schema.missing
            ),
        );
    }
    if removed.is_empty() && added.is_empty() && a.schema.features != b.schema.features {
        add(
            "schema",
            Severity::Info,
            "feature order changed (inputs are mapped by name, so clients are unaffected)".into(),
        );
    }
    if a.schema.missing == MissingPolicy::ImputeMean && b.schema.missing == MissingPolicy::Reject {
        add(
            "schema",
            Severity::Breaking,
            "missing-value policy became stricter (impute_mean -> reject)".into(),
        );
    } else if a.schema.missing != b.schema.missing {
        add(
            "schema",
            Severity::Info,
            format!("missing-value policy changed to {:?}", b.schema.missing),
        );
    }
    if a.schema.extra_columns == ExtraColumns::Ignore
        && b.schema.extra_columns == ExtraColumns::Reject
    {
        add(
            "schema",
            Severity::Breaking,
            "extra-column policy became stricter (ignore -> reject)".into(),
        );
    }
    // Preprocessing.
    if a.preprocessing.normalize != b.preprocessing.normalize {
        add(
            "preprocessing",
            Severity::Warning,
            format!("normalization changed to {:?}", b.preprocessing.normalize),
        );
    }
    for fb in &b.preprocessing.features {
        if let Some(fa) = a.preprocessing.features.iter().find(|f| f.name == fb.name) {
            let dm = (fb.mean - fa.mean).abs() / fa.std.max(1e-12);
            let rs = fb.std / fa.std;
            if dm > 1e-6 || (rs - 1.0).abs() > 1e-6 {
                add(
                    "preprocessing",
                    Severity::Warning,
                    format!(
                        "feature '{}' refitted: mean shift {:.4} std-units, std ratio {:.4}",
                        fb.name, dm, rs
                    ),
                );
            }
        }
    }
    // Graph and bounds.
    if a.graph != b.graph {
        let pa: usize = a.tensors.iter().map(|t| t.shape[0] * t.shape[1]).sum();
        let pb: usize = b.tensors.iter().map(|t| t.shape[0] * t.shape[1]).sum();
        add(
            "graph",
            Severity::Warning,
            format!(
                "model graph changed ({} -> {} nodes, {pa} -> {pb} parameters)",
                a.graph.nodes.len(),
                b.graph.nodes.len()
            ),
        );
    }
    if a.task.output_cols() != b.task.output_cols() {
        add(
            "graph",
            Severity::Breaking,
            format!(
                "output width {} -> {}",
                a.task.output_cols(),
                b.task.output_cols()
            ),
        );
    }
    if b.bounds.max_batch < a.bounds.max_batch {
        add(
            "bounds",
            Severity::Breaking,
            format!(
                "max_batch reduced {} -> {}; larger client batches would be rejected",
                a.bounds.max_batch, b.bounds.max_batch
            ),
        );
    }
    // Memory.
    let ra = current.memory_report(a.bounds.sessions)?;
    let rb = candidate.memory_report(a.bounds.sessions)?;
    if let Some(budget) = a.bounds.tensor_budget_bytes
        && rb.total_planned_bytes > budget
    {
        add(
            "memory",
            Severity::Breaking,
            format!(
                "candidate needs {} planned bytes for {} session(s); current deployment budget is {budget}",
                rb.total_planned_bytes, a.bounds.sessions
            ),
        );
    }
    if rb.total_planned_bytes != ra.total_planned_bytes {
        add(
            "memory",
            Severity::Info,
            format!(
                "planned tensor bytes {} -> {}",
                ra.total_planned_bytes, rb.total_planned_bytes
            ),
        );
    }
    let weights_equal = current.model().weights() == candidate.model().weights();
    if a.graph == b.graph && !weights_equal {
        add("weights", Severity::Info, "weights differ".into());
    }

    let numeric = match &opts.data {
        Some(path) => Some(numeric_diff(current, candidate, path, opts.max_rows)?),
        None => None,
    };
    if let (Some(n), Some(t)) = (&numeric, opts.max_abs_diff)
        && n.max_abs_diff > t
    {
        changes.push(Change {
            area: "numeric".into(),
            severity: Severity::Breaking,
            message: format!(
                "max |output difference| {} exceeds threshold {t}",
                n.max_abs_diff
            ),
        });
    }
    let worst = changes.iter().map(|c| c.severity).max();
    let identical = changes.is_empty()
        && weights_equal
        && a.schema == b.schema
        && a.preprocessing == b.preprocessing
        && numeric.as_ref().is_none_or(|n| n.max_abs_diff == 0.0);
    let verdict = match worst {
        Some(Severity::Breaking) => Verdict::Breaking,
        _ if identical => Verdict::Identical,
        _ => Verdict::Compatible,
    };
    Ok(CompareReport {
        verdict,
        changes,
        memory: MemoryDiff {
            current_total_planned_bytes: ra.total_planned_bytes,
            candidate_total_planned_bytes: rb.total_planned_bytes,
            current_arena_bytes: ra.arena_bytes_per_session,
            candidate_arena_bytes: rb.arena_bytes_per_session,
        },
        numeric,
    })
}

const CHUNK: usize = 256;

fn numeric_diff(
    a: &Arc<Package>,
    b: &Arc<Package>,
    path: &std::path::Path,
    max_rows: Option<u64>,
) -> Result<NumericDiff> {
    let (ma, mb) = (a.manifest(), b.manifest());
    if ma.task.output_cols() != mb.task.output_cols() {
        return Err(ArtifactError::Inconsistent(
            "cannot compare outputs numerically: output widths differ".into(),
        ));
    }
    let mut sa = CsvSource::open(path, &ma.schema, false)?;
    let mut sb = CsvSource::open(path, &mb.schema, false)?;
    let labels = sa.has_label() && ma.schema.label_kind == mb.schema.label_kind;
    let batch = CHUNK.min(ma.bounds.max_batch).min(mb.bounds.max_batch);
    let mut pa = a.predictor(batch)?;
    let mut pb = b.predictor(batch)?;
    let (nfa, nfb, no) = (pa.n_features(), pb.n_features(), ma.task.output_cols());
    let mut raw_a = vec![None; batch * nfa];
    let mut raw_b = vec![None; batch * nfb];
    let mut y = vec![0.0f32; batch];
    let (mut oa, mut ob) = (vec![0.0f32; batch * no], vec![0.0f32; batch * no]);
    let (mut ra, mut rb) = (RawRow::default(), RawRow::default());
    let (mut rows, mut max_abs, mut sum_abs) = (0u64, 0.0f32, 0.0f64);
    let (mut agree, mut score_a, mut score_b) = (0u64, 0.0f64, 0.0f64);
    loop {
        let mut n = 0;
        while n < batch && max_rows.is_none_or(|m| rows + (n as u64) < m) {
            let more_a = sa.read_row(&mut ra)?;
            let more_b = sb.read_row(&mut rb)?;
            if !more_a || !more_b {
                break;
            }
            raw_a[n * nfa..(n + 1) * nfa].copy_from_slice(&ra.features);
            raw_b[n * nfb..(n + 1) * nfb].copy_from_slice(&rb.features);
            y[n] = ra.label.unwrap_or(0.0);
            n += 1;
        }
        if n == 0 {
            break;
        }
        pa.predict_raw(&raw_a[..n * nfa], n, &mut oa[..n * no])?;
        pb.predict_raw(&raw_b[..n * nfb], n, &mut ob[..n * no])?;
        for r in 0..n {
            let (xa, xb) = (&oa[r * no..(r + 1) * no], &ob[r * no..(r + 1) * no]);
            for (u, v) in xa.iter().zip(xb) {
                let d = (u - v).abs();
                max_abs = max_abs.max(d);
                sum_abs += d as f64;
            }
            let (da, db) = (decision(&ma.task, xa), decision(&ma.task, xb));
            if da == db {
                agree += 1;
            }
            if labels {
                match ma.task {
                    Task::Regression => {
                        score_a += ((xa[0] - y[r]) as f64).powi(2);
                        score_b += ((xb[0] - y[r]) as f64).powi(2);
                    }
                    _ => {
                        score_a += f64::from(u8::from(da == y[r] as usize));
                        score_b += f64::from(u8::from(db == y[r] as usize));
                    }
                }
            }
        }
        rows += n as u64;
    }
    if rows == 0 {
        return Err(ArtifactError::Data(boundml_data::DataError::Empty(
            format!("{} has no rows", path.display()),
        )));
    }
    let classification = ma.task != Task::Regression;
    let (metric_name, ca, cb) = if !labels {
        (None, None, None)
    } else if classification {
        (
            Some("accuracy".into()),
            Some(score_a / rows as f64),
            Some(score_b / rows as f64),
        )
    } else {
        (
            Some("rmse".into()),
            Some((score_a / rows as f64).sqrt()),
            Some((score_b / rows as f64).sqrt()),
        )
    };
    Ok(NumericDiff {
        rows,
        max_abs_diff: max_abs,
        mean_abs_diff: sum_abs / (rows as f64 * no as f64),
        decision_agreement: classification.then(|| agree as f64 / rows as f64),
        metric_name,
        current_metric: ca,
        candidate_metric: cb,
    })
}

/// Predicted class (classification) or 0 (regression).
pub fn decision(task: &Task, out: &[f32]) -> usize {
    match task {
        Task::Regression => 0,
        Task::Binary => usize::from(out[0] >= 0.5),
        Task::Multiclass { .. } => {
            out.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0
        }
    }
}
