//! The embeddable prediction API: raw (unpreprocessed) feature values in, model
//! outputs out, through exactly the fitted preprocessing stored in the package.

use crate::error::{ArtifactError, Result};
use crate::package::Package;
use boundml_core::InferenceSession;
use boundml_data::ExtraColumns;
use std::collections::HashMap;
use std::sync::Arc;

/// A mutable prediction context (one arena + input buffers). Not `Sync`; create one
/// per worker thread from a shared `Arc<Package>`.
pub struct Predictor {
    pkg: Arc<Package>,
    session: InferenceSession,
    x: Vec<f32>,
    index: HashMap<String, usize>,
    seen: Vec<bool>,
}

impl Predictor {
    /// `max_batch` must not exceed the package's declared `bounds.max_batch`. The
    /// package's tensor budget (if any) is checked before the arena is allocated.
    pub fn new(pkg: Arc<Package>, max_batch: usize) -> Result<Self> {
        let b = &pkg.manifest().bounds;
        if max_batch == 0 || max_batch > b.max_batch {
            return Err(ArtifactError::Core(
                boundml_core::CoreError::BatchOutOfRange {
                    got: max_batch,
                    max: b.max_batch,
                },
            ));
        }
        let session = InferenceSession::with_budget(
            Arc::clone(pkg.model()),
            max_batch,
            b.planner,
            b.tensor_budget_bytes,
            b.sessions,
        )?;
        let nf = pkg.manifest().schema.features.len();
        let index = pkg
            .manifest()
            .schema
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.clone(), i))
            .collect();
        Ok(Predictor {
            x: vec![0.0; max_batch * nf],
            seen: vec![false; nf],
            pkg,
            session,
            index,
        })
    }

    pub fn package(&self) -> &Arc<Package> {
        &self.pkg
    }
    pub fn max_batch(&self) -> usize {
        self.session.max_batch()
    }
    pub fn n_features(&self) -> usize {
        self.seen.len()
    }
    pub fn output_cols(&self) -> usize {
        self.pkg.manifest().task.output_cols()
    }

    /// Predict `b` rows of raw values in schema feature order (`raw.len() == b * n_features`).
    /// Allocation-free on success.
    pub fn predict_raw(&mut self, raw: &[Option<f64>], b: usize, out: &mut [f32]) -> Result<()> {
        let nf = self.n_features();
        if b == 0 || b > self.max_batch() {
            return Err(ArtifactError::Core(
                boundml_core::CoreError::BatchOutOfRange {
                    got: b,
                    max: self.max_batch(),
                },
            ));
        }
        if raw.len() != b * nf {
            return Err(ArtifactError::Input {
                row: 0,
                message: format!(
                    "expected {} values ({b} rows x {nf} features), got {}",
                    b * nf,
                    raw.len()
                ),
            });
        }
        let pre = &self.pkg.manifest().preprocessing;
        for r in 0..b {
            if let Err(i) = pre.apply(
                &raw[r * nf..(r + 1) * nf],
                &mut self.x[r * nf..(r + 1) * nf],
            ) {
                let f = &pre.features[i].name;
                let message = match raw[r * nf + i] {
                    None => format!(
                        "feature '{f}' is missing and the package's missing policy is 'reject'"
                    ),
                    Some(_) => format!("feature '{f}' overflows f32 after preprocessing"),
                };
                return Err(ArtifactError::Input { row: r, message });
            }
        }
        self.session.predict_into(&self.x[..b * nf], b, out)?;
        Ok(())
    }

    /// Convenience single-row prediction (allocates the result vector).
    pub fn predict_one(&mut self, raw: &[Option<f64>]) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.output_cols()];
        self.predict_raw(raw, 1, &mut out)?;
        Ok(out)
    }

    /// Map `(name, value)` pairs onto schema order in `dst` (`n_features` long).
    /// Absent features become missing; duplicate names are errors; unknown names
    /// follow the schema's `extra_columns` policy. Non-finite values are rejected.
    pub fn fill_named<'a>(
        &mut self,
        pairs: impl IntoIterator<Item = (&'a str, Option<f64>)>,
        dst: &mut [Option<f64>],
        row: usize,
    ) -> Result<()> {
        dst.iter_mut().for_each(|v| *v = None);
        self.seen.iter_mut().for_each(|s| *s = false);
        let reject_extra = self.pkg.manifest().schema.extra_columns == ExtraColumns::Reject;
        for (name, value) in pairs {
            match self.index.get(name) {
                Some(&i) => {
                    if std::mem::replace(&mut self.seen[i], true) {
                        return Err(ArtifactError::Input {
                            row,
                            message: format!("feature '{name}' given twice"),
                        });
                    }
                    if let Some(v) = value {
                        if !v.is_finite() {
                            return Err(ArtifactError::Input {
                                row,
                                message: format!("feature '{name}' is not finite"),
                            });
                        }
                    }
                    dst[i] = value;
                }
                None if reject_extra => {
                    return Err(ArtifactError::Input {
                        row,
                        message: format!(
                            "unknown field '{}' and the schema rejects extra columns",
                            name.chars().take(64).collect::<String>()
                        ),
                    });
                }
                None => {}
            }
        }
        Ok(())
    }
}
