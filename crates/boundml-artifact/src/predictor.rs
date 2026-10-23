//! The embeddable prediction API: raw (unpreprocessed) feature values in, model
//! outputs out, through exactly the fitted preprocessing stored in the package.

use crate::error::{ArtifactError, Result};
use crate::package::{Package, SessionGuard};
use boundml_core::{CoreError, InferenceSession};
use boundml_data::{ApplyError, ExtraColumns};
use std::collections::HashMap;
use std::sync::Arc;

/// A mutable prediction context: one planned arena (preprocessed inputs are written
/// straight into its input buffer). Not `Sync`; create one per worker thread from a
/// shared `Arc<Package>`.
///
/// Each `Predictor` holds one of the package's `bounds.sessions` reservations for its
/// whole lifetime. Creating more simultaneous predictors than the package declares
/// fails with `BML-A010`, so the declared tensor budget (which covers exactly that
/// many sessions) is enforced. The reservation is released when the predictor is
/// dropped.
pub struct Predictor {
    pkg: Package,
    session: InferenceSession,
    index: HashMap<String, usize>,
    seen: Vec<bool>,
    // Declared last so the reservation is released after the arena is freed.
    _reservation: SessionGuard,
}

impl Predictor {
    /// `max_batch` must not exceed the package's declared `bounds.max_batch`. A
    /// session reservation is taken first, then the package's tensor budget (if any)
    /// is checked, and only then is the arena allocated.
    pub fn new(pkg: Arc<Package>, max_batch: usize) -> Result<Self> {
        Self::from_package((*pkg).clone(), max_batch)
    }

    pub(crate) fn from_package(pkg: Package, max_batch: usize) -> Result<Self> {
        let b = &pkg.manifest().bounds;
        if max_batch == 0 || max_batch > b.max_batch {
            return Err(ArtifactError::Core(CoreError::BatchOutOfRange {
                got: max_batch,
                max: b.max_batch,
            }));
        }
        let reservation = pkg.reserve_session()?;
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
            seen: vec![false; nf],
            pkg,
            session,
            index,
            _reservation: reservation,
        })
    }

    pub fn package(&self) -> &Package {
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

    /// Predict `b` rows of raw values in schema feature order (`raw.len() == b * n_features`,
    /// `out.len() == b * output_cols`). Allocation-free on success.
    pub fn predict_raw(&mut self, raw: &[Option<f64>], b: usize, out: &mut [f32]) -> Result<()> {
        let nf = self.n_features();
        if b == 0 || b > self.max_batch() {
            return Err(ArtifactError::Core(CoreError::BatchOutOfRange {
                got: b,
                max: self.max_batch(),
            }));
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
        let mut failure: Option<ArtifactError> = None;
        let result = self.session.predict_staged(b, out, |x| {
            for r in 0..b {
                if let Err(e) = pre.apply(&raw[r * nf..(r + 1) * nf], &mut x[r * nf..(r + 1) * nf])
                {
                    let i = match e {
                        ApplyError::Feature(i) => i,
                        e @ ApplyError::Length { .. } => {
                            failure = Some(ArtifactError::Input {
                                row: r,
                                message: e.to_string(),
                            });
                            return Err(CoreError::NonFinite("input"));
                        }
                    };
                    let f = &pre.features[i].name;
                    let message = match raw[r * nf + i] {
                        None => format!(
                            "feature '{f}' is missing and the package's missing policy is 'reject'"
                        ),
                        Some(_) => format!("feature '{f}' overflows f32 after preprocessing"),
                    };
                    failure = Some(ArtifactError::Input { row: r, message });
                    return Err(CoreError::NonFinite("input"));
                }
            }
            Ok(())
        });
        match (failure, result) {
            (Some(e), _) => Err(e),
            (None, r) => Ok(r?),
        }
    }

    /// Convenience single-row prediction (allocates the result vector).
    pub fn predict_one(&mut self, raw: &[Option<f64>]) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.output_cols()];
        self.predict_raw(raw, 1, &mut out)?;
        Ok(out)
    }

    /// Map `(name, value)` pairs onto schema order in `dst`, which must be exactly
    /// `n_features` long. Absent features become missing; duplicate names are errors;
    /// unknown names follow the schema's `extra_columns` policy. Non-finite values are
    /// rejected. `dst` is only modified when the call succeeds.
    pub fn fill_named<'a>(
        &mut self,
        pairs: impl IntoIterator<Item = (&'a str, Option<f64>)>,
        dst: &mut [Option<f64>],
        row: usize,
    ) -> Result<()> {
        let nf = self.n_features();
        if dst.len() != nf {
            return Err(ArtifactError::Input {
                row,
                message: format!(
                    "destination holds {} values, the schema has {nf} features",
                    dst.len()
                ),
            });
        }
        self.seen.iter_mut().for_each(|s| *s = false);
        let reject_extra = self.pkg.manifest().schema.extra_columns == ExtraColumns::Reject;
        // Validate everything first, then write, so a rejected row leaves `dst` intact.
        let mut staged: Vec<(usize, Option<f64>)> = Vec::new();
        for (name, value) in pairs {
            match self.index.get(name) {
                Some(&i) => {
                    if std::mem::replace(&mut self.seen[i], true) {
                        return Err(ArtifactError::Input {
                            row,
                            message: format!("feature '{name}' given twice"),
                        });
                    }
                    if let Some(v) = value
                        && !v.is_finite()
                    {
                        return Err(ArtifactError::Input {
                            row,
                            message: format!("feature '{name}' is not finite"),
                        });
                    }
                    staged.push((i, value));
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
        dst.iter_mut().for_each(|v| *v = None);
        for (i, v) in staged {
            dst[i] = v;
        }
        Ok(())
    }
}
