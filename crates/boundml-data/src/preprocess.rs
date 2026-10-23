use crate::csvsrc::{CsvSource, RawRow};
use crate::error::{DataError, Result};
use crate::schema::{MissingPolicy, Schema};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Standard deviation below which a column is treated as constant.
const CONSTANT_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Normalize {
    /// `(x - mean) / std` with training statistics. Constant columns use `std = 1`.
    #[default]
    Standardize,
    /// Pass values through unchanged (imputation still applies).
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureStats {
    pub name: String,
    pub mean: f64,
    pub std: f64,
    /// Rows where this feature was present during fitting.
    pub count: u64,
    pub missing: u64,
    pub constant: bool,
}

/// Fitted, serializable preprocessing. Created only by [`Preprocessor::fit`] or
/// deserialization (followed by [`Preprocessor::validate`]); never refitted at inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preprocessor {
    pub normalize: Normalize,
    pub missing: MissingPolicy,
    pub features: Vec<FeatureStats>,
    pub rows_fitted: u64,
}

impl Preprocessor {
    /// Fit on a training CSV in one streaming pass (Welford's algorithm, f64).
    pub fn fit(path: &Path, schema: &Schema, normalize: Normalize) -> Result<Self> {
        let mut src = CsvSource::open(path, schema, true)?;
        let n = schema.features.len();
        let mut count = vec![0u64; n];
        let mut mean = vec![0.0f64; n];
        let mut m2 = vec![0.0f64; n];
        let mut row = RawRow::default();
        let mut rows = 0u64;
        while src.read_row(&mut row)? {
            rows += 1;
            for (i, v) in row.features.iter().enumerate() {
                match v {
                    Some(v) => {
                        count[i] += 1;
                        let d = v - mean[i];
                        mean[i] += d / count[i] as f64;
                        m2[i] += d * (v - mean[i]);
                    }
                    None if schema.missing == MissingPolicy::Reject => {
                        return Err(DataError::Missing {
                            path: src.path().to_string(),
                            line: row.line,
                            column: schema.features[i].clone(),
                        });
                    }
                    None => {}
                }
            }
        }
        if rows == 0 {
            return Err(DataError::Empty(format!(
                "training file {} has no rows",
                path.display()
            )));
        }
        let mut features = Vec::with_capacity(n);
        for i in 0..n {
            if count[i] == 0 {
                return Err(DataError::Empty(format!(
                    "feature '{}' has no observed values in the training data",
                    schema.features[i]
                )));
            }
            let var = m2[i] / count[i] as f64;
            let std = var.sqrt();
            let constant = std < CONSTANT_EPS;
            features.push(FeatureStats {
                name: schema.features[i].clone(),
                mean: mean[i],
                std: if constant { 1.0 } else { std },
                count: count[i],
                missing: rows - count[i],
                constant,
            });
        }
        Ok(Preprocessor {
            normalize,
            missing: schema.missing,
            features,
            rows_fitted: rows,
        })
    }

    /// Check internal consistency and agreement with a schema (used after loading).
    pub fn validate(&self, schema: &Schema) -> Result<()> {
        if self.features.len() != schema.features.len() {
            return Err(DataError::Preprocess(format!(
                "{} fitted features vs {} schema features",
                self.features.len(),
                schema.features.len()
            )));
        }
        for (f, name) in self.features.iter().zip(&schema.features) {
            if &f.name != name {
                return Err(DataError::Preprocess(format!(
                    "fitted feature '{}' does not match schema feature '{name}' at the same position",
                    f.name
                )));
            }
            if !f.mean.is_finite() || !f.std.is_finite() || f.std <= 0.0 {
                return Err(DataError::Preprocess(format!(
                    "invalid statistics for '{}'",
                    f.name
                )));
            }
        }
        if self.missing != schema.missing {
            return Err(DataError::Preprocess(
                "missing-value policy differs from schema".into(),
            ));
        }
        Ok(())
    }

    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Transform one row in schema order into `out`. Allocation-free.
    /// `Err(i)`: feature `i` is missing under the reject policy or overflows f32.
    pub fn apply(&self, raw: &[Option<f64>], out: &mut [f32]) -> std::result::Result<(), usize> {
        debug_assert_eq!(raw.len(), self.features.len());
        for (i, (v, f)) in raw.iter().zip(&self.features).enumerate() {
            let v = match (v, self.missing) {
                (Some(v), _) => *v,
                (None, MissingPolicy::ImputeMean) => f.mean,
                (None, MissingPolicy::Reject) => return Err(i),
            };
            let t = match self.normalize {
                Normalize::Standardize => (v - f.mean) / f.std,
                Normalize::None => v,
            };
            out[i] = t as f32;
            if !out[i].is_finite() {
                return Err(i);
            }
        }
        Ok(())
    }
}
