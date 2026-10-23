//! Package manifest (format version 1). See `docs/package-format.md`.

use crate::error::{ArtifactError, Result};
use boundml_core::{GraphDef, PlannerKind, Task};
use boundml_data::{Preprocessor, Schema};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PACKAGE_FORMAT: &str = "boundml-package";
pub const PACKAGE_FORMAT_VERSION: u32 = 1;
/// Oldest runtime that can execute packages written by this version.
pub const MIN_RUNTIME_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInfo {
    pub name: String,
    pub shape: [usize; 2],
    pub dtype: String,
    pub layout: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    /// Largest batch any session may execute; plans are sized for it.
    pub max_batch: usize,
    /// Planned tensor-storage budget in bytes for `sessions` sessions, if enforced.
    #[serde(default)]
    pub tensor_budget_bytes: Option<u64>,
    #[serde(default = "one")]
    pub sessions: usize,
    #[serde(default)]
    pub planner: PlannerKind,
}

fn one() -> usize {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataRef {
    pub role: String,
    pub file_name: String,
    pub sha256: String,
    pub rows: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub created_unix: u64,
    pub created_by: String,
    /// e.g. `boundml-train`, `import:pytorch`, `import:keras`.
    pub source: String,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub training_config_sha256: Option<String>,
    #[serde(default)]
    pub data: Vec<DataRef>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub notes: Option<String>,
}

/// Raw (pre-preprocessing) inputs with expected outputs. Detects execution drift on
/// these inputs only; it does not prove correctness on other inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestVectors {
    /// Where the inputs came from, e.g. `synthetic` or `source-framework`.
    pub origin: String,
    pub inputs: Vec<Vec<Option<f64>>>,
    pub expected: Vec<Vec<f32>>,
    pub atol: f32,
    pub rtol: f32,
}

pub const MAX_TEST_VECTORS: usize = 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileHash {
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub format_version: u32,
    pub min_runtime_version: String,
    pub name: String,
    pub task: Task,
    pub schema: Schema,
    pub preprocessing: Preprocessor,
    pub graph: GraphDef,
    pub tensors: Vec<TensorInfo>,
    pub bounds: Bounds,
    pub provenance: Provenance,
    pub files: BTreeMap<String, FileHash>,
    #[serde(default)]
    pub test_vectors: Option<TestVectors>,
}

impl Manifest {
    pub fn check_version(&self) -> Result<()> {
        if self.format != PACKAGE_FORMAT {
            return Err(ArtifactError::Format(format!(
                "format is '{}', expected '{PACKAGE_FORMAT}'",
                self.format
            )));
        }
        if self.format_version != PACKAGE_FORMAT_VERSION {
            return Err(ArtifactError::Version(format!(
                "package format version {} is not supported (this runtime reads version {PACKAGE_FORMAT_VERSION})",
                self.format_version
            )));
        }
        let need = parse_version(&self.min_runtime_version)?;
        let have = parse_version(boundml_core::VERSION)?;
        if need > have {
            return Err(ArtifactError::Version(format!(
                "package requires runtime >= {}, this is {}",
                self.min_runtime_version,
                boundml_core::VERSION
            )));
        }
        Ok(())
    }

    /// Human-readable names of output columns.
    pub fn output_names(&self) -> Vec<String> {
        match (&self.task, self.schema.classes()) {
            (Task::Regression, _) => vec![self.schema.label.clone()],
            (Task::Binary, Some(c)) => vec![format!("p_{}", c[1])],
            (Task::Multiclass { .. }, Some(c)) => c.iter().map(|c| format!("p_{c}")).collect(),
            (_, None) => vec!["output".into()],
        }
    }
}

pub fn parse_version(v: &str) -> Result<(u64, u64, u64)> {
    let parts: Vec<&str> = v.split('.').collect();
    let bad = || ArtifactError::Format(format!("invalid version '{v}'"));
    if parts.len() != 3 {
        return Err(bad());
    }
    let p = |s: &str| s.parse::<u64>().map_err(|_| bad());
    Ok((p(parts[0])?, p(parts[1])?, p(parts[2])?))
}
