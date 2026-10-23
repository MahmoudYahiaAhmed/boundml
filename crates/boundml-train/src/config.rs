//! Declarative training configuration (`model.toml`). Paths are relative to the
//! directory containing the configuration file.

use crate::error::{Result, TrainError};
use boundml_core::{Activation, DenseSpec, OptimizerConfig, PlannerKind};
use boundml_data::{ExtraColumns, LabelKind, MissingPolicy, Normalize, Schema};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MAX_CONFIG_BYTES: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Regression,
    Classification,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    pub train: PathBuf,
    #[serde(default)]
    pub valid: Option<PathBuf>,
    pub features: Vec<String>,
    pub label: String,
    pub task: TaskKind,
    /// Class labels in index order (classification only).
    #[serde(default)]
    pub classes: Vec<String>,
    #[serde(default)]
    pub missing: MissingPolicy,
    #[serde(default)]
    pub extra_columns: ExtraColumns,
    #[serde(default)]
    pub normalize: Normalize,
    /// Rows held in the shuffle buffer (1 = file order).
    #[serde(default = "default_shuffle")]
    pub shuffle_buffer: usize,
}

fn default_shuffle() -> usize {
    1024
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[serde(default)]
    pub hidden: Vec<usize>,
    #[serde(default)]
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: usize,
    pub optimizer: OptimizerConfig,
    /// Also checkpoint every N optimizer steps (0 = only at epoch end / stop).
    #[serde(default)]
    pub checkpoint_every_steps: u64,
    /// Budget for planned training tensors (e.g. "64MiB").
    #[serde(default)]
    pub tensor_budget: Option<String>,
    #[serde(default)]
    pub planner: PlannerKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeployConfig {
    #[serde(default = "default_max_batch")]
    pub max_batch: usize,
    /// Budget for planned inference tensors across `sessions` (e.g. "8MiB").
    #[serde(default)]
    pub tensor_budget: Option<String>,
    #[serde(default = "default_sessions")]
    pub sessions: usize,
    #[serde(default)]
    pub planner: PlannerKind,
}

impl Default for DeployConfig {
    fn default() -> Self {
        DeployConfig {
            max_batch: default_max_batch(),
            tensor_budget: None,
            sessions: default_sessions(),
            planner: PlannerKind::Reuse,
        }
    }
}

fn default_max_batch() -> usize {
    256
}
fn default_sessions() -> usize {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub name: String,
    #[serde(default)]
    pub seed: u64,
    pub data: DataConfig,
    #[serde(default)]
    pub model: ModelConfig,
    pub training: TrainingConfig,
    #[serde(default)]
    pub deploy: DeployConfig,
}

impl TrainConfig {
    /// Load, parse and statically validate. Returns the config and its base directory.
    pub fn load(path: &Path) -> Result<(TrainConfig, PathBuf)> {
        let bytes = boundml_artifact::fsutil::read_limited(path, MAX_CONFIG_BYTES)?;
        let text = String::from_utf8(bytes)
            .map_err(|_| TrainError::Config(format!("{} is not UTF-8", path.display())))?;
        let cfg: TrainConfig = toml::from_str(&text)
            .map_err(|e| TrainError::Config(format!("{}: {e}", path.display())))?;
        cfg.validate()?;
        let base = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf();
        Ok((cfg, base))
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(TrainError::Config(m.to_string()));
        if self.name.is_empty() || self.name.len() > 128 {
            return bad("name must be 1..=128 bytes");
        }
        let t = &self.training;
        if t.epochs == 0 || t.epochs > 100_000 {
            return bad("training.epochs must be 1..=100000");
        }
        if t.batch_size == 0 || t.batch_size > 1 << 20 {
            return bad("training.batch_size must be 1..=1048576");
        }
        t.optimizer.validate()?;
        if self.deploy.max_batch == 0 || self.deploy.sessions == 0 || self.deploy.sessions > 1024 {
            return bad("deploy.max_batch must be >= 1 and deploy.sessions 1..=1024");
        }
        if self.model.hidden.iter().any(|&h| h == 0 || h > 1 << 16) {
            return bad("model.hidden widths must be 1..=65536");
        }
        match (self.data.task, self.data.classes.is_empty()) {
            (TaskKind::Regression, false) => {
                return bad("data.classes must be empty for regression");
            }
            (TaskKind::Classification, true) => {
                return bad("data.classes is required for classification");
            }
            _ => {}
        }
        for b in [&t.tensor_budget, &self.deploy.tensor_budget]
            .into_iter()
            .flatten()
        {
            parse_bytes(b)?;
        }
        self.schema().validate()?;
        self.dense_spec().graph()?;
        Ok(())
    }

    pub fn schema(&self) -> Schema {
        Schema {
            features: self.data.features.clone(),
            label: self.data.label.clone(),
            label_kind: match self.data.task {
                TaskKind::Regression => LabelKind::Regression,
                TaskKind::Classification => LabelKind::Classes {
                    classes: self.data.classes.clone(),
                },
            },
            missing: self.data.missing,
            extra_columns: self.data.extra_columns,
        }
    }

    pub fn dense_spec(&self) -> DenseSpec {
        DenseSpec {
            inputs: self.data.features.len(),
            hidden: self.model.hidden.clone(),
            activation: self.model.activation,
            task: self.schema().task(),
        }
    }

    pub fn train_budget(&self) -> Result<Option<u64>> {
        self.training
            .tensor_budget
            .as_deref()
            .map(parse_bytes)
            .transpose()
    }

    pub fn deploy_budget(&self) -> Result<Option<u64>> {
        self.deploy
            .tensor_budget
            .as_deref()
            .map(parse_bytes)
            .transpose()
    }
}

/// Parse sizes like `1048576`, `512KiB`, `64MiB`, `2GiB`, `10MB`.
pub fn parse_bytes(s: &str) -> Result<u64> {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let n: u64 = num
        .parse()
        .map_err(|_| TrainError::Config(format!("invalid size '{s}'")))?;
    let mult: u64 = match unit.trim() {
        "" | "B" => 1,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        u => {
            return Err(TrainError::Config(format!(
                "unknown size unit '{u}' in '{s}'"
            )));
        }
    };
    n.checked_mul(mult)
        .ok_or_else(|| TrainError::Config(format!("size '{s}' overflows")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_bytes("64MiB").unwrap(), 64 << 20);
        assert_eq!(parse_bytes("10MB").unwrap(), 10_000_000);
        assert_eq!(parse_bytes("123").unwrap(), 123);
        assert!(parse_bytes("12 parsecs").is_err());
        assert!(parse_bytes("99999999999999999999GiB").is_err());
    }
}
