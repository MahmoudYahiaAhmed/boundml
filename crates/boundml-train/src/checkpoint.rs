//! Resumable training checkpoints (distinct from inference packages).
//!
//! Layout: `state.json`, `state.sha256`, `tensors.safetensors`. Written with
//! `write_dir_atomic`, which keeps the previous checkpoint at `<dir>.prev`.

use crate::config::TrainConfig;
use crate::error::{Result, TrainError};
use crate::metrics::LabelStats;
use boundml_artifact::DataRef;
use boundml_artifact::fsutil::{
    decode_tensors, encode_tensors, prev_path, read_limited, sha256_hex, write_dir_atomic,
};
use boundml_core::{DenseSpec, Graph, Optimizer, OptimizerState};
use boundml_data::{Preprocessor, ReaderPosition, Schema};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const CHECKPOINT_FORMAT: &str = "boundml-checkpoint";
pub const CHECKPOINT_VERSION: u32 = 1;
const STATE: &str = "state.json";
const STATE_HASH: &str = "state.sha256";
const TENSORS: &str = "tensors.safetensors";
const MAX_STATE_BYTES: u64 = 64 << 20;
const MAX_TENSOR_BYTES: u64 = 4 << 30;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    /// Epoch currently in progress (or next to start).
    pub epoch: u32,
    pub batches_in_epoch: u64,
    pub global_step: u64,
    /// Data-order position within the epoch; `None` at an epoch boundary.
    pub reader: Option<ReaderPosition>,
    pub epoch_loss_sum: f64,
    pub epoch_rows: u64,
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: u32,
    pub train_loss: f64,
    pub valid: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointState {
    pub format: String,
    pub format_version: u32,
    pub runtime: String,
    pub config: TrainConfig,
    pub config_sha256: String,
    pub schema: Schema,
    pub preprocessing: Preprocessor,
    pub spec: DenseSpec,
    pub label_stats: LabelStats,
    pub optimizer_step: u64,
    pub progress: Progress,
    pub history: Vec<EpochRecord>,
    pub data: Vec<DataRef>,
    pub tensors_sha256: String,
}

/// A checkpoint in memory: state plus parameters and optimizer slots.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub state: CheckpointState,
    pub params: Vec<f32>,
    pub optimizer: OptimizerState,
}

impl Checkpoint {
    pub fn save(&mut self, dir: &Path, graph: &Graph) -> Result<()> {
        let mut named: Vec<(String, [usize; 2], &[f32])> = Vec::new();
        for (i, p) in graph.params().iter().enumerate() {
            let o = graph.param_offset(i);
            let len = graph.param_len(i);
            named.push((
                format!("param.{}", p.name),
                [p.rows, p.cols],
                &self.params[o..o + len],
            ));
            for (s, slot) in self.optimizer.slots.iter().enumerate() {
                named.push((
                    format!("opt{s}.{}", p.name),
                    [p.rows, p.cols],
                    &slot[o..o + len],
                ));
            }
        }
        let tensors = encode_tensors(&named)?;
        self.state.tensors_sha256 = sha256_hex(&tensors);
        self.state.optimizer_step = self.optimizer.step;
        let json = serde_json::to_vec_pretty(&self.state)
            .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        let hash = format!("{}\n", sha256_hex(&json));
        write_dir_atomic(
            dir,
            &[
                (STATE, &json),
                (STATE_HASH, hash.as_bytes()),
                (TENSORS, &tensors),
            ],
        )?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Checkpoint> {
        if !dir.exists() && prev_path(dir).exists() {
            return Err(TrainError::Checkpoint(format!(
                "{} is missing but {} exists (an interrupted write?); pass that path to resume from the previous checkpoint",
                dir.display(),
                prev_path(dir).display()
            )));
        }
        boundml_artifact::fsutil::check_dir(dir)?;
        let json = read_limited(&dir.join(STATE), MAX_STATE_BYTES)?;
        let want = String::from_utf8_lossy(&read_limited(&dir.join(STATE_HASH), 1024)?)
            .trim()
            .to_string();
        if sha256_hex(&json) != want {
            return Err(TrainError::Checkpoint(
                "state.json does not match state.sha256 (corrupted?)".into(),
            ));
        }
        let state: CheckpointState = serde_json::from_slice(&json)
            .map_err(|e| TrainError::Checkpoint(format!("state.json: {e}")))?;
        if state.format != CHECKPOINT_FORMAT || state.format_version != CHECKPOINT_VERSION {
            return Err(TrainError::Checkpoint(format!(
                "unsupported checkpoint format {} v{}",
                state.format, state.format_version
            )));
        }
        let bytes = read_limited(&dir.join(TENSORS), MAX_TENSOR_BYTES)?;
        if sha256_hex(&bytes) != state.tensors_sha256 {
            return Err(TrainError::Checkpoint(
                "tensors.safetensors hash mismatch (corrupted?)".into(),
            ));
        }
        state.preprocessing.validate(&state.schema)?;
        let graph = state.spec.graph()?;
        let slots = state.config.training.optimizer.state_slots();
        let decoded = decode_tensors(&bytes, (1 + slots) * graph.params().len())?;
        let find = |name: &str, shape: [usize; 2]| -> Result<&Vec<f32>> {
            decoded
                .iter()
                .find(|(n, s, _)| n == name && *s == shape)
                .map(|(_, _, v)| v)
                .ok_or_else(|| {
                    TrainError::Checkpoint(format!("missing or misshaped tensor '{name}'"))
                })
        };
        let n = graph.param_count();
        let mut params = vec![0.0f32; n];
        let mut opt_slots = vec![vec![0.0f32; n]; slots];
        for (i, p) in graph.params().iter().enumerate() {
            let o = graph.param_offset(i);
            let len = graph.param_len(i);
            params[o..o + len]
                .copy_from_slice(find(&format!("param.{}", p.name), [p.rows, p.cols])?);
            for (s, slot) in opt_slots.iter_mut().enumerate() {
                slot[o..o + len]
                    .copy_from_slice(find(&format!("opt{s}.{}", p.name), [p.rows, p.cols])?);
            }
        }
        if decoded.len() != (1 + slots) * graph.params().len() {
            return Err(TrainError::Checkpoint("unexpected extra tensors".into()));
        }
        let optimizer = OptimizerState {
            step: state.optimizer_step,
            slots: opt_slots,
        };
        // Validates slot shapes against the configured optimizer.
        Optimizer::new(state.config.training.optimizer.clone(), &graph)?
            .restore(optimizer.clone())?;
        Ok(Checkpoint {
            state,
            params,
            optimizer,
        })
    }
}
