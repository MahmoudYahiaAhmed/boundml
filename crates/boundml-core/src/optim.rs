//! SGD (with optional momentum and coupled L2) and AdamW (decoupled weight decay),
//! matching PyTorch's update rules for `torch.optim.SGD` and `torch.optim.AdamW`.
//!
//! Weight decay applies only to parameters whose [`ParamSpec::decay`] is set
//! (by convention: weight matrices, not biases).
//!
//! [`ParamSpec::decay`]: crate::graph::ParamSpec

use crate::error::{CoreError, Result};
use crate::graph::Graph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OptimizerConfig {
    Sgd {
        lr: f32,
        #[serde(default)]
        momentum: f32,
        #[serde(default)]
        weight_decay: f32,
    },
    #[serde(rename = "adamw")]
    AdamW {
        lr: f32,
        #[serde(default = "beta1")]
        beta1: f32,
        #[serde(default = "beta2")]
        beta2: f32,
        #[serde(default = "eps")]
        eps: f32,
        #[serde(default = "wd")]
        weight_decay: f32,
    },
}

fn beta1() -> f32 {
    0.9
}
fn beta2() -> f32 {
    0.999
}
fn eps() -> f32 {
    1e-8
}
fn wd() -> f32 {
    0.01
}

impl OptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(CoreError::Optimizer(m));
        let pos = |name: &str, v: f32| -> Result<()> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(CoreError::Optimizer(format!(
                    "{name} must be finite and > 0, got {v}"
                )))
            }
        };
        let nonneg = |name: &str, v: f32| -> Result<()> {
            if v.is_finite() && v >= 0.0 {
                Ok(())
            } else {
                Err(CoreError::Optimizer(format!(
                    "{name} must be finite and >= 0, got {v}"
                )))
            }
        };
        match *self {
            OptimizerConfig::Sgd {
                lr,
                momentum,
                weight_decay,
            } => {
                pos("lr", lr)?;
                nonneg("weight_decay", weight_decay)?;
                if !(0.0..1.0).contains(&momentum) {
                    return bad(format!("momentum must be in [0, 1), got {momentum}"));
                }
            }
            OptimizerConfig::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                pos("lr", lr)?;
                pos("eps", eps)?;
                nonneg("weight_decay", weight_decay)?;
                for (n, b) in [("beta1", beta1), ("beta2", beta2)] {
                    if !(0.0..1.0).contains(&b) {
                        return bad(format!("{n} must be in [0, 1), got {b}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Number of f32 state slots per parameter.
    pub fn state_slots(&self) -> usize {
        match *self {
            OptimizerConfig::Sgd { momentum, .. } if momentum > 0.0 => 1,
            OptimizerConfig::Sgd { .. } => 0,
            OptimizerConfig::AdamW { .. } => 2,
        }
    }
}

/// Serializable optimizer state for checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerState {
    pub step: u64,
    pub slots: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct Optimizer {
    config: OptimizerConfig,
    step: u64,
    /// `(start, len, decay)` per parameter in the flat store.
    segments: Vec<(usize, usize, bool)>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Optimizer {
    pub fn new(config: OptimizerConfig, graph: &Graph) -> Result<Self> {
        config.validate()?;
        let segments = graph
            .params()
            .iter()
            .enumerate()
            .map(|(i, p)| (graph.param_offset(i), graph.param_len(i), p.decay))
            .collect();
        let n = graph.param_count();
        let slots = config.state_slots();
        Ok(Optimizer {
            config,
            step: 0,
            segments,
            m: if slots >= 1 { vec![0.0; n] } else { Vec::new() },
            v: if slots >= 2 { vec![0.0; n] } else { Vec::new() },
        })
    }

    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }
    pub fn steps(&self) -> u64 {
        self.step
    }

    pub fn state(&self) -> OptimizerState {
        let mut slots = Vec::new();
        if !self.m.is_empty() {
            slots.push(self.m.clone());
        }
        if !self.v.is_empty() {
            slots.push(self.v.clone());
        }
        OptimizerState {
            step: self.step,
            slots,
        }
    }

    pub fn restore(&mut self, state: OptimizerState) -> Result<()> {
        let slots = self.config.state_slots();
        let n = self.segments.iter().map(|s| s.1).sum::<usize>();
        if state.slots.len() != slots || state.slots.iter().any(|s| s.len() != n) {
            return Err(CoreError::Optimizer(format!(
                "checkpoint optimizer state has {} slot(s); this optimizer needs {slots} of {n} values",
                state.slots.len()
            )));
        }
        if state.slots.iter().flatten().any(|v| !v.is_finite()) {
            return Err(CoreError::NonFinite("optimizer state"));
        }
        let mut it = state.slots.into_iter();
        self.m = if slots >= 1 {
            it.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        self.v = if slots >= 2 {
            it.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        self.step = state.step;
        Ok(())
    }

    /// Apply one update. Allocation-free.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<()> {
        let n: usize = self.segments.iter().map(|s| s.1).sum();
        if params.len() != n || grads.len() != n {
            return Err(CoreError::Length {
                what: "optimizer parameters/gradients",
                expected: n,
                got: params.len().min(grads.len()),
            });
        }
        if grads.iter().any(|g| !g.is_finite()) {
            return Err(CoreError::NonFinite("gradients"));
        }
        self.step += 1;
        match self.config {
            OptimizerConfig::Sgd {
                lr,
                momentum,
                weight_decay,
            } => {
                for &(start, len, decay) in &self.segments {
                    let wd = if decay { weight_decay } else { 0.0 };
                    for i in start..start + len {
                        let g = grads[i] + wd * params[i];
                        let d = if momentum > 0.0 {
                            self.m[i] = momentum * self.m[i] + g;
                            self.m[i]
                        } else {
                            g
                        };
                        params[i] -= lr * d;
                    }
                }
            }
            OptimizerConfig::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                let t = self.step as i32;
                let bc1 = 1.0 - (beta1 as f64).powi(t);
                let bc2_sqrt = (1.0 - (beta2 as f64).powi(t)).sqrt();
                let step_size = (lr as f64 / bc1) as f32;
                let bc2_sqrt = bc2_sqrt as f32;
                for &(start, len, decay) in &self.segments {
                    for i in start..start + len {
                        if decay {
                            params[i] *= 1.0 - lr * weight_decay;
                        }
                        let g = grads[i];
                        self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
                        self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
                        let denom = self.v[i].sqrt() / bc2_sqrt + eps;
                        params[i] -= step_size * self.m[i] / denom;
                    }
                }
            }
        }
        Ok(())
    }
}
