//! Supported dense model family and the immutable model / mutable session split.
//!
//! * [`DenseSpec`] declares linear regression, logistic/softmax classification and
//!   multilayer perceptrons; it lowers to a [`GraphDef`] with both an inference
//!   head and a training loss.
//! * [`Model`] is validated, immutable state (graph + weights); share it via `Arc`.
//! * [`InferenceSession`] owns one preallocated arena. Create one per thread/worker;
//!   sessions never share scratch or gradients.

use crate::error::{CoreError, Result};
use crate::exec::Executor;
use crate::graph::{Graph, GraphBuilder, GraphDef};
use crate::plan::{MemoryReport, Mode, Plan, PlannerKind};
use crate::rng::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Task {
    /// Single continuous target, MSE loss, identity output.
    Regression,
    /// Labels 0/1, BCE-with-logits loss, output = probability of class 1.
    Binary,
    /// Class indices `0..classes`, cross-entropy loss, output = class probabilities.
    Multiclass { classes: usize },
}

impl Task {
    pub fn output_cols(&self) -> usize {
        match self {
            Task::Regression | Task::Binary => 1,
            Task::Multiclass { classes } => *classes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[default]
    Relu,
    Tanh,
    Sigmoid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseSpec {
    pub inputs: usize,
    /// Hidden layer widths; empty = linear/logistic/softmax regression.
    #[serde(default)]
    pub hidden: Vec<usize>,
    #[serde(default)]
    pub activation: Activation,
    pub task: Task,
}

pub const MAX_LAYERS: usize = 64;

impl DenseSpec {
    pub fn graph_def(&self) -> Result<GraphDef> {
        if self.hidden.len() > MAX_LAYERS {
            return Err(CoreError::Graph(format!(
                "at most {MAX_LAYERS} hidden layers are supported"
            )));
        }
        if let Task::Multiclass { classes } = self.task {
            if classes < 2 {
                return Err(CoreError::Graph(
                    "multiclass needs at least 2 classes".into(),
                ));
            }
        }
        let mut g = GraphBuilder::new();
        let x = g.input("x", self.inputs);
        let mut h = x;
        let mut width = self.inputs;
        let out_cols = self.task.output_cols();
        let n_layers = self.hidden.len() + 1;
        for layer in 0..n_layers {
            let last = layer + 1 == n_layers;
            let out = if last { out_cols } else { self.hidden[layer] };
            let w = g.param(&format!("dense{layer}.weight"), width, out, true);
            let bias = g.param(&format!("dense{layer}.bias"), 1, out, false);
            let mm = g.matmul(&format!("dense{layer}.matmul"), h, w);
            let name = if last && self.task == Task::Regression {
                "output".to_string()
            } else if last {
                "logits".to_string()
            } else {
                format!("dense{layer}.pre")
            };
            h = g.add(&name, mm, bias);
            if !last {
                let act = format!("dense{layer}.act");
                h = match self.activation {
                    Activation::Relu => g.relu(&act, h),
                    Activation::Tanh => g.tanh(&act, h),
                    Activation::Sigmoid => g.sigmoid(&act, h),
                };
            }
            width = out;
        }
        let logits = h;
        let target = g.target("y", 1);
        let (output, loss) = match self.task {
            Task::Regression => (logits, g.mse_loss("loss", logits, target)),
            Task::Binary => {
                let o = g.sigmoid("output", logits);
                (o, g.bce_with_logits_loss("loss", logits, target))
            }
            Task::Multiclass { .. } => {
                let o = g.softmax("output", logits);
                (o, g.cross_entropy_loss("loss", logits, target))
            }
        };
        Ok(g.into_def(output, Some(loss)))
    }

    pub fn graph(&self) -> Result<Graph> {
        Graph::new(self.graph_def()?)
    }

    /// Seeded initialization: He-uniform for weights feeding ReLU, Xavier-uniform
    /// otherwise; biases zero.
    pub fn init_params(&self, graph: &Graph, rng: &mut Rng) -> Vec<f32> {
        let mut out = vec![0.0f32; graph.param_count()];
        let n_layers = self.hidden.len() + 1;
        for (i, p) in graph.params().iter().enumerate() {
            if !p.decay {
                continue;
            }
            let layer = i / 2;
            let relu_next = self.activation == Activation::Relu && layer + 1 < n_layers;
            let bound = if relu_next {
                (6.0 / p.rows as f32).sqrt()
            } else {
                (6.0 / (p.rows + p.cols) as f32).sqrt()
            };
            let o = graph.param_offset(i);
            for v in &mut out[o..o + graph.param_len(i)] {
                *v = rng.uniform(-bound, bound);
            }
        }
        out
    }
}

/// Validated, immutable model state shared between sessions.
#[derive(Debug)]
pub struct Model {
    graph: Arc<Graph>,
    weights: Vec<f32>,
}

impl Model {
    pub fn new(graph: Graph, weights: Vec<f32>) -> Result<Model> {
        if weights.len() != graph.param_count() {
            return Err(CoreError::Length {
                what: "weights",
                expected: graph.param_count(),
                got: weights.len(),
            });
        }
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(CoreError::NonFinite("weights"));
        }
        Ok(Model {
            graph: Arc::new(graph),
            weights,
        })
    }
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn input_cols(&self) -> usize {
        self.graph.input_cols()
    }
    pub fn output_cols(&self) -> usize {
        self.graph.output_shape().cols.resolve(1)
    }

    /// Plan inference without allocating tensor storage.
    pub fn plan_inference(&self, max_batch: usize, planner: PlannerKind) -> Result<Plan> {
        Plan::new(&self.graph, Mode::Inference, max_batch, planner)
    }

    pub fn inference_report(
        &self,
        max_batch: usize,
        sessions: usize,
        planner: PlannerKind,
    ) -> Result<MemoryReport> {
        Ok(self
            .plan_inference(max_batch, planner)?
            .report(&self.graph, sessions, 0))
    }
}

/// A mutable inference context with its own arena.
#[derive(Debug)]
pub struct InferenceSession {
    model: Arc<Model>,
    exec: Executor,
}

impl InferenceSession {
    pub fn new(model: Arc<Model>, max_batch: usize) -> Result<Self> {
        Self::with_budget(model, max_batch, PlannerKind::Reuse, None, 1)
    }

    /// Plan, check `budget` (bytes for `sessions` sessions plus shared weights) and
    /// only then allocate this session's arena.
    pub fn with_budget(
        model: Arc<Model>,
        max_batch: usize,
        planner: PlannerKind,
        budget: Option<u64>,
        sessions: usize,
    ) -> Result<Self> {
        let plan = model.plan_inference(max_batch, planner)?;
        if let Some(b) = budget {
            plan.report(model.graph(), sessions, 0).check_budget(b)?;
        }
        let exec = Executor::from_plan(Arc::clone(model.graph()), plan);
        Ok(InferenceSession { model, exec })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub fn max_batch(&self) -> usize {
        self.exec.plan().max_batch
    }

    /// Predict `b` rows of already-preprocessed features into `out`. Allocation-free.
    pub fn predict_into(&mut self, x: &[f32], b: usize, out: &mut [f32]) -> Result<()> {
        self.exec.infer(self.model.weights(), x, b, out)
    }
}
