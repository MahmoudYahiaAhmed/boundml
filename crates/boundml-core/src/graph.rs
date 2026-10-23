//! A small static graph IR over rank-2 f32 tensors.
//!
//! A [`GraphDef`] is plain serializable data (used inside packages). [`Graph::new`]
//! validates it — topological order, operator arity, shape rules, broadcasting,
//! gradient flow — and is the only way to obtain an executable [`Graph`].
//!
//! One graph can carry both an inference `output` and a training `loss`; planning
//! executes only the ancestors of whichever root the mode requires.

use crate::error::{CoreError, Result};
use crate::shape::{Dim, Shape, broadcastable};
use serde::{Deserialize, Serialize};

pub type NodeId = usize;

/// Upper bounds that keep validation and planning cost proportional to trusted limits.
pub const MAX_NODES: usize = 4096;
pub const MAX_PARAMS: usize = 1024;
pub const MAX_NAME_LEN: usize = 128;

/// Supported operators. Semantics are documented in `docs/architecture.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    /// The feature matrix `[B, cols]`, written by the caller before execution.
    Input {
        cols: usize,
    },
    /// Training labels `[B, cols]`; never differentiated.
    Target {
        cols: usize,
    },
    /// A trainable parameter (index into [`GraphDef::params`]).
    Param {
        param: usize,
    },
    /// `op(a) @ op(b)` where `op` optionally transposes.
    MatMul {
        a: NodeId,
        b: NodeId,
        #[serde(default)]
        ta: bool,
        #[serde(default)]
        tb: bool,
    },
    /// Elementwise `a + b`; `b` may broadcast along a dimension of size 1.
    Add {
        a: NodeId,
        b: NodeId,
    },
    /// Elementwise `a * b`; `b` may broadcast along a dimension of size 1.
    Mul {
        a: NodeId,
        b: NodeId,
    },
    /// `max(x, 0)`; derivative at 0 is defined as 0.
    Relu {
        x: NodeId,
    },
    Sigmoid {
        x: NodeId,
    },
    Tanh {
        x: NodeId,
    },
    /// Row-wise numerically stable softmax.
    Softmax {
        x: NodeId,
    },
    /// Row-wise numerically stable log-softmax.
    LogSoftmax {
        x: NodeId,
    },
    Transpose {
        x: NodeId,
    },
    /// Sum of all elements -> `[1, 1]`.
    SumAll {
        x: NodeId,
    },
    /// Mean of all elements -> `[1, 1]`.
    MeanAll {
        x: NodeId,
    },
    /// Mean over all elements of `(pred - target)^2` -> `[1, 1]`.
    MseLoss {
        pred: NodeId,
        target: NodeId,
    },
    /// Mean binary cross-entropy computed from logits -> `[1, 1]`.
    BceWithLogitsLoss {
        logits: NodeId,
        target: NodeId,
    },
    /// Mean multiclass cross-entropy from logits `[R, K]` and class indices `[R, 1]`.
    CrossEntropyLoss {
        logits: NodeId,
        target: NodeId,
    },
}

impl Op {
    /// Operand node ids in a fixed order.
    pub fn inputs(&self) -> ([NodeId; 2], usize) {
        use Op::*;
        match *self {
            Input { .. } | Target { .. } | Param { .. } => ([0, 0], 0),
            MatMul { a, b, .. } | Add { a, b } | Mul { a, b } => ([a, b], 2),
            Relu { x }
            | Sigmoid { x }
            | Tanh { x }
            | Softmax { x }
            | LogSoftmax { x }
            | Transpose { x }
            | SumAll { x }
            | MeanAll { x } => ([x, 0], 1),
            MseLoss { pred, target } => ([pred, target], 2),
            BceWithLogitsLoss { logits, target } | CrossEntropyLoss { logits, target } => {
                ([logits, target], 2)
            }
        }
    }

    pub fn input_list(&self) -> impl Iterator<Item = NodeId> {
        let (ids, n) = self.inputs();
        ids.into_iter().take(n)
    }

    pub fn is_loss(&self) -> bool {
        matches!(
            self,
            Op::MseLoss { .. } | Op::BceWithLogitsLoss { .. } | Op::CrossEntropyLoss { .. }
        )
    }

    /// Whether backward reads operand `i`'s forward value.
    pub fn backward_reads_input(&self, i: usize) -> bool {
        match self {
            Op::MatMul { .. } | Op::Mul { .. } => true,
            op if op.is_loss() => true,
            _ => {
                let _ = i;
                false
            }
        }
    }

    /// Whether backward reads this node's own forward output.
    pub fn backward_reads_output(&self) -> bool {
        matches!(
            self,
            Op::Relu { .. }
                | Op::Sigmoid { .. }
                | Op::Tanh { .. }
                | Op::Softmax { .. }
                | Op::LogSoftmax { .. }
        )
    }

    pub fn kind(&self) -> &'static str {
        use Op::*;
        match self {
            Input { .. } => "input",
            Target { .. } => "target",
            Param { .. } => "param",
            MatMul { .. } => "matmul",
            Add { .. } => "add",
            Mul { .. } => "mul",
            Relu { .. } => "relu",
            Sigmoid { .. } => "sigmoid",
            Tanh { .. } => "tanh",
            Softmax { .. } => "softmax",
            LogSoftmax { .. } => "log_softmax",
            Transpose { .. } => "transpose",
            SumAll { .. } => "sum_all",
            MeanAll { .. } => "mean_all",
            MseLoss { .. } => "mse_loss",
            BceWithLogitsLoss { .. } => "bce_with_logits_loss",
            CrossEntropyLoss { .. } => "cross_entropy_loss",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSpec {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    /// Whether decoupled weight decay applies (weights: yes, biases: no by convention).
    #[serde(default)]
    pub decay: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeDef {
    pub name: String,
    #[serde(flatten)]
    pub op: Op,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphDef {
    pub params: Vec<ParamSpec>,
    pub nodes: Vec<NodeDef>,
    /// Inference root.
    pub output: NodeId,
    /// Training root (a `[1, 1]` node), if the graph is trainable.
    pub loss: Option<NodeId>,
}

/// A validated graph. Immutable after construction.
#[derive(Debug, Clone)]
pub struct Graph {
    def: GraphDef,
    shapes: Vec<Shape>,
    requires_grad: Vec<bool>,
    param_offsets: Vec<usize>,
    param_total: usize,
    input: NodeId,
    target: Option<NodeId>,
}

fn gerr<T>(msg: impl Into<String>) -> Result<T> {
    Err(CoreError::Graph(msg.into()))
}

impl Graph {
    pub fn new(def: GraphDef) -> Result<Graph> {
        if def.nodes.is_empty() || def.nodes.len() > MAX_NODES {
            return gerr(format!(
                "node count {} outside 1..={MAX_NODES}",
                def.nodes.len()
            ));
        }
        if def.params.len() > MAX_PARAMS {
            return gerr(format!(
                "parameter count {} exceeds {MAX_PARAMS}",
                def.params.len()
            ));
        }
        let mut param_offsets = Vec::with_capacity(def.params.len());
        let mut total: usize = 0;
        let mut seen = std::collections::HashSet::new();
        for p in &def.params {
            check_name(&p.name)?;
            if !seen.insert(p.name.as_str()) {
                return gerr(format!("duplicate parameter name '{}'", p.name));
            }
            let n = Shape::fixed(p.rows, p.cols)?.elements(1)?;
            param_offsets.push(total);
            total = total
                .checked_add(n)
                .ok_or(CoreError::Overflow("parameter elements"))?;
        }
        if total > crate::shape::MAX_ELEMENTS {
            return gerr("total parameter elements exceed the supported maximum");
        }

        let mut shapes: Vec<Shape> = Vec::with_capacity(def.nodes.len());
        let mut requires_grad = Vec::with_capacity(def.nodes.len());
        let mut input = None;
        let mut target = None;
        let mut node_names = std::collections::HashSet::new();
        for (id, node) in def.nodes.iter().enumerate() {
            check_name(&node.name)?;
            if !node_names.insert(node.name.as_str()) {
                return gerr(format!("duplicate node name '{}'", node.name));
            }
            for operand in node.op.input_list() {
                if operand >= id {
                    return gerr(format!(
                        "node {id} '{}' references node {operand}, which is not earlier in topological order",
                        node.name
                    ));
                }
            }
            let ctx = |m: String| CoreError::Shape(format!("node {id} '{}': {m}", node.name));
            let (shape, rg) = match node.op {
                Op::Input { cols } => {
                    if input.replace(id).is_some() {
                        return gerr("graph must contain exactly one input node");
                    }
                    (Shape::batch(cols)?, false)
                }
                Op::Target { cols } => {
                    if target.replace(id).is_some() {
                        return gerr("graph may contain at most one target node");
                    }
                    (Shape::batch(cols)?, false)
                }
                Op::Param { param } => {
                    let p = def
                        .params
                        .get(param)
                        .ok_or_else(|| CoreError::Graph(format!("unknown parameter {param}")))?;
                    (Shape::fixed(p.rows, p.cols)?, true)
                }
                Op::MatMul { a, b, ta, tb } => {
                    let sa = if ta {
                        shapes[a].transposed()
                    } else {
                        shapes[a]
                    };
                    let sb = if tb {
                        shapes[b].transposed()
                    } else {
                        shapes[b]
                    };
                    if sa.cols != sb.rows {
                        return Err(ctx(format!("matmul inner dimensions differ: {sa} @ {sb}")));
                    }
                    (
                        Shape::new(sa.rows, sb.cols)?,
                        requires_grad[a] || requires_grad[b],
                    )
                }
                Op::Add { a, b } | Op::Mul { a, b } => {
                    if !broadcastable(shapes[a], shapes[b]) {
                        return Err(ctx(format!(
                            "operand {} cannot broadcast to {}",
                            shapes[b], shapes[a]
                        )));
                    }
                    (shapes[a], requires_grad[a] || requires_grad[b])
                }
                Op::Relu { x } | Op::Sigmoid { x } | Op::Tanh { x } => {
                    (shapes[x], requires_grad[x])
                }
                Op::Softmax { x } | Op::LogSoftmax { x } => (shapes[x], requires_grad[x]),
                Op::Transpose { x } => (shapes[x].transposed(), requires_grad[x]),
                Op::SumAll { x } | Op::MeanAll { x } => (Shape::SCALAR, requires_grad[x]),
                Op::MseLoss { pred, target: t }
                | Op::BceWithLogitsLoss {
                    logits: pred,
                    target: t,
                } => {
                    if shapes[pred] != shapes[t] {
                        return Err(ctx(format!(
                            "prediction {} and target {} shapes differ",
                            shapes[pred], shapes[t]
                        )));
                    }
                    if requires_grad[t] {
                        return Err(ctx("loss target must not depend on parameters".into()));
                    }
                    (Shape::SCALAR, requires_grad[pred])
                }
                Op::CrossEntropyLoss { logits, target: t } => {
                    let (sl, st) = (shapes[logits], shapes[t]);
                    if st.rows != sl.rows || st.cols != Dim::Fixed(1) {
                        return Err(ctx(format!(
                            "cross-entropy target must be [{}, 1], got {st}",
                            sl.rows
                        )));
                    }
                    match sl.cols {
                        Dim::Fixed(k) if k >= 2 => {}
                        _ => {
                            return Err(ctx(format!(
                                "cross-entropy needs >= 2 fixed classes, got {sl}"
                            )));
                        }
                    }
                    if requires_grad[t] {
                        return Err(ctx("loss target must not depend on parameters".into()));
                    }
                    (Shape::SCALAR, requires_grad[logits])
                }
            };
            shapes.push(shape);
            requires_grad.push(rg);
        }
        let input = match input {
            Some(i) => i,
            None => return gerr("graph must contain exactly one input node"),
        };
        if def.output >= def.nodes.len() {
            return gerr("output node out of range");
        }
        if let Some(l) = def.loss {
            if l >= def.nodes.len() || !shapes[l].is_scalar() {
                return gerr("loss must reference a [1, 1] node");
            }
            if !requires_grad[l] {
                return gerr("loss does not depend on any parameter");
            }
        }
        let graph = Graph {
            def,
            shapes,
            requires_grad,
            param_offsets,
            param_total: total,
            input,
            target,
        };
        let reach_out = graph.ancestors(graph.def.output);
        if let Some(t) = graph.target {
            if reach_out[t] {
                return gerr("inference output must not depend on the training target");
            }
        }
        let reach_loss = graph.def.loss.map(|l| graph.ancestors(l));
        for (id, n) in graph.def.nodes.iter().enumerate() {
            let used = reach_out[id] || reach_loss.as_ref().is_some_and(|r| r[id]);
            if !used {
                return gerr(format!(
                    "node {id} '{}' is not used by the output or loss",
                    n.name
                ));
            }
        }
        for (pi, p) in graph.def.params.iter().enumerate() {
            let used = graph
                .def
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::Param { param } if param == pi));
            if !used {
                return gerr(format!("parameter '{}' is not referenced", p.name));
            }
        }
        Ok(graph)
    }

    /// Mask of nodes the given root depends on (including itself).
    pub fn ancestors(&self, root: NodeId) -> Vec<bool> {
        let mut mask = vec![false; self.def.nodes.len()];
        mask[root] = true;
        for id in (0..=root).rev() {
            if mask[id] {
                for i in self.def.nodes[id].op.input_list() {
                    mask[i] = true;
                }
            }
        }
        mask
    }

    pub fn def(&self) -> &GraphDef {
        &self.def
    }
    pub fn nodes(&self) -> &[NodeDef] {
        &self.def.nodes
    }
    pub fn op(&self, id: NodeId) -> &Op {
        &self.def.nodes[id].op
    }
    pub fn shape(&self, id: NodeId) -> Shape {
        self.shapes[id]
    }
    pub fn requires_grad(&self, id: NodeId) -> bool {
        self.requires_grad[id]
    }
    pub fn params(&self) -> &[ParamSpec] {
        &self.def.params
    }
    /// Offset of parameter `p` in the flat parameter store.
    pub fn param_offset(&self, p: usize) -> usize {
        self.param_offsets[p]
    }
    pub fn param_len(&self, p: usize) -> usize {
        self.def.params[p].rows * self.def.params[p].cols
    }
    /// Total number of f32 parameters.
    pub fn param_count(&self) -> usize {
        self.param_total
    }
    pub fn input(&self) -> NodeId {
        self.input
    }
    pub fn input_cols(&self) -> usize {
        self.shapes[self.input].cols.resolve(0)
    }
    pub fn target(&self) -> Option<NodeId> {
        self.target
    }
    pub fn output(&self) -> NodeId {
        self.def.output
    }
    pub fn output_shape(&self) -> Shape {
        self.shapes[self.def.output]
    }
    pub fn loss(&self) -> Option<NodeId> {
        self.def.loss
    }
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > MAX_NAME_LEN
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    {
        return gerr(format!(
            "name '{}' must be 1..={MAX_NAME_LEN} chars of [A-Za-z0-9_.-]",
            name.chars().take(MAX_NAME_LEN).collect::<String>()
        ));
    }
    Ok(())
}

/// Ergonomic construction of a [`GraphDef`] in Rust code.
///
/// ```
/// use boundml_core::graph::GraphBuilder;
/// let mut g = GraphBuilder::new();
/// let x = g.input("x", 3);
/// let w = g.param("w", 3, 1, true);
/// let b = g.param("b", 1, 1, false);
/// let xw = g.matmul("xw", x, w);
/// let out = g.add("out", xw, b);
/// let y = g.target("y", 1);
/// let loss = g.mse_loss("loss", out, y);
/// let graph = g.build(out, Some(loss)).unwrap();
/// assert_eq!(graph.param_count(), 4);
/// ```
#[derive(Debug, Default)]
pub struct GraphBuilder {
    params: Vec<ParamSpec>,
    nodes: Vec<NodeDef>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node(&mut self, name: impl Into<String>, op: Op) -> NodeId {
        self.nodes.push(NodeDef {
            name: name.into(),
            op,
        });
        self.nodes.len() - 1
    }

    pub fn input(&mut self, name: &str, cols: usize) -> NodeId {
        self.node(name, Op::Input { cols })
    }
    pub fn target(&mut self, name: &str, cols: usize) -> NodeId {
        self.node(name, Op::Target { cols })
    }
    /// Declares a parameter and returns the node that reads it.
    pub fn param(&mut self, name: &str, rows: usize, cols: usize, decay: bool) -> NodeId {
        self.params.push(ParamSpec {
            name: name.to_string(),
            rows,
            cols,
            decay,
        });
        let param = self.params.len() - 1;
        self.node(name, Op::Param { param })
    }
    pub fn matmul(&mut self, name: &str, a: NodeId, b: NodeId) -> NodeId {
        self.node(
            name,
            Op::MatMul {
                a,
                b,
                ta: false,
                tb: false,
            },
        )
    }
    pub fn matmul_t(&mut self, name: &str, a: NodeId, b: NodeId, ta: bool, tb: bool) -> NodeId {
        self.node(name, Op::MatMul { a, b, ta, tb })
    }
    pub fn add(&mut self, name: &str, a: NodeId, b: NodeId) -> NodeId {
        self.node(name, Op::Add { a, b })
    }
    pub fn mul(&mut self, name: &str, a: NodeId, b: NodeId) -> NodeId {
        self.node(name, Op::Mul { a, b })
    }
    pub fn relu(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::Relu { x })
    }
    pub fn sigmoid(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::Sigmoid { x })
    }
    pub fn tanh(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::Tanh { x })
    }
    pub fn softmax(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::Softmax { x })
    }
    pub fn log_softmax(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::LogSoftmax { x })
    }
    pub fn transpose(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::Transpose { x })
    }
    pub fn sum_all(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::SumAll { x })
    }
    pub fn mean_all(&mut self, name: &str, x: NodeId) -> NodeId {
        self.node(name, Op::MeanAll { x })
    }
    pub fn mse_loss(&mut self, name: &str, pred: NodeId, target: NodeId) -> NodeId {
        self.node(name, Op::MseLoss { pred, target })
    }
    pub fn bce_with_logits_loss(&mut self, name: &str, logits: NodeId, target: NodeId) -> NodeId {
        self.node(name, Op::BceWithLogitsLoss { logits, target })
    }
    pub fn cross_entropy_loss(&mut self, name: &str, logits: NodeId, target: NodeId) -> NodeId {
        self.node(name, Op::CrossEntropyLoss { logits, target })
    }

    pub fn into_def(self, output: NodeId, loss: Option<NodeId>) -> GraphDef {
        GraphDef {
            params: self.params,
            nodes: self.nodes,
            output,
            loss,
        }
    }

    pub fn build(self, output: NodeId, loss: Option<NodeId>) -> Result<Graph> {
        Graph::new(self.into_def(output, loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_forward_reference_and_bad_shapes() {
        let def = GraphDef {
            params: vec![],
            nodes: vec![
                NodeDef {
                    name: "x".into(),
                    op: Op::Input { cols: 2 },
                },
                NodeDef {
                    name: "r".into(),
                    op: Op::Relu { x: 2 },
                },
            ],
            output: 1,
            loss: None,
        };
        assert!(matches!(Graph::new(def), Err(CoreError::Graph(_))));

        let mut g = GraphBuilder::new();
        let x = g.input("x", 3);
        let w = g.param("w", 4, 2, true);
        let o = g.matmul("o", x, w);
        assert!(matches!(g.build(o, None), Err(CoreError::Shape(_))));
    }

    #[test]
    fn rejects_unused_nodes_and_target_in_output() {
        let mut g = GraphBuilder::new();
        let x = g.input("x", 2);
        let _dangling = g.relu("dangling", x);
        let w = g.param("w", 2, 1, true);
        let o = g.matmul("o", x, w);
        assert!(g.build(o, None).is_err());

        let mut g = GraphBuilder::new();
        let x = g.input("x", 1);
        let t = g.target("t", 1);
        let w = g.param("w", 1, 1, true);
        let xw = g.matmul("xw", x, w);
        let o = g.add("o", xw, t);
        assert!(g.build(o, None).is_err());
    }

    #[test]
    fn json_roundtrip_and_unknown_fields() {
        let mut g = GraphBuilder::new();
        let x = g.input("x", 2);
        let w = g.param("w", 2, 3, true);
        let h = g.matmul("h", x, w);
        let s = g.softmax("s", h);
        let def = g.into_def(s, None);
        let json = serde_json::to_string(&def).unwrap();
        let back: GraphDef = serde_json::from_str(&json).unwrap();
        assert_eq!(def, back);
        let bad = json.replace("\"op\":\"softmax\"", "\"op\":\"conv2d\"");
        assert!(serde_json::from_str::<GraphDef>(&bad).is_err());
        let extra = json.replace("\"op\":\"softmax\"", "\"op\":\"softmax\",\"axis\":0");
        assert!(serde_json::from_str::<GraphDef>(&extra).is_err());
    }

    #[test]
    fn rejects_invalid_names() {
        let mut g = GraphBuilder::new();
        let x = g.input("bad name", 2);
        let w = g.param("w", 2, 1, true);
        let o = g.matmul("o", x, w);
        assert!(g.build(o, None).is_err());
    }
}
