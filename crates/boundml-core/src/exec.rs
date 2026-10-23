//! Plan execution: forward and reverse-mode backward over one preallocated arena.
//!
//! An [`Executor`] owns a [`Plan`] and the arena sized by it; it is the mutable,
//! per-session state. Parameters are borrowed per call, so many executors can share
//! one immutable weight buffer. After construction, `infer`, `forward_backward` and
//! `loss` perform no heap allocation (verified by `tests/alloc_free.rs`).
//!
//! Graph lifetime and gradient semantics: every `forward_backward` call zeroes all
//! gradients first, recomputes activations, and leaves parameter gradients readable
//! via [`Executor::param_grads`] until the next call. Gradients are never accumulated
//! across calls; activations are only retained within a single call.

use crate::arena::{Src, split};
use crate::error::{CoreError, Result};
use crate::graph::{Graph, NodeId, Op};
use crate::kernels::{self as k, Bcast};
use crate::plan::{Mode, Plan, PlannerKind};
use std::sync::Arc;

#[derive(Debug)]
pub struct Executor {
    graph: Arc<Graph>,
    plan: Plan,
    arena: Vec<f32>,
}

impl Executor {
    /// Plan and allocate. Use [`Executor::from_plan`] to check a budget in between.
    pub fn new(
        graph: Arc<Graph>,
        mode: Mode,
        max_batch: usize,
        planner: PlannerKind,
    ) -> Result<Self> {
        let plan = Plan::new(&graph, mode, max_batch, planner)?;
        Ok(Self::from_plan(graph, plan))
    }

    /// Allocate the arena for an already-checked plan. This is the only allocation of
    /// tensor storage an executor performs.
    pub fn from_plan(graph: Arc<Graph>, plan: Plan) -> Self {
        let arena = vec![0.0f32; plan.arena_elems];
        Executor { graph, plan, arena }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }

    fn elems(&self, n: NodeId, b: usize) -> usize {
        let (r, c) = self.graph.shape(n).dims(b);
        r * c
    }

    fn value_src(&self, n: NodeId, b: usize) -> Src {
        match *self.graph.op(n) {
            Op::Param { param } => {
                let o = self.graph.param_offset(param);
                Src::Param(o..o + self.graph.param_len(param))
            }
            _ => Src::Arena(self.value_range(n, b)),
        }
    }

    fn value_range(&self, n: NodeId, b: usize) -> std::ops::Range<usize> {
        let buf = self.plan.value_buf[n].expect("planned value buffer");
        self.plan.range(buf, self.elems(n, b))
    }

    fn grad_range(&self, n: NodeId, b: usize) -> std::ops::Range<usize> {
        match *self.graph.op(n) {
            Op::Param { param } => {
                let base = self.plan.buffers[self.plan.param_grad_buf.expect("param grads")].offset;
                let o = base + self.graph.param_offset(param);
                o..o + self.graph.param_len(param)
            }
            _ => {
                let buf = self.plan.grad_buf[n].expect("planned gradient buffer");
                self.plan.range(buf, self.elems(n, b))
            }
        }
    }

    fn check_batch(&self, b: usize) -> Result<()> {
        if b == 0 || b > self.plan.max_batch {
            return Err(CoreError::BatchOutOfRange {
                got: b,
                max: self.plan.max_batch,
            });
        }
        Ok(())
    }

    fn check_params(&self, params: &[f32]) -> Result<()> {
        if params.len() != self.graph.param_count() {
            return Err(CoreError::Length {
                what: "parameters",
                expected: self.graph.param_count(),
                got: params.len(),
            });
        }
        Ok(())
    }

    fn load(&mut self, node: NodeId, data: &[f32], b: usize, what: &'static str) -> Result<()> {
        let r = self.value_range(node, b);
        if data.len() != r.len() {
            return Err(CoreError::Length {
                what,
                expected: r.len(),
                got: data.len(),
            });
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::NonFinite(what));
        }
        self.arena[r].copy_from_slice(data);
        Ok(())
    }

    /// Run the inference plan and copy the output into `out` (`b * output_cols`).
    pub fn infer(&mut self, params: &[f32], x: &[f32], b: usize, out: &mut [f32]) -> Result<()> {
        if self.plan.mode != Mode::Inference {
            return Err(CoreError::Graph(
                "executor was planned for training, not inference".into(),
            ));
        }
        self.check_batch(b)?;
        self.check_params(params)?;
        let output = self.graph.output();
        let n_out = self.elems(output, b);
        if out.len() != n_out {
            return Err(CoreError::Length {
                what: "output",
                expected: n_out,
                got: out.len(),
            });
        }
        self.load(self.graph.input(), x, b, "input")?;
        self.forward(params, b)?;
        let r = self.value_range(output, b);
        out.copy_from_slice(&self.arena[r]);
        Ok(())
    }

    /// Forward pass of the training plan only; returns the loss.
    pub fn loss(&mut self, params: &[f32], x: &[f32], y: &[f32], b: usize) -> Result<f32> {
        self.prepare_training(params, x, y, b)?;
        self.forward(params, b)?;
        self.read_loss(b)
    }

    /// Forward + backward. Parameter gradients are then available via `param_grads`.
    pub fn forward_backward(
        &mut self,
        params: &[f32],
        x: &[f32],
        y: &[f32],
        b: usize,
    ) -> Result<f32> {
        self.prepare_training(params, x, y, b)?;
        self.forward(params, b)?;
        let loss = self.read_loss(b)?;
        self.backward(params, b)?;
        Ok(loss)
    }

    /// Gradient of the loss w.r.t. the flat parameter vector from the last backward.
    pub fn param_grads(&self) -> &[f32] {
        match self.plan.param_grad_buf {
            Some(buf) => {
                let o = self.plan.buffers[buf].offset;
                &self.arena[o..o + self.graph.param_count()]
            }
            None => &[],
        }
    }

    fn prepare_training(&mut self, params: &[f32], x: &[f32], y: &[f32], b: usize) -> Result<()> {
        if self.plan.mode != Mode::Training {
            return Err(CoreError::Graph(
                "executor was planned for inference, not training".into(),
            ));
        }
        self.check_batch(b)?;
        self.check_params(params)?;
        self.load(self.graph.input(), x, b, "input")?;
        let t = self
            .graph
            .target()
            .expect("validated: training graphs have a target");
        self.load(t, y, b, "target")
    }

    fn read_loss(&self, b: usize) -> Result<f32> {
        let l = self.graph.loss().expect("validated");
        let v = self.arena[self.value_range(l, b).start];
        if !v.is_finite() {
            return Err(CoreError::NonFinite("loss"));
        }
        Ok(v)
    }

    fn forward(&mut self, params: &[f32], b: usize) -> Result<()> {
        for i in 0..self.plan.forward.len() {
            let n = self.plan.forward[i];
            self.forward_node(params, n, b)?;
        }
        Ok(())
    }

    fn forward_node(&mut self, params: &[f32], n: NodeId, b: usize) -> Result<()> {
        let g = Arc::clone(&self.graph);
        let op = g.op(n);
        if matches!(op, Op::Input { .. } | Op::Target { .. } | Op::Param { .. }) {
            return Ok(());
        }
        let w = self.value_range(n, b);
        match *op {
            Op::MatMul { a, b: bb, ta, tb } => {
                let (m, kk, nn) = mm_dims(&g, a, bb, ta, tb, b);
                let (sa, sb) = (self.value_src(a, b), self.value_src(bb, b));
                let ([va, vb], out) = split(&mut self.arena, params, [sa, sb], w);
                k::gemm(m, nn, kk, va, ta, vb, tb, out, false);
            }
            Op::Add { a, b: bb } | Op::Mul { a, b: bb } => {
                let bc = bcast(&g, a, bb, b);
                let (sa, sb) = (self.value_src(a, b), self.value_src(bb, b));
                let ([va, vb], out) = split(&mut self.arena, params, [sa, sb], w);
                if matches!(op, Op::Add { .. }) {
                    k::add_fwd(va, vb, bc, out);
                } else {
                    k::mul_fwd(va, vb, bc, out);
                }
            }
            Op::Relu { x }
            | Op::Sigmoid { x }
            | Op::Tanh { x }
            | Op::Softmax { x }
            | Op::LogSoftmax { x }
            | Op::Transpose { x }
            | Op::SumAll { x }
            | Op::MeanAll { x } => {
                let (r, c) = g.shape(x).dims(b);
                let sx = self.value_src(x, b);
                let ([vx], out) = split(&mut self.arena, params, [sx], w);
                match op {
                    Op::Relu { .. } => k::relu_fwd(vx, out),
                    Op::Sigmoid { .. } => k::sigmoid_fwd(vx, out),
                    Op::Tanh { .. } => k::tanh_fwd(vx, out),
                    Op::Softmax { .. } => k::softmax_fwd(vx, c, out),
                    Op::LogSoftmax { .. } => k::log_softmax_fwd(vx, c, out),
                    Op::Transpose { .. } => k::transpose(vx, r, c, out, false),
                    Op::SumAll { .. } => out[0] = k::sum_all(vx),
                    _ => out[0] = k::sum_all(vx) / vx.len() as f32,
                }
            }
            Op::MseLoss { pred: p, target: t }
            | Op::BceWithLogitsLoss {
                logits: p,
                target: t,
            }
            | Op::CrossEntropyLoss {
                logits: p,
                target: t,
            } => {
                let classes = g.shape(p).dims(b).1;
                let (sp, st) = (self.value_src(p, b), self.value_src(t, b));
                let ([vp, vt], out) = split(&mut self.arena, params, [sp, st], w);
                out[0] = match op {
                    Op::MseLoss { .. } => k::mse_fwd(vp, vt),
                    Op::BceWithLogitsLoss { .. } => k::bce_logits_fwd(vp, vt),
                    _ => k::cross_entropy_fwd(vp, vt, classes)?,
                };
            }
            Op::Input { .. } | Op::Target { .. } | Op::Param { .. } => {}
        }
        Ok(())
    }

    fn backward(&mut self, params: &[f32], b: usize) -> Result<()> {
        if let Some(buf) = self.plan.param_grad_buf {
            let o = self.plan.buffers[buf].offset;
            self.arena[o..o + self.graph.param_count()].fill(0.0);
        }
        let loss = self.graph.loss().expect("validated");
        for pos in 0..self.plan.backward.len() {
            let (from, to) = self.plan.zero_ranges[pos];
            for zi in from..to {
                let buf = self.plan.zero_list[zi];
                let node = match self.plan.buffers[buf].kind {
                    crate::plan::BufKind::Grad(n) => n,
                    _ => continue,
                };
                let r = self.grad_range(node, b);
                self.arena[r].fill(0.0);
            }
            let n = self.plan.backward[pos];
            if n == loss {
                let r = self.grad_range(n, b);
                self.arena[r.start] = 1.0;
            }
            self.backward_node(params, n, b)?;
        }
        Ok(())
    }

    fn backward_node(&mut self, params: &[f32], n: NodeId, b: usize) -> Result<()> {
        let g = Arc::clone(&self.graph);
        let op = g.op(n);
        let gn = Src::Arena(self.grad_range(n, b));
        match *op {
            Op::Input { .. } | Op::Target { .. } | Op::Param { .. } => {}
            Op::MatMul { a, b: bb, ta, tb } => {
                let (m, kk, nn) = mm_dims(&g, a, bb, ta, tb, b);
                if g.requires_grad(a) {
                    let w = self.grad_range(a, b);
                    let sb = self.value_src(bb, b);
                    let ([gv, vb], ga) = split(&mut self.arena, params, [gn.clone(), sb], w);
                    if !ta {
                        k::gemm(m, kk, nn, gv, false, vb, !tb, ga, true);
                    } else {
                        k::gemm(kk, m, nn, vb, tb, gv, true, ga, true);
                    }
                }
                if g.requires_grad(bb) {
                    let w = self.grad_range(bb, b);
                    let sa = self.value_src(a, b);
                    let ([gv, va], gb) = split(&mut self.arena, params, [gn, sa], w);
                    if !tb {
                        k::gemm(kk, nn, m, va, !ta, gv, false, gb, true);
                    } else {
                        k::gemm(nn, kk, m, gv, true, va, ta, gb, true);
                    }
                }
            }
            Op::Add { a, b: bb } => {
                let bc = bcast(&g, a, bb, b);
                if g.requires_grad(a) {
                    let w = self.grad_range(a, b);
                    let ([gv], ga) = split(&mut self.arena, params, [gn.clone()], w);
                    k::acc(ga, gv);
                }
                if g.requires_grad(bb) {
                    let w = self.grad_range(bb, b);
                    let ([gv], gb) = split(&mut self.arena, params, [gn], w);
                    k::acc_reduce(gb, gv, bc);
                }
            }
            Op::Mul { a, b: bb } => {
                let bc = bcast(&g, a, bb, b);
                if g.requires_grad(a) {
                    let w = self.grad_range(a, b);
                    let sb = self.value_src(bb, b);
                    let ([gv, vb], ga) = split(&mut self.arena, params, [gn.clone(), sb], w);
                    k::mul_bwd_a(ga, gv, vb, bc);
                }
                if g.requires_grad(bb) {
                    let w = self.grad_range(bb, b);
                    let sa = self.value_src(a, b);
                    let ([gv, va], gb) = split(&mut self.arena, params, [gn, sa], w);
                    k::mul_bwd_b(gb, gv, va, bc);
                }
            }
            Op::Relu { x }
            | Op::Sigmoid { x }
            | Op::Tanh { x }
            | Op::Softmax { x }
            | Op::LogSoftmax { x } => {
                if !g.requires_grad(x) {
                    return Ok(());
                }
                let c = g.shape(x).dims(b).1;
                let w = self.grad_range(x, b);
                let sy = self.value_src(n, b);
                let ([gv, vy], gx) = split(&mut self.arena, params, [gn, sy], w);
                match op {
                    Op::Relu { .. } => k::relu_bwd(gx, gv, vy),
                    Op::Sigmoid { .. } => k::sigmoid_bwd(gx, gv, vy),
                    Op::Tanh { .. } => k::tanh_bwd(gx, gv, vy),
                    Op::Softmax { .. } => k::softmax_bwd(gx, gv, vy, c),
                    _ => k::log_softmax_bwd(gx, gv, vy, c),
                }
            }
            Op::Transpose { x } | Op::SumAll { x } | Op::MeanAll { x } => {
                if !g.requires_grad(x) {
                    return Ok(());
                }
                let (r, c) = g.shape(x).dims(b);
                let w = self.grad_range(x, b);
                let ([gv], gx) = split(&mut self.arena, params, [gn], w);
                match op {
                    Op::Transpose { .. } => k::transpose(gv, c, r, gx, true),
                    Op::SumAll { .. } => k::acc_scalar(gx, gv[0]),
                    _ => k::acc_scalar(gx, gv[0] / (r * c) as f32),
                }
            }
            Op::MseLoss { pred: p, target: t }
            | Op::BceWithLogitsLoss {
                logits: p,
                target: t,
            }
            | Op::CrossEntropyLoss {
                logits: p,
                target: t,
            } => {
                if !g.requires_grad(p) {
                    return Ok(());
                }
                let classes = g.shape(p).dims(b).1;
                let w = self.grad_range(p, b);
                let (sp, st) = (self.value_src(p, b), self.value_src(t, b));
                let ([gv, vp, vt], gp) = split(&mut self.arena, params, [gn, sp, st], w);
                match op {
                    Op::MseLoss { .. } => k::mse_bwd(gp, gv[0], vp, vt),
                    Op::BceWithLogitsLoss { .. } => k::bce_logits_bwd(gp, gv[0], vp, vt),
                    _ => k::cross_entropy_bwd(gp, gv[0], vp, vt, classes)?,
                }
            }
        }
        Ok(())
    }
}

/// Logical `(m, k, n)` of `op(a) @ op(b)` at batch `b`.
fn mm_dims(
    g: &Graph,
    a: NodeId,
    bb: NodeId,
    ta: bool,
    tb: bool,
    b: usize,
) -> (usize, usize, usize) {
    let (ra, ca) = g.shape(a).dims(b);
    let (rb, cb) = g.shape(bb).dims(b);
    let (m, kk) = if ta { (ca, ra) } else { (ra, ca) };
    let nn = if tb { rb } else { cb };
    (m, kk, nn)
}

fn bcast(g: &Graph, a: NodeId, bb: NodeId, b: usize) -> Bcast {
    let (rows, cols) = g.shape(a).dims(b);
    let (b_rows, b_cols) = g.shape(bb).dims(b);
    Bcast {
        rows,
        cols,
        b_rows,
        b_cols,
    }
}
