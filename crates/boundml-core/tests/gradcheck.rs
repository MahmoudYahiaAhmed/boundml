//! Gradient and forward checks against an independent f64 reference with central
//! finite differences. Tolerances are for f32 production math vs f64 reference.

mod reference;

use boundml_core::graph::{GraphBuilder, GraphDef, Op};
use boundml_core::{Activation, DenseSpec, Executor, Graph, Mode, PlannerKind, Rng, Task};
use reference::M;
use std::sync::Arc;

const ATOL: f64 = 5e-4;
const RTOL: f64 = 5e-3;

fn rand_vec(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.uniform(-scale, scale)).collect()
}

fn to_m(r: usize, c: usize, v: &[f32]) -> M {
    M {
        r,
        c,
        d: v.iter().map(|x| *x as f64).collect(),
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (*g as f64 - w).abs();
        assert!(
            err <= ATOL + RTOL * w.abs(),
            "{what}[{i}]: got {g}, want {w} (err {err})"
        );
    }
}

/// Fails if any ReLU input is within `eps` of the kink, where finite differences are invalid.
fn assert_away_from_kinks(def: &GraphDef, vals: &[M]) {
    for n in &def.nodes {
        if let Op::Relu { x } = n.op {
            assert!(
                vals[x].d.iter().all(|v| v.abs() > 1e-3),
                "ReLU input too close to 0; change seed"
            );
        }
    }
}

fn check(def: GraphDef, batch: usize, y: Vec<f32>, seed: u64) {
    let graph = Graph::new(def.clone()).unwrap();
    let mut rng = Rng::seed(seed);
    let params = rand_vec(&mut rng, graph.param_count(), 0.8);
    let in_cols = graph.input_cols();
    let x = rand_vec(&mut rng, batch * in_cols, 1.5);

    let pf64: Vec<f64> = params.iter().map(|v| *v as f64).collect();
    let xm = to_m(batch, in_cols, &x);
    let ym = to_m(batch, y.len() / batch, &y);
    let vals = reference::eval(&def, &pf64, &xm, Some(&ym));
    assert_away_from_kinks(&def, &vals);
    let want_loss = vals[def.loss.unwrap()].d[0];
    let want_grad = reference::fd_grad(&def, &pf64, &xm, &ym, 1e-6);

    let g = Arc::new(graph);
    let mut results = Vec::new();
    for planner in [PlannerKind::Conservative, PlannerKind::Reuse] {
        let mut ex = Executor::new(Arc::clone(&g), Mode::Training, batch + 3, planner).unwrap();
        for _ in 0..2 {
            // Repeated calls must not accumulate stale gradients.
            let loss = ex.forward_backward(&params, &x, &y, batch).unwrap();
            assert_close("loss", &[loss], &[want_loss]);
            assert_close("grad", ex.param_grads(), &want_grad);
        }
        results.push(ex.param_grads().to_vec());

        let mut inf = Executor::new(Arc::clone(&g), Mode::Inference, batch, planner).unwrap();
        let out_shape = g.output_shape();
        let (r, c) = out_shape.dims(batch);
        let mut out = vec![0.0; r * c];
        inf.infer(&params, &x, batch, &mut out).unwrap();
        assert_close("output", &out, &vals[def.output].d);
    }
    assert_eq!(
        results[0], results[1],
        "planners must give identical results"
    );
}

fn dense(inputs: usize, hidden: Vec<usize>, activation: Activation, task: Task) -> GraphDef {
    DenseSpec {
        inputs,
        hidden,
        activation,
        task,
    }
    .graph_def()
    .unwrap()
}

#[test]
fn regression_mlp_tanh() {
    check(
        dense(3, vec![5, 4], Activation::Tanh, Task::Regression),
        6,
        vec![0.3, -1.0, 2.0, 0.0, 0.5, 1.5],
        1,
    );
}

#[test]
fn linear_regression() {
    check(
        dense(4, vec![], Activation::Relu, Task::Regression),
        5,
        vec![1.0, 2.0, -1.0, 0.0, 0.25],
        2,
    );
}

#[test]
fn binary_mlp_relu() {
    check(
        dense(4, vec![8], Activation::Relu, Task::Binary),
        7,
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        3,
    );
}

#[test]
fn logistic_regression() {
    check(
        dense(3, vec![], Activation::Relu, Task::Binary),
        4,
        vec![1.0, 0.0, 0.0, 1.0],
        4,
    );
}

#[test]
fn multiclass_mlp_sigmoid() {
    let task = Task::Multiclass { classes: 4 };
    check(
        dense(3, vec![6], Activation::Sigmoid, task),
        5,
        vec![0.0, 3.0, 1.0, 2.0, 3.0],
        5,
    );
}

#[test]
fn multiclass_two_layer_relu() {
    let task = Task::Multiclass { classes: 3 };
    check(
        dense(5, vec![7, 6], Activation::Relu, task),
        8,
        vec![0., 1., 2., 0., 1., 2., 2., 1.],
        12,
    );
}

/// Shared parameters, fan-out activations, transposed matmuls, broadcasting,
/// softmax/log-softmax and reductions in one graph.
#[test]
fn fan_out_and_all_ops() {
    let mut g = GraphBuilder::new();
    let x = g.input("x", 3);
    let w = g.param("w", 3, 3, true);
    let bias = g.param("bias", 1, 3, false);
    let scale = g.param("scale", 1, 1, false);
    let colw = g.param("colw", 3, 1, true);
    let rowb = g.param("rowb", 1, 1, false);
    let h = g.matmul("h", x, w);
    let h2 = g.matmul("h2", h, w); // w used twice
    let m = g.mul("m", h2, h); // h used twice
    let s = g.add("s", m, bias); // row broadcast
    let sc = g.mul("sc", s, scale); // scalar broadcast
    let th = g.tanh("th", sc);
    let ls = g.log_softmax("ls", th);
    let mean_ls = g.mean_all("mean_ls", ls);
    let sm = g.softmax("sm", h);
    let r = g.matmul_t("r", sm, w, false, true); // sm @ w^T
    let r2 = g.matmul_t("r2", w, r, true, true); // w^T @ r^T -> [3, B]
    let r2t = g.transpose("r2t", r2);
    let sum_r = g.sum_all("sum_r", r2t);
    let q = g.matmul_t("q", h, h, true, false); // h^T h, fan-out on both operands
    let sum_q = g.mean_all("sum_q", q);
    let sq = g.mul("sq", x, x); // input fan-out (no grad)
    let pre = g.matmul("pre", sq, colw);
    let pred = g.add("pred", pre, rowb);
    let sig = g.sigmoid("sig", pred);
    let y = g.target("y", 1);
    let mse = g.mse_loss("mse", sig, y);
    let a1 = g.add("a1", mean_ls, sum_r);
    let a2 = g.add("a2", a1, sum_q);
    let loss = g.add("loss", a2, mse);
    let def = g.into_def(ls, Some(loss));
    check(def, 4, vec![0.1, 0.9, 0.4, 0.5], 9);
}

#[test]
fn degenerate_batch_of_one_and_extreme_logits() {
    let def = dense(2, vec![], Activation::Relu, Task::Binary);
    let graph = Arc::new(Graph::new(def).unwrap());
    let mut ex = Executor::new(Arc::clone(&graph), Mode::Training, 4, PlannerKind::Reuse).unwrap();
    // Huge weights => logits around +/-1e4. Loss and gradients must stay finite.
    let params = vec![1e4, -1e4, 0.0];
    let loss = ex
        .forward_backward(&params, &[1.0, 0.0], &[0.0], 1)
        .unwrap();
    assert!((loss - 1e4).abs() / 1e4 < 1e-4, "loss {loss}");
    assert!(ex.param_grads().iter().all(|g| g.is_finite()));
}

#[test]
fn rejects_bad_inputs_cleanly() {
    let def = dense(
        2,
        vec![3],
        Activation::Relu,
        Task::Multiclass { classes: 3 },
    );
    let graph = Arc::new(Graph::new(def).unwrap());
    let params = vec![0.1; graph.param_count()];
    let mut ex = Executor::new(Arc::clone(&graph), Mode::Training, 2, PlannerKind::Reuse).unwrap();
    assert!(
        ex.forward_backward(&params, &[0.0; 6], &[0.0; 3], 3)
            .is_err(),
        "batch > max"
    );
    assert!(
        ex.forward_backward(&params, &[f32::NAN, 0.0], &[0.0], 1)
            .is_err(),
        "NaN input"
    );
    assert!(
        ex.forward_backward(&params, &[0.0, 0.0], &[3.0], 1)
            .is_err(),
        "class out of range"
    );
    assert!(
        ex.forward_backward(&params[1..], &[0.0, 0.0], &[0.0], 1)
            .is_err(),
        "param length"
    );
    let mut inf = Executor::new(graph, Mode::Inference, 2, PlannerKind::Reuse).unwrap();
    let mut out = vec![0.0; 2];
    assert!(
        inf.infer(&params, &[0.0, 0.0], 1, &mut out).is_err(),
        "output length"
    );
}
