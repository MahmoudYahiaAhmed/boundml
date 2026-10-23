//! SGD and AdamW updates vs an independent f64 transcription of the published
//! algorithms (Loshchilov & Hutter, "Decoupled Weight Decay Regularization";
//! PyTorch SGD momentum convention).

use boundml_core::{GraphBuilder, Optimizer, OptimizerConfig, OptimizerState};

fn graph() -> boundml_core::Graph {
    let mut g = GraphBuilder::new();
    let x = g.input("x", 2);
    let w = g.param("w", 2, 1, true);
    let b = g.param("b", 1, 1, false);
    let xw = g.matmul("xw", x, w);
    let o = g.add("o", xw, b);
    g.build(o, None).unwrap()
}

const DECAY: [bool; 3] = [true, true, false];
const GRADS: [[f32; 3]; 4] = [
    [0.5, -1.0, 0.25],
    [0.1, 0.2, -0.3],
    [-2.0, 0.0, 1.0],
    [1e-3, 3.0, -0.7],
];

#[test]
fn adamw_matches_reference() {
    let (lr, b1, b2, eps, wd) = (0.01f64, 0.9f64, 0.99f64, 1e-8f64, 0.1f64);
    let cfg = OptimizerConfig::AdamW {
        lr: lr as f32,
        beta1: b1 as f32,
        beta2: b2 as f32,
        eps: eps as f32,
        weight_decay: wd as f32,
    };
    let g = graph();
    let mut opt = Optimizer::new(cfg, &g).unwrap();
    let mut p32 = vec![1.0f32, -0.5, 0.2];
    let mut p = vec![1.0f64, -0.5, 0.2];
    let (mut m, mut v) = (vec![0.0f64; 3], vec![0.0f64; 3]);
    for (t, grads) in GRADS.iter().enumerate() {
        opt.step(&mut p32, grads).unwrap();
        let t = (t + 1) as i32;
        for i in 0..3 {
            let gi = grads[i] as f64;
            if DECAY[i] {
                p[i] -= lr * wd * p[i];
            }
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
            let mhat = m[i] / (1.0 - b1.powi(t));
            let vhat = v[i] / (1.0 - b2.powi(t));
            p[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
        for i in 0..3 {
            assert!(
                (p32[i] as f64 - p[i]).abs() < 1e-6,
                "step {t} param {i}: {} vs {}",
                p32[i],
                p[i]
            );
        }
    }
    assert_eq!(opt.steps(), 4);
}

#[test]
fn sgd_momentum_matches_reference() {
    let (lr, mu, wd) = (0.1f64, 0.9f64, 0.05f64);
    let cfg = OptimizerConfig::Sgd {
        lr: lr as f32,
        momentum: mu as f32,
        weight_decay: wd as f32,
    };
    let mut opt = Optimizer::new(cfg, &graph()).unwrap();
    let mut p32 = vec![1.0f32, -0.5, 0.2];
    let mut p = vec![1.0f64, -0.5, 0.2];
    let mut buf = vec![0.0f64; 3];
    for grads in GRADS {
        opt.step(&mut p32, &grads).unwrap();
        for i in 0..3 {
            let g = grads[i] as f64 + if DECAY[i] { wd * p[i] } else { 0.0 };
            buf[i] = mu * buf[i] + g;
            p[i] -= lr * buf[i];
        }
        for i in 0..3 {
            assert!((p32[i] as f64 - p[i]).abs() < 1e-6);
        }
    }
}

#[test]
fn plain_sgd_and_state_roundtrip() {
    let cfg = OptimizerConfig::AdamW {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let g = graph();
    let mut a = Optimizer::new(cfg.clone(), &g).unwrap();
    let mut pa = vec![0.3f32, 0.1, -0.2];
    a.step(&mut pa, &GRADS[0]).unwrap();
    // Resume into a fresh optimizer from saved state; subsequent updates must agree.
    let OptimizerState { step, slots } = a.state();
    let mut b = Optimizer::new(cfg, &g).unwrap();
    b.restore(OptimizerState { step, slots }).unwrap();
    let mut pb = pa.clone();
    for grads in &GRADS[1..] {
        a.step(&mut pa, grads).unwrap();
        b.step(&mut pb, grads).unwrap();
    }
    assert_eq!(pa, pb);

    let mut s = Optimizer::new(
        OptimizerConfig::Sgd {
            lr: 0.5,
            momentum: 0.0,
            weight_decay: 0.0,
        },
        &g,
    )
    .unwrap();
    let mut p = vec![1.0f32, 1.0, 1.0];
    s.step(&mut p, &[1.0, 2.0, -2.0]).unwrap();
    assert_eq!(p, vec![0.5, 0.0, 2.0]);
}

#[test]
fn invalid_configs_and_nonfinite_grads_rejected() {
    let g = graph();
    for cfg in [
        OptimizerConfig::Sgd {
            lr: 0.0,
            momentum: 0.0,
            weight_decay: 0.0,
        },
        OptimizerConfig::Sgd {
            lr: 0.1,
            momentum: 1.0,
            weight_decay: 0.0,
        },
        OptimizerConfig::AdamW {
            lr: 0.1,
            beta1: 1.0,
            beta2: 0.9,
            eps: 1e-8,
            weight_decay: 0.0,
        },
        OptimizerConfig::AdamW {
            lr: f32::NAN,
            beta1: 0.9,
            beta2: 0.9,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    ] {
        assert!(Optimizer::new(cfg, &g).is_err());
    }
    let mut o = Optimizer::new(
        OptimizerConfig::Sgd {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
        },
        &g,
    )
    .unwrap();
    let mut p = vec![0.0; 3];
    assert!(o.step(&mut p, &[f32::INFINITY, 0.0, 0.0]).is_err());
}
