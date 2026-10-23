//! Independent f64 reference interpreter for `GraphDef`, used only by tests.
//! Written separately from the production kernels: naive loops, no shared code.

use boundml_core::graph::{GraphDef, Op};

#[derive(Clone, Debug)]
pub struct M {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f64>,
}

impl M {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }
    fn t(&self) -> M {
        let mut d = vec![0.0; self.d.len()];
        for i in 0..self.r {
            for j in 0..self.c {
                d[j * self.r + i] = self.at(i, j);
            }
        }
        M {
            r: self.c,
            c: self.r,
            d,
        }
    }
    fn map(&self, f: impl Fn(f64) -> f64) -> M {
        M {
            r: self.r,
            c: self.c,
            d: self.d.iter().map(|v| f(*v)).collect(),
        }
    }
    fn scalar(v: f64) -> M {
        M {
            r: 1,
            c: 1,
            d: vec![v],
        }
    }
}

fn bget(b: &M, i: usize, j: usize) -> f64 {
    b.at(if b.r == 1 { 0 } else { i }, if b.c == 1 { 0 } else { j })
}

fn lse(row: &[f64]) -> f64 {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

/// Evaluate all nodes; returns per-node values.
pub fn eval(def: &GraphDef, params: &[f64], x: &M, y: Option<&M>) -> Vec<M> {
    let mut offsets = vec![];
    let mut o = 0;
    for p in &def.params {
        offsets.push(o);
        o += p.rows * p.cols;
    }
    let mut v: Vec<M> = Vec::new();
    for n in &def.nodes {
        let out = match n.op {
            Op::Input { .. } => x.clone(),
            Op::Target { cols } => y.cloned().unwrap_or(M {
                r: x.r,
                c: cols,
                d: vec![0.0; x.r * cols],
            }),
            Op::Param { param } => {
                let p = &def.params[param];
                M {
                    r: p.rows,
                    c: p.cols,
                    d: params[offsets[param]..offsets[param] + p.rows * p.cols].to_vec(),
                }
            }
            Op::MatMul { a, b, ta, tb } => {
                let a = if ta { v[a].t() } else { v[a].clone() };
                let b = if tb { v[b].t() } else { v[b].clone() };
                let mut d = vec![0.0; a.r * b.c];
                for i in 0..a.r {
                    for j in 0..b.c {
                        d[i * b.c + j] = (0..a.c).map(|p| a.at(i, p) * b.at(p, j)).sum();
                    }
                }
                M { r: a.r, c: b.c, d }
            }
            Op::Add { a, b } | Op::Mul { a, b } => {
                let (a, b2) = (&v[a], &v[b]);
                let add = matches!(n.op, Op::Add { .. });
                let mut d = vec![0.0; a.d.len()];
                for i in 0..a.r {
                    for j in 0..a.c {
                        let (p, q) = (a.at(i, j), bget(b2, i, j));
                        d[i * a.c + j] = if add { p + q } else { p * q };
                    }
                }
                M { r: a.r, c: a.c, d }
            }
            Op::Relu { x } => v[x].map(|t| t.max(0.0)),
            Op::Sigmoid { x } => v[x].map(|t| 1.0 / (1.0 + (-t).exp())),
            Op::Tanh { x } => v[x].map(f64::tanh),
            Op::Softmax { x } | Op::LogSoftmax { x } => {
                let m = &v[x];
                let mut d = vec![0.0; m.d.len()];
                for i in 0..m.r {
                    let row = &m.d[i * m.c..(i + 1) * m.c];
                    let l = lse(row);
                    for j in 0..m.c {
                        let ls = row[j] - l;
                        d[i * m.c + j] = if matches!(n.op, Op::Softmax { .. }) {
                            ls.exp()
                        } else {
                            ls
                        };
                    }
                }
                M { r: m.r, c: m.c, d }
            }
            Op::Transpose { x } => v[x].t(),
            Op::SumAll { x } => M::scalar(v[x].d.iter().sum()),
            Op::MeanAll { x } => M::scalar(v[x].d.iter().sum::<f64>() / v[x].d.len() as f64),
            Op::MseLoss { pred, target } => {
                let (p, t) = (&v[pred], &v[target]);
                M::scalar(
                    p.d.iter()
                        .zip(&t.d)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / p.d.len() as f64,
                )
            }
            Op::BceWithLogitsLoss { logits, target } => {
                let (z, t) = (&v[logits], &v[target]);
                let s: f64 =
                    z.d.iter()
                        .zip(&t.d)
                        .map(|(z, y)| {
                            let p = 1.0 / (1.0 + (-z).exp());
                            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                        })
                        .sum();
                M::scalar(s / z.d.len() as f64)
            }
            Op::CrossEntropyLoss { logits, target } => {
                let (z, t) = (&v[logits], &v[target]);
                let mut s = 0.0;
                for i in 0..z.r {
                    let row = &z.d[i * z.c..(i + 1) * z.c];
                    s += lse(row) - row[t.d[i] as usize];
                }
                M::scalar(s / z.r as f64)
            }
        };
        v.push(out);
    }
    v
}

/// Central finite-difference gradient of the loss w.r.t. every parameter (f64).
pub fn fd_grad(def: &GraphDef, params: &[f64], x: &M, y: &M, h: f64) -> Vec<f64> {
    let loss = def.loss.expect("loss");
    let mut p = params.to_vec();
    (0..params.len())
        .map(|i| {
            let orig = p[i];
            p[i] = orig + h;
            let lp = eval(def, &p, x, Some(y))[loss].d[0];
            p[i] = orig - h;
            let lm = eval(def, &p, x, Some(y))[loss].d[0];
            p[i] = orig;
            (lp - lm) / (2.0 * h)
        })
        .collect()
}
