//! Scalar-loop CPU kernels over row-major f32 slices.
//!
//! Every kernel is scratch-free and allocation-free: it reads its operand slices and
//! writes (or accumulates into) exactly one output slice. Callers pass slices whose
//! lengths match the resolved shapes; kernels `debug_assert` this.
//! Backward kernels *accumulate* (`+=`) so fan-out gradients sum correctly.

use crate::error::{CoreError, Result};

/// Dimensions of a broadcastable right-hand operand `b` relative to `a` = `[rows, cols]`.
#[derive(Debug, Clone, Copy)]
pub struct Bcast {
    pub rows: usize,
    pub cols: usize,
    pub b_rows: usize,
    pub b_cols: usize,
}

impl Bcast {
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        let bi = if self.b_rows == 1 { 0 } else { i };
        let bj = if self.b_cols == 1 { 0 } else { j };
        bi * self.b_cols + bj
    }
    fn same(&self) -> bool {
        self.rows == self.b_rows && self.cols == self.b_cols
    }
}

/// `c (+)= op(a) @ op(b)` with logical `op(a): [m, k]`, `op(b): [k, n]`.
///
/// `ta`: `a` is stored as `[k, m]`. `tb`: `b` is stored as `[n, k]`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    ta: bool,
    b: &[f32],
    tb: bool,
    c: &mut [f32],
    accumulate: bool,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if !accumulate {
        c.fill(0.0);
    }
    let a_at = |i: usize, p: usize| if ta { a[p * m + i] } else { a[i * k + p] };
    if !tb {
        for i in 0..m {
            let crow = &mut c[i * n..(i + 1) * n];
            for p in 0..k {
                let aip = a_at(i, p);
                let brow = &b[p * n..(p + 1) * n];
                for (cv, bv) in crow.iter_mut().zip(brow) {
                    *cv += aip * bv;
                }
            }
        }
    } else if !ta {
        for i in 0..m {
            let arow = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let brow = &b[j * k..(j + 1) * k];
                let s: f32 = arow.iter().zip(brow).map(|(x, y)| x * y).sum();
                c[i * n + j] += s;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let brow = &b[j * k..(j + 1) * k];
                let mut s = 0.0f32;
                for (p, bv) in brow.iter().enumerate() {
                    s += a[p * m + i] * bv;
                }
                c[i * n + j] += s;
            }
        }
    }
}

pub fn add_fwd(a: &[f32], b: &[f32], bc: Bcast, out: &mut [f32]) {
    if bc.same() {
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
        return;
    }
    for i in 0..bc.rows {
        for j in 0..bc.cols {
            out[i * bc.cols + j] = a[i * bc.cols + j] + b[bc.idx(i, j)];
        }
    }
}

pub fn mul_fwd(a: &[f32], b: &[f32], bc: Bcast, out: &mut [f32]) {
    for i in 0..bc.rows {
        for j in 0..bc.cols {
            out[i * bc.cols + j] = a[i * bc.cols + j] * b[bc.idx(i, j)];
        }
    }
}

/// `ga += g` (gradient of the full-shape operand of an add).
pub fn acc(ga: &mut [f32], g: &[f32]) {
    for (x, y) in ga.iter_mut().zip(g) {
        *x += y;
    }
}

/// `gb[bidx] += g` — reduce a full-shape gradient into a broadcast operand.
pub fn acc_reduce(gb: &mut [f32], g: &[f32], bc: Bcast) {
    for i in 0..bc.rows {
        for j in 0..bc.cols {
            gb[bc.idx(i, j)] += g[i * bc.cols + j];
        }
    }
}

/// Gradient for `a` in `a * b`: `ga += g * b[bidx]`.
pub fn mul_bwd_a(ga: &mut [f32], g: &[f32], b: &[f32], bc: Bcast) {
    for i in 0..bc.rows {
        for j in 0..bc.cols {
            ga[i * bc.cols + j] += g[i * bc.cols + j] * b[bc.idx(i, j)];
        }
    }
}

/// Gradient for `b` in `a * b`: `gb[bidx] += g * a`.
pub fn mul_bwd_b(gb: &mut [f32], g: &[f32], a: &[f32], bc: Bcast) {
    for i in 0..bc.rows {
        for j in 0..bc.cols {
            gb[bc.idx(i, j)] += g[i * bc.cols + j] * a[i * bc.cols + j];
        }
    }
}

pub fn relu_fwd(x: &[f32], y: &mut [f32]) {
    for (o, v) in y.iter_mut().zip(x) {
        *o = if *v > 0.0 { *v } else { 0.0 };
    }
}

/// Uses the forward output: d/dx relu = 1 if y > 0 else 0 (0 at x = 0 by convention).
pub fn relu_bwd(gx: &mut [f32], g: &[f32], y: &[f32]) {
    for ((d, gv), yv) in gx.iter_mut().zip(g).zip(y) {
        if *yv > 0.0 {
            *d += gv;
        }
    }
}

#[inline]
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

pub fn sigmoid_fwd(x: &[f32], y: &mut [f32]) {
    for (o, v) in y.iter_mut().zip(x) {
        *o = sigmoid(*v);
    }
}

pub fn sigmoid_bwd(gx: &mut [f32], g: &[f32], y: &[f32]) {
    for ((d, gv), yv) in gx.iter_mut().zip(g).zip(y) {
        *d += gv * yv * (1.0 - yv);
    }
}

pub fn tanh_fwd(x: &[f32], y: &mut [f32]) {
    for (o, v) in y.iter_mut().zip(x) {
        *o = v.tanh();
    }
}

pub fn tanh_bwd(gx: &mut [f32], g: &[f32], y: &[f32]) {
    for ((d, gv), yv) in gx.iter_mut().zip(g).zip(y) {
        *d += gv * (1.0 - yv * yv);
    }
}

/// Returns `(max, log(sum(exp(row - max))))` for a row.
#[inline]
fn row_lse(row: &[f32]) -> (f32, f32) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = row.iter().map(|v| (v - max).exp()).sum();
    (max, s.ln())
}

pub fn softmax_fwd(x: &[f32], cols: usize, y: &mut [f32]) {
    for (xr, yr) in x.chunks_exact(cols).zip(y.chunks_exact_mut(cols)) {
        let (max, lse) = row_lse(xr);
        for (o, v) in yr.iter_mut().zip(xr) {
            *o = (v - max - lse).exp();
        }
    }
}

/// `gx_i += y_i * (g_i - sum_j g_j y_j)`.
pub fn softmax_bwd(gx: &mut [f32], g: &[f32], y: &[f32], cols: usize) {
    for ((dr, gr), yr) in gx
        .chunks_exact_mut(cols)
        .zip(g.chunks_exact(cols))
        .zip(y.chunks_exact(cols))
    {
        let dot: f32 = gr.iter().zip(yr).map(|(a, b)| a * b).sum();
        for ((d, gv), yv) in dr.iter_mut().zip(gr).zip(yr) {
            *d += yv * (gv - dot);
        }
    }
}

pub fn log_softmax_fwd(x: &[f32], cols: usize, y: &mut [f32]) {
    for (xr, yr) in x.chunks_exact(cols).zip(y.chunks_exact_mut(cols)) {
        let (max, lse) = row_lse(xr);
        for (o, v) in yr.iter_mut().zip(xr) {
            *o = v - max - lse;
        }
    }
}

/// `gx_i += g_i - exp(y_i) * sum_j g_j`.
pub fn log_softmax_bwd(gx: &mut [f32], g: &[f32], y: &[f32], cols: usize) {
    for ((dr, gr), yr) in gx
        .chunks_exact_mut(cols)
        .zip(g.chunks_exact(cols))
        .zip(y.chunks_exact(cols))
    {
        let s: f32 = gr.iter().sum();
        for ((d, gv), yv) in dr.iter_mut().zip(gr).zip(yr) {
            *d += gv - yv.exp() * s;
        }
    }
}

/// `y = x^T` where `x: [rows, cols]`; `accumulate` adds instead of overwriting.
pub fn transpose(x: &[f32], rows: usize, cols: usize, y: &mut [f32], accumulate: bool) {
    for i in 0..rows {
        for j in 0..cols {
            let v = x[i * cols + j];
            if accumulate {
                y[j * rows + i] += v;
            } else {
                y[j * rows + i] = v;
            }
        }
    }
}

pub fn sum_all(x: &[f32]) -> f32 {
    x.iter().map(|v| f64::from(*v)).sum::<f64>() as f32
}

/// `gx += g * scale` for a scalar upstream gradient.
pub fn acc_scalar(gx: &mut [f32], g: f32) {
    for d in gx.iter_mut() {
        *d += g;
    }
}

pub fn mse_fwd(p: &[f32], t: &[f32]) -> f32 {
    let n = p.len() as f64;
    let s: f64 = p
        .iter()
        .zip(t)
        .map(|(a, b)| {
            let d = f64::from(*a) - f64::from(*b);
            d * d
        })
        .sum();
    (s / n) as f32
}

pub fn mse_bwd(gp: &mut [f32], g: f32, p: &[f32], t: &[f32]) {
    let k = 2.0 * g / p.len() as f32;
    for ((d, a), b) in gp.iter_mut().zip(p).zip(t) {
        *d += k * (a - b);
    }
}

/// Mean of `max(z, 0) - z*y + log(1 + exp(-|z|))` — stable for extreme logits.
pub fn bce_logits_fwd(z: &[f32], y: &[f32]) -> f32 {
    let n = z.len() as f64;
    let s: f64 = z
        .iter()
        .zip(y)
        .map(|(zv, yv)| {
            let (zv, yv) = (f64::from(*zv), f64::from(*yv));
            zv.max(0.0) - zv * yv + (-zv.abs()).exp().ln_1p()
        })
        .sum();
    (s / n) as f32
}

pub fn bce_logits_bwd(gz: &mut [f32], g: f32, z: &[f32], y: &[f32]) {
    let k = g / z.len() as f32;
    for ((d, zv), yv) in gz.iter_mut().zip(z).zip(y) {
        *d += k * (sigmoid(*zv) - yv);
    }
}

fn class_index(v: f32, classes: usize, row: usize) -> Result<usize> {
    if v >= 0.0 && v.fract() == 0.0 && (v as usize) < classes {
        Ok(v as usize)
    } else {
        Err(CoreError::ClassIndex {
            value: v,
            classes,
            row,
        })
    }
}

/// Mean over rows of `-log_softmax(z)[target]`.
pub fn cross_entropy_fwd(z: &[f32], t: &[f32], classes: usize) -> Result<f32> {
    let mut s = 0.0f64;
    for (row, (zr, tv)) in z.chunks_exact(classes).zip(t).enumerate() {
        let c = class_index(*tv, classes, row)?;
        let (max, lse) = row_lse(zr);
        s += f64::from(max + lse - zr[c]);
    }
    Ok((s / t.len() as f64) as f32)
}

pub fn cross_entropy_bwd(
    gz: &mut [f32],
    g: f32,
    z: &[f32],
    t: &[f32],
    classes: usize,
) -> Result<()> {
    let k = g / t.len() as f32;
    for (row, ((dr, zr), tv)) in gz
        .chunks_exact_mut(classes)
        .zip(z.chunks_exact(classes))
        .zip(t)
        .enumerate()
    {
        let c = class_index(*tv, classes, row)?;
        let (max, lse) = row_lse(zr);
        for (j, (d, zv)) in dr.iter_mut().zip(zr).enumerate() {
            let p = (zv - max - lse).exp();
            let onehot = if j == c { 1.0 } else { 0.0 };
            *d += k * (p - onehot);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_all_transpose_variants_agree() {
        // A: [2,3], B: [3,2]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expect = [58.0, 64.0, 139.0, 154.0];
        let mut at = [0.0; 6];
        transpose(&a, 2, 3, &mut at, false);
        let mut bt = [0.0; 6];
        transpose(&b, 3, 2, &mut bt, false);
        for (aa, ta) in [(&a, false), (&at, true)] {
            for (bb, tb) in [(&b, false), (&bt, true)] {
                let mut c = [0.0; 4];
                gemm(2, 2, 3, aa, ta, bb, tb, &mut c, false);
                assert_eq!(c, expect, "ta={ta} tb={tb}");
            }
        }
    }

    #[test]
    fn stable_losses_for_extreme_logits() {
        let z = [1000.0f32, -1000.0, 0.0];
        let y = [1.0f32, 0.0, 1.0];
        let l = bce_logits_fwd(&z, &y);
        assert!(l.is_finite());
        assert!((l - (2f32.ln() / 3.0)).abs() < 1e-6);
        let mut g = [0.0; 3];
        bce_logits_bwd(&mut g, 1.0, &z, &y);
        assert!(g.iter().all(|v| v.is_finite()));

        let zc = [1000.0f32, -1000.0, 0.0];
        let lc = cross_entropy_fwd(&zc, &[0.0], 3).unwrap();
        assert!(lc.abs() < 1e-6);
        let lc2 = cross_entropy_fwd(&zc, &[1.0], 3).unwrap();
        assert!((lc2 - 2000.0).abs() < 1e-2);
        let mut s = [0.0; 3];
        softmax_fwd(&zc, 3, &mut s);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cross_entropy_rejects_bad_class() {
        assert!(cross_entropy_fwd(&[0.0, 0.0], &[2.0], 2).is_err());
        assert!(cross_entropy_fwd(&[0.0, 0.0], &[0.5], 2).is_err());
        assert!(cross_entropy_fwd(&[0.0, 0.0], &[-1.0], 2).is_err());
    }
}
