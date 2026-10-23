//! Streaming evaluation metrics (bounded memory) and constant-predictor baselines.

use boundml_core::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const P_EPS: f64 = 1e-7;

/// Label statistics from the training file, used for constant baselines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelStats {
    pub rows: u64,
    pub mean: f64,
    pub class_counts: Vec<u64>,
}

impl LabelStats {
    pub fn new(task: &Task) -> Self {
        LabelStats {
            rows: 0,
            mean: 0.0,
            class_counts: vec![0; class_count(task)],
        }
    }

    pub fn add(&mut self, y: f32) {
        self.rows += 1;
        self.mean += (y as f64 - self.mean) / self.rows as f64;
        if let Some(c) = self.class_counts.get_mut(y as usize) {
            *c += 1;
        }
    }

    /// Output of the constant baseline: training mean (regression) or class priors.
    pub fn constant_output(&self, task: &Task) -> Vec<f32> {
        let n = self.rows.max(1) as f64;
        match task {
            Task::Regression => vec![self.mean as f32],
            Task::Binary => vec![(self.class_counts[1] as f64 / n) as f32],
            Task::Multiclass { .. } => self
                .class_counts
                .iter()
                .map(|&c| (c as f64 / n) as f32)
                .collect(),
        }
    }
}

fn class_count(task: &Task) -> usize {
    match task {
        Task::Regression => 0,
        Task::Binary => 2,
        Task::Multiclass { classes } => *classes,
    }
}

#[derive(Debug, Clone)]
pub struct MetricAcc {
    task: Task,
    n: u64,
    sum_sq: f64,
    sum_abs: f64,
    sum_y: f64,
    sum_y2: f64,
    log_loss: f64,
    /// `confusion[truth * k + predicted]`.
    confusion: Vec<u64>,
}

impl MetricAcc {
    pub fn new(task: Task) -> Self {
        let k = class_count(&task);
        MetricAcc {
            task,
            n: 0,
            sum_sq: 0.0,
            sum_abs: 0.0,
            sum_y: 0.0,
            sum_y2: 0.0,
            log_loss: 0.0,
            confusion: vec![0; k * k],
        }
    }

    pub fn rows(&self) -> u64 {
        self.n
    }

    /// Add one row: `out` is the model output row, `y` the label (value or class index).
    pub fn add(&mut self, out: &[f32], y: f32) {
        self.n += 1;
        match self.task {
            Task::Regression => {
                let d = out[0] as f64 - y as f64;
                self.sum_sq += d * d;
                self.sum_abs += d.abs();
                self.sum_y += y as f64;
                self.sum_y2 += (y as f64) * (y as f64);
            }
            Task::Binary => {
                let p = (out[0] as f64).clamp(P_EPS, 1.0 - P_EPS);
                let t = y as usize;
                self.log_loss -= if t == 1 { p.ln() } else { (1.0 - p).ln() };
                let pred = usize::from(out[0] >= 0.5);
                self.confusion[t * 2 + pred] += 1;
            }
            Task::Multiclass { classes } => {
                let t = y as usize;
                self.log_loss -= (out[t] as f64).clamp(P_EPS, 1.0).ln();
                let pred = boundml_artifact::compare::decision(&self.task, out);
                self.confusion[t * classes + pred] += 1;
            }
        }
    }

    pub fn finish(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        let n = self.n.max(1) as f64;
        m.insert("rows".into(), self.n as f64);
        match self.task {
            Task::Regression => {
                let mse = self.sum_sq / n;
                m.insert("mse".into(), mse);
                m.insert("rmse".into(), mse.sqrt());
                m.insert("mae".into(), self.sum_abs / n);
                let sst = self.sum_y2 - self.sum_y * self.sum_y / n;
                if sst > 0.0 {
                    m.insert("r2".into(), 1.0 - self.sum_sq / sst);
                }
            }
            Task::Binary | Task::Multiclass { .. } => {
                let k = class_count(&self.task);
                let correct: u64 = (0..k).map(|i| self.confusion[i * k + i]).sum();
                m.insert("accuracy".into(), correct as f64 / n);
                m.insert("log_loss".into(), self.log_loss / n);
                let f1 = |c: usize| {
                    let tp = self.confusion[c * k + c] as f64;
                    let pred: u64 = (0..k).map(|t| self.confusion[t * k + c]).sum();
                    let truth: u64 = (0..k).map(|p| self.confusion[c * k + p]).sum();
                    let prec = if pred > 0 { tp / pred as f64 } else { 0.0 };
                    let rec = if truth > 0 { tp / truth as f64 } else { 0.0 };
                    let f = if prec + rec > 0.0 {
                        2.0 * prec * rec / (prec + rec)
                    } else {
                        0.0
                    };
                    (prec, rec, f)
                };
                if k == 2 {
                    let (p, r, f) = f1(1);
                    m.insert("precision".into(), p);
                    m.insert("recall".into(), r);
                    m.insert("f1".into(), f);
                } else {
                    let macro_f1 = (0..k).map(|c| f1(c).2).sum::<f64>() / k as f64;
                    m.insert("macro_f1".into(), macro_f1);
                }
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_metrics() {
        let mut a = MetricAcc::new(Task::Regression);
        for (p, y) in [(1.0, 1.0), (2.0, 3.0), (3.0, 2.0)] {
            a.add(&[p], y);
        }
        let m = a.finish();
        assert!((m["mse"] - 2.0 / 3.0).abs() < 1e-12);
        assert!((m["mae"] - 2.0 / 3.0).abs() < 1e-12);
        assert!((m["r2"] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn classification_metrics_and_baseline() {
        let mut a = MetricAcc::new(Task::Binary);
        for (p, y) in [(0.9, 1.0), (0.2, 0.0), (0.6, 0.0), (0.4, 1.0)] {
            a.add(&[p], y);
        }
        let m = a.finish();
        assert_eq!(m["accuracy"], 0.5);
        assert_eq!(m["precision"], 0.5);
        let mut s = LabelStats::new(&Task::Multiclass { classes: 3 });
        for y in [0.0, 2.0, 2.0, 2.0] {
            s.add(y);
        }
        assert_eq!(
            s.constant_output(&Task::Multiclass { classes: 3 }),
            vec![0.25, 0.0, 0.75]
        );
    }
}
