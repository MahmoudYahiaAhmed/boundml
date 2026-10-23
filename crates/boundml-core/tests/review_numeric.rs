//! Regressions from the external source review of e567a00 (reconstructed from the
//! review's written reproductions): finite weights and inputs must never yield a
//! successful non-finite prediction.

use boundml_core::{Activation, DenseSpec, InferenceSession, Model, Task};
use std::sync::Arc;

fn linear(weights: Vec<f32>) -> Arc<Model> {
    let spec = DenseSpec {
        inputs: 2,
        hidden: vec![],
        activation: Activation::Relu,
        task: Task::Regression,
    };
    Arc::new(Model::new(spec.graph().unwrap(), weights).unwrap())
}

#[test]
fn finite_overflow_to_nan_is_an_error_not_a_prediction() {
    // 2*3e38 overflows to +inf, 2*(-3e38) to -inf, and their sum is NaN.
    let mut s = InferenceSession::new(linear(vec![3e38, -3e38, 0.0]), 4).unwrap();
    let mut out = [0.0f32; 1];
    let r = s.predict_into(&[2.0, 2.0], 1, &mut out);
    assert!(r.is_err(), "non-finite output accepted: {out:?}");
    assert!(r.unwrap_err().to_string().contains("BML-C007"));
}

#[test]
fn positive_and_negative_infinity_are_errors() {
    for (w, label) in [
        (vec![3e38, 3e38, 0.0], "+inf"),
        (vec![-3e38, -3e38, 0.0], "-inf"),
    ] {
        let mut s = InferenceSession::new(linear(w), 1).unwrap();
        let mut out = [0.0f32; 1];
        assert!(
            s.predict_into(&[2.0, 2.0], 1, &mut out).is_err(),
            "{label} accepted"
        );
    }
}

#[test]
fn one_bad_row_fails_the_batch_and_finite_predictions_still_work() {
    let mut s = InferenceSession::new(linear(vec![3e38, -3e38, 0.0]), 4).unwrap();
    let mut out = [0.0f32; 2];
    // Row 0 is fine (1e-38 scale), row 1 overflows.
    assert!(
        s.predict_into(&[1e-30, 0.0, 2.0, 2.0], 2, &mut out)
            .is_err()
    );
    let mut ok = InferenceSession::new(linear(vec![1.5, -0.5, 0.25]), 4).unwrap();
    let mut out = [0.0f32; 2];
    ok.predict_into(&[2.0, 2.0, 0.0, 1.0], 2, &mut out).unwrap();
    assert_eq!(out, [2.25, -0.25]);
}
