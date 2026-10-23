//! Regressions from the external source review of e567a00 (reconstructed from the
//! review's written reproductions).

use boundml_artifact::*;
use boundml_core::{Activation, DenseSpec, PlannerKind};
use boundml_data::*;
use std::sync::Arc;

fn schema() -> Schema {
    Schema {
        features: vec!["a".into(), "b".into()],
        label: "y".into(),
        label_kind: LabelKind::Regression,
        missing: MissingPolicy::Reject,
        extra_columns: ExtraColumns::Ignore,
    }
}

fn identity_pre() -> Preprocessor {
    Preprocessor {
        normalize: Normalize::None,
        missing: MissingPolicy::Reject,
        features: ["a", "b"]
            .iter()
            .map(|n| FeatureStats {
                name: n.to_string(),
                mean: 0.0,
                std: 1.0,
                count: 1,
                missing: 0,
                constant: false,
            })
            .collect(),
        rows_fitted: 1,
    }
}

/// Linear regression y = w0*a + w1*b + bias.
fn spec(weights: [f32; 3], sessions: usize) -> PackageSpec {
    let dense = DenseSpec {
        inputs: 2,
        hidden: vec![],
        activation: Activation::Relu,
        task: boundml_core::Task::Regression,
    };
    PackageSpec {
        name: "review".into(),
        schema: schema(),
        preprocessing: identity_pre(),
        graph: dense.graph().unwrap().def().clone(),
        weights: weights.to_vec(),
        bounds: Bounds {
            max_batch: 8,
            tensor_budget_bytes: None,
            sessions,
            planner: PlannerKind::Reuse,
        },
        provenance: Provenance::default(),
        test_inputs: vec![],
        test_expected: None,
        test_origin: "synthetic".into(),
        test_atol: 1e-6,
        test_rtol: 1e-5,
    }
}

const NAN_W: [f32; 3] = [3e38, -3e38, 0.0];

#[test]
fn package_with_nan_outputs_cannot_pass_its_self_test() {
    let d = tempfile::tempdir().unwrap();
    // Expected [0] for an input whose real output is NaN: must not save.
    let mut s = spec(NAN_W, 1);
    s.test_inputs = vec![vec![Some(2.0), Some(2.0)]];
    s.test_expected = Some(vec![vec![0.0]]);
    assert!(Package::save(&d.path().join("p1"), s).is_err());
    assert!(!d.path().join("p1").exists(), "nothing written");
    // Expected computed by the runtime itself (NaN): must not save either.
    let mut s = spec(NAN_W, 1);
    s.test_inputs = vec![vec![Some(2.0), Some(2.0)]];
    assert!(Package::save(&d.path().join("p2"), s).is_err());
    // Non-finite expected values are rejected even for a finite model.
    let mut s = spec([1.0, 1.0, 0.0], 1);
    s.test_inputs = vec![vec![Some(2.0), Some(2.0)]];
    s.test_expected = Some(vec![vec![f32::NAN]]);
    assert!(Package::save(&d.path().join("p3"), s).is_err());
}

#[test]
fn compare_with_nan_candidate_is_never_compatible() {
    let d = tempfile::tempdir().unwrap();
    let data = d.path().join("clients.csv");
    // Only the NaN-producing row: a finite drift elsewhere must not be what trips the gate.
    std::fs::write(&data, "a,b,y\n2,2,1\n").unwrap();
    let cur = Arc::new(Package::save(&d.path().join("cur"), spec([1.0, 1.0, 0.0], 1)).unwrap());
    let cand = Arc::new(Package::save(&d.path().join("cand"), spec(NAN_W, 1)).unwrap());
    let opts = CompareOptions {
        data: Some(data),
        max_abs_diff: Some(0.0),
        max_rows: None,
    };
    let r = compare(&cur, &cand, &opts).unwrap();
    assert_eq!(r.verdict, Verdict::Breaking, "{r:?}");
}

#[test]
fn compare_rejects_invalid_thresholds() {
    let d = tempfile::tempdir().unwrap();
    let a = Arc::new(Package::save(&d.path().join("a"), spec([1.0, 1.0, 0.0], 1)).unwrap());
    for t in [f32::NAN, f32::INFINITY, -1.0] {
        let opts = CompareOptions {
            data: None,
            max_abs_diff: Some(t),
            max_rows: None,
        };
        assert!(compare(&a, &a, &opts).is_err(), "threshold {t} accepted");
    }
}
