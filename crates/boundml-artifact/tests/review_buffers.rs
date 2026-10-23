//! Buffer boundaries of the public prediction API (review finding: fill_named
//! panicked on an undersized destination).

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

#[test]
fn fill_named_validates_destination_length() {
    let d = tempfile::tempdir().unwrap();
    let pkg = Arc::new(Package::save(&d.path().join("p"), spec([1.0, 1.0, 0.0], 1)).unwrap());
    let mut p = pkg.predictor(4).unwrap();
    for len in [0usize, 1, 3, 10] {
        let mut dst = vec![Some(9.0); len];
        let e = p
            .fill_named([("a", Some(1.0))], &mut dst, 7)
            .expect_err("wrong length accepted");
        assert!(e.to_string().contains("destination holds"), "{e}");
        assert!(
            dst.iter().all(|v| *v == Some(9.0)),
            "destination modified on error"
        );
    }
    let mut dst = vec![Some(9.0); 2];
    p.fill_named([("b", Some(2.0))], &mut dst, 0).unwrap();
    assert_eq!(dst, vec![None, Some(2.0)]);
    // A rejected row leaves the previous contents intact.
    assert!(
        p.fill_named([("a", Some(1.0)), ("a", Some(2.0))], &mut dst, 0)
            .is_err()
    );
    assert_eq!(dst, vec![None, Some(2.0)]);
}

#[test]
fn predict_raw_rejects_wrong_input_and_output_lengths() {
    let d = tempfile::tempdir().unwrap();
    let pkg = Arc::new(Package::save(&d.path().join("p"), spec([1.0, 1.0, 0.0], 1)).unwrap());
    let mut p = pkg.predictor(4).unwrap();
    let mut out = [0.0f32; 2];
    assert!(
        p.predict_raw(&[Some(1.0); 3], 2, &mut out).is_err(),
        "short input"
    );
    assert!(
        p.predict_raw(&[Some(1.0); 5], 2, &mut out).is_err(),
        "oversized input"
    );
    assert!(
        p.predict_raw(&[Some(1.0); 4], 2, &mut out[..1]).is_err(),
        "short output"
    );
    assert!(
        p.predict_raw(&[Some(1.0); 4], 2, &mut [0.0f32; 3]).is_err(),
        "oversized output"
    );
    p.predict_raw(&[Some(1.0), Some(2.0), Some(0.5), Some(0.5)], 2, &mut out)
        .unwrap();
    assert_eq!(out, [3.0, 1.0]);
}
