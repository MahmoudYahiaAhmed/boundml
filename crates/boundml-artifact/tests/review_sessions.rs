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

#[test]
fn predictors_beyond_declared_sessions_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let pkg = Arc::new(Package::save(&d.path().join("p"), spec([1.0, 1.0, 0.0], 2)).unwrap());
    let p1 = pkg.predictor(4).unwrap();
    let p2 = Predictor::new(Arc::clone(&pkg), 4).unwrap();
    let third = pkg.predictor(4);
    assert!(third.is_err(), "third session allowed with sessions = 2");
    drop(p1);
    let p3 = pkg.predictor(4).expect("a released session can be reused");
    drop((p2, p3));
}

#[test]
fn concurrent_reservations_never_exceed_the_allowance() {
    let d = tempfile::tempdir().unwrap();
    let pkg = Arc::new(Package::save(&d.path().join("p"), spec([1.0, 1.0, 0.0], 3)).unwrap());
    let barrier = Arc::new(std::sync::Barrier::new(16));
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let (pkg, barrier) = (Arc::clone(&pkg), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                let p = pkg.predictor(4);
                let ok = p.is_ok();
                // Hold it until every thread has tried.
                std::thread::sleep(std::time::Duration::from_millis(50));
                drop(p);
                ok
            })
        })
        .collect();
    let granted = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .filter(|ok| *ok)
        .count();
    assert!(
        granted <= 3,
        "{granted} sessions granted with an allowance of 3"
    );
    assert!(granted >= 1);
    assert_eq!(pkg.sessions_in_use(), 0, "all reservations released");
}

#[test]
fn failed_construction_and_verification_do_not_leak_sessions() {
    let d = tempfile::tempdir().unwrap();
    let mut s = spec([1.0, 1.0, 0.0], 1);
    s.test_inputs = vec![vec![Some(1.0), Some(2.0)]];
    let pkg = Arc::new(Package::save(&d.path().join("p"), s).unwrap());
    assert!(pkg.predictor(0).is_err());
    assert!(pkg.predictor(9).is_err(), "above max_batch");
    assert_eq!(pkg.sessions_in_use(), 0);
    // verify borrows one session from the same allowance and returns it.
    assert!(pkg.verify().unwrap().passed);
    assert_eq!(pkg.sessions_in_use(), 0);
    // With the only session held, verification cannot run: an error, not a "failed" self-test.
    let held = pkg.predictor(1).unwrap();
    let e = pkg.verify().expect_err("no free session");
    assert!(e.to_string().contains("BML-A010"), "{e}");
    // Clones of the package share the allowance (no reset by cloning).
    let clone = Arc::new((*pkg).clone());
    assert!(clone.predictor(1).is_err());
    drop(held);
    assert!(clone.predictor(1).is_ok());
    // Loading the directory again is a separate, independent instance.
    let other = Arc::new(Package::load(&d.path().join("p")).unwrap());
    let _a = pkg.predictor(1).unwrap();
    assert!(other.predictor(1).is_ok());
}
