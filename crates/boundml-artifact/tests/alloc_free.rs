//! `Predictor::predict_raw` (fitted preprocessing + planned execution) performs no
//! heap allocation after construction.

use boundml_artifact::{Bounds, Package, PackageSpec, Provenance};
use boundml_core::{Activation, DenseSpec, PlannerKind, Rng};
use boundml_data::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

struct Counting;
thread_local! { static COUNT: Cell<usize> = const { Cell::new(0) }; }

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        // SAFETY: forwards to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from `System` with layout `l`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        // SAFETY: forwards to the system allocator.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn predict_raw_does_not_allocate() {
    let schema = Schema {
        features: vec!["a".into(), "b".into(), "c".into()],
        label: "y".into(),
        label_kind: LabelKind::Classes {
            classes: vec!["x".into(), "y".into(), "z".into()],
        },
        missing: MissingPolicy::ImputeMean,
        extra_columns: ExtraColumns::Ignore,
    };
    let spec = DenseSpec {
        inputs: 3,
        hidden: vec![8],
        activation: Activation::Relu,
        task: schema.task(),
    };
    let g = spec.graph().unwrap();
    let pre = Preprocessor {
        normalize: Normalize::Standardize,
        missing: MissingPolicy::ImputeMean,
        features: ["a", "b", "c"]
            .iter()
            .map(|n| FeatureStats {
                name: n.to_string(),
                mean: 1.0,
                std: 2.0,
                count: 5,
                missing: 0,
                constant: false,
            })
            .collect(),
        rows_fitted: 5,
    };
    let d = tempfile::tempdir().unwrap();
    let pkg = Arc::new(
        Package::save(
            &d.path().join("p"),
            PackageSpec {
                name: "alloc".into(),
                schema,
                preprocessing: pre,
                graph: g.def().clone(),
                weights: spec.init_params(&g, &mut Rng::seed(3)),
                bounds: Bounds {
                    max_batch: 16,
                    tensor_budget_bytes: Some(1 << 20),
                    sessions: 1,
                    planner: PlannerKind::Reuse,
                },
                provenance: Provenance::default(),
                test_inputs: vec![],
                test_expected: None,
                test_origin: String::new(),
                test_atol: 0.0,
                test_rtol: 0.0,
            },
        )
        .unwrap(),
    );
    let mut p = pkg.predictor(16).unwrap();
    let raw: Vec<Option<f64>> = (0..48)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else {
                Some(i as f64 * 0.1)
            }
        })
        .collect();
    let mut out = vec![0.0f32; 48];
    p.predict_raw(&raw, 16, &mut out).unwrap();
    let before = COUNT.with(|c| c.get());
    for b in [1usize, 5, 16] {
        p.predict_raw(&raw[..b * 3], b, &mut out[..b * 3]).unwrap();
    }
    assert_eq!(COUNT.with(|c| c.get()) - before, 0);
}
