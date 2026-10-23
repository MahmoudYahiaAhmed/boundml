//! Regressions from the external source review of e567a00 (reconstructed from the
//! review's written reproductions).

use boundml_artifact::fsutil::sha256_hex;
use boundml_artifact::*;
use boundml_core::{Activation, DenseSpec, PlannerKind};
use boundml_data::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;
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

fn edit_manifest(dir: &Path, f: impl FnOnce(&mut serde_json::Value)) {
    let p = dir.join("manifest.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    f(&mut v);
    let bytes = serde_json::to_vec_pretty(&v).unwrap();
    std::fs::write(&p, &bytes).unwrap();
    std::fs::write(
        dir.join("MANIFEST.sha256"),
        format!("{}\n", sha256_hex(&bytes)),
    )
    .unwrap();
}

#[test]
fn invalid_metadata_is_rejected_before_weight_io() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("p");
    Package::save(&dir, spec([1.0, 1.0, 0.0], 1)).unwrap();
    edit_manifest(&dir, |v| v["bounds"]["sessions"] = 0.into());
    std::fs::remove_file(dir.join("weights.safetensors")).unwrap();
    let e = Package::load(&dir).err().unwrap().to_string();
    assert!(
        e.contains("BML-A006") && e.contains("sessions"),
        "weight I/O reached first: {e}"
    );
}

// --- measured allocation: invalid metadata must not materialize the payload -------

struct Counting;
thread_local! { static BYTES: Cell<usize> = const { Cell::new(0) }; }

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + l.size()));
        // SAFETY: forwards to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from `System` with layout `l`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + n));
        // SAFETY: forwards to the system allocator.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const PAYLOAD: usize = 4 << 20;

/// Replace the weights with a 4 MiB file whose declared size and hash match, so
/// only metadata validation can stop the load before the payload is read.
fn big_payload_package(dir: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    Package::save(dir, spec([1.0, 1.0, 0.0], 1)).unwrap();
    let payload = vec![0u8; PAYLOAD];
    std::fs::write(dir.join("weights.safetensors"), &payload).unwrap();
    let digest = sha256_hex(&payload);
    edit_manifest(dir, |v| {
        v["files"]["weights.safetensors"]["bytes"] = PAYLOAD.into();
        v["files"]["weights.safetensors"]["sha256"] = digest.into();
        edit(v);
    });
}

fn bytes_allocated_by<F: FnOnce()>(f: F) -> usize {
    let before = BYTES.with(|b| b.get());
    f();
    BYTES.with(|b| b.get()) - before
}

#[test]
fn invalid_bounds_are_rejected_without_reading_a_large_payload() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("p");
    big_payload_package(&dir, |v| v["bounds"]["sessions"] = 0.into());
    let mut err = String::new();
    let n = bytes_allocated_by(|| err = Package::load(&dir).err().unwrap().to_string());
    assert!(err.contains("sessions"), "{err}");
    assert!(
        n < PAYLOAD / 4,
        "load allocated {n} bytes before rejecting metadata"
    );
}

#[test]
fn declared_size_inconsistent_with_the_graph_is_rejected_before_reading() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("p");
    // Valid bounds, but 4 MiB cannot be the payload of a 3-parameter graph.
    big_payload_package(&dir, |_| {});
    let mut err = String::new();
    let n = bytes_allocated_by(|| err = Package::load(&dir).err().unwrap().to_string());
    assert!(
        err.contains("BML-A006") && err.contains("declared weights size"),
        "{err}"
    );
    assert!(
        n < PAYLOAD / 4,
        "load allocated {n} bytes before rejecting the declared size"
    );
}

#[test]
fn valid_packages_still_load_after_preflight() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("p");
    Package::save(&dir, spec([1.0, 2.0, 0.5], 1)).unwrap();
    let pkg = Arc::new(Package::load(&dir).unwrap());
    let mut p = pkg.predictor(1).unwrap();
    assert_eq!(p.predict_one(&[Some(1.0), Some(1.0)]).unwrap(), vec![3.5]);
}
