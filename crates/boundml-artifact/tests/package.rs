use boundml_artifact::fsutil::sha256_hex;
use boundml_artifact::*;
use boundml_core::{Activation, DenseSpec, PlannerKind, Rng};
use boundml_data::*;
use std::path::Path;
use std::sync::Arc;

fn schema(classes: &[&str]) -> Schema {
    Schema {
        features: vec!["a".into(), "b".into(), "c".into()],
        label: "y".into(),
        label_kind: LabelKind::Classes {
            classes: classes.iter().map(|s| s.to_string()).collect(),
        },
        missing: MissingPolicy::ImputeMean,
        extra_columns: ExtraColumns::Ignore,
    }
}

fn pre(s: &Schema) -> Preprocessor {
    Preprocessor {
        normalize: Normalize::Standardize,
        missing: s.missing,
        features: s
            .features
            .iter()
            .enumerate()
            .map(|(i, n)| FeatureStats {
                name: n.clone(),
                mean: i as f64,
                std: 2.0,
                count: 10,
                missing: 0,
                constant: false,
            })
            .collect(),
        rows_fitted: 10,
    }
}

fn spec_for(s: Schema, seed: u64, max_batch: usize) -> PackageSpec {
    let dense = DenseSpec {
        inputs: 3,
        hidden: vec![4],
        activation: Activation::Tanh,
        task: s.task(),
    };
    let graph = dense.graph().unwrap();
    let weights = dense.init_params(&graph, &mut Rng::seed(seed));
    PackageSpec {
        name: "test-model".into(),
        preprocessing: pre(&s),
        schema: s,
        graph: graph.def().clone(),
        weights,
        bounds: Bounds {
            max_batch,
            tensor_budget_bytes: None,
            sessions: 1,
            planner: PlannerKind::Reuse,
        },
        provenance: Provenance {
            source: "test".into(),
            ..Default::default()
        },
        test_inputs: vec![
            vec![Some(1.0), Some(-2.0), None],
            vec![Some(0.5), Some(0.0), Some(3.0)],
        ],
        test_expected: None,
        test_origin: "synthetic".into(),
        test_atol: 1e-6,
        test_rtol: 1e-5,
    }
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
    }
}

/// Rewrite manifest.json via `f` and fix up its hash so only `f`'s change is tested.
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
fn save_load_verify_and_predict() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("m.boundml");
    let saved = Package::save(&dir, spec_for(schema(&["x", "y", "z"]), 1, 8)).unwrap();
    let pkg = Arc::new(Package::load(&dir).unwrap());
    assert_eq!(pkg.manifest(), saved.manifest());
    assert_eq!(pkg.model().weights(), saved.model().weights());
    let r = pkg.verify().unwrap();
    assert!(r.passed && r.vectors == 2);

    let mut p = pkg.predictor(4).unwrap();
    let out = p.predict_one(&[Some(1.0), Some(-2.0), None]).unwrap();
    assert!(
        (out.iter().sum::<f32>() - 1.0).abs() < 1e-5,
        "softmax output sums to 1"
    );
    assert_eq!(
        out,
        pkg.manifest().test_vectors.as_ref().unwrap().expected[0]
    );

    let mut row = vec![None; 3];
    p.fill_named(
        [
            ("c", None),
            ("a", Some(1.0)),
            ("b", Some(-2.0)),
            ("zzz", Some(9.0)),
        ],
        &mut row,
        0,
    )
    .unwrap();
    assert_eq!(row, vec![Some(1.0), Some(-2.0), None]);
    assert!(
        p.fill_named([("a", Some(1.0)), ("a", Some(2.0))], &mut row, 0)
            .is_err()
    );
    assert!(p.fill_named([("a", Some(f64::NAN))], &mut row, 0).is_err());
    assert!(pkg.predictor(9).is_err(), "max_batch above package bound");
    assert!(
        p.predict_raw(&[Some(1.0); 15], 5, &mut [0.0; 15]).is_err(),
        "batch above session bound"
    );
}

#[test]
fn corruption_and_unsupported_artifacts_fail_cleanly() {
    let d = tempfile::tempdir().unwrap();
    let good = d.path().join("good");
    Package::save(&good, spec_for(schema(&["no", "yes"]), 2, 8)).unwrap();
    let case = |name: &str, mutate: &dyn Fn(&Path), needle: &str| {
        let dir = d.path().join(name);
        copy_dir(&good, &dir);
        mutate(&dir);
        let e = Package::load(&dir)
            .err()
            .unwrap_or_else(|| panic!("{name} loaded"))
            .to_string();
        assert!(e.contains(needle), "{name}: {e}");
    };
    case(
        "flip-weight",
        &|p| {
            let w = p.join("weights.safetensors");
            let mut b = std::fs::read(&w).unwrap();
            let n = b.len() - 3;
            b[n] ^= 0xFF;
            std::fs::write(&w, b).unwrap();
        },
        "BML-A004",
    );
    case(
        "truncated",
        &|p| {
            let w = p.join("weights.safetensors");
            let b = std::fs::read(&w).unwrap();
            std::fs::write(&w, &b[..b.len() / 2]).unwrap();
        },
        "BML-A004",
    );
    case(
        "manifest-tamper",
        &|p| {
            let m = p.join("manifest.json");
            let s = std::fs::read_to_string(&m)
                .unwrap()
                .replace("test-model", "evil-model");
            std::fs::write(&m, s).unwrap();
        },
        "BML-A004",
    );
    case(
        "missing-file",
        &|p| std::fs::remove_file(p.join("weights.safetensors")).unwrap(),
        "BML-A001",
    );
    case(
        "future-version",
        &|p| edit_manifest(p, |v| v["format_version"] = 2.into()),
        "BML-A003",
    );
    case(
        "future-runtime",
        &|p| edit_manifest(p, |v| v["min_runtime_version"] = "99.0.0".into()),
        "BML-A003",
    );
    case(
        "unknown-op",
        &|p| edit_manifest(p, |v| v["graph"]["nodes"][3]["op"] = "conv2d".into()),
        "BML-A002",
    );
    case(
        "unknown-field",
        &|p| edit_manifest(p, |v| v["surprise"] = 1.into()),
        "BML-A002",
    );
    case(
        "oversized-decl",
        &|p| {
            edit_manifest(p, |v| {
                v["files"]["weights.safetensors"]["bytes"] = (1u64 << 40).into()
            })
        },
        "BML-A005",
    );
    case(
        "shape-mismatch",
        &|p| edit_manifest(p, |v| v["tensors"][0]["shape"] = serde_json::json!([3, 5])),
        "BML-A006",
    );
    case(
        "schema-width",
        &|p| {
            edit_manifest(p, |v| {
                v["schema"]["features"]
                    .as_array_mut()
                    .unwrap()
                    .push("d".into());
            })
        },
        "BML-D008",
    );
    case(
        "huge-dim",
        &|p| edit_manifest(p, |v| v["graph"]["nodes"][0]["cols"] = (1u64 << 40).into()),
        "BML-C001",
    );
    case(
        "budget",
        &|p| edit_manifest(p, |v| v["bounds"]["tensor_budget_bytes"] = 16.into()),
        "BML-C005",
    );
    case(
        "bad-vectors",
        &|p| {
            edit_manifest(p, |v| {
                v["test_vectors"]["expected"][0] = serde_json::json!([1.0, 2.0])
            })
        },
        "BML-A006",
    );
    case(
        "not-a-dir",
        &|p| {
            std::fs::remove_dir_all(p).unwrap();
            std::fs::write(p, b"x").unwrap();
        },
        "BML-A002",
    );
    case(
        "garbage-manifest",
        &|p| {
            std::fs::write(p.join("manifest.json"), b"{not json").unwrap();
            std::fs::write(p.join("MANIFEST.sha256"), sha256_hex(b"{not json")).unwrap();
        },
        "BML-A002",
    );
}

#[test]
fn wrong_expected_vectors_fail_self_test() {
    let d = tempfile::tempdir().unwrap();
    let mut spec = spec_for(schema(&["no", "yes"]), 3, 8);
    spec.test_expected = Some(vec![vec![0.123], vec![0.456]]);
    let e = Package::save(&d.path().join("p"), spec)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("BML-A007"), "{e}");
}

#[test]
fn budget_rejected_before_allocation() {
    let d = tempfile::tempdir().unwrap();
    let mut spec = spec_for(schema(&["no", "yes"]), 4, 1024);
    spec.bounds.tensor_budget_bytes = Some(1000);
    let e = Package::save(&d.path().join("p"), spec)
        .err()
        .unwrap()
        .to_string();
    assert!(
        e.contains("BML-C005") && e.contains("largest contributors"),
        "{e}"
    );
}

#[test]
fn compare_classifies_changes() {
    let d = tempfile::tempdir().unwrap();
    let data = d.path().join("data.csv");
    std::fs::write(&data, "a,b,c,y\n1,2,3,yes\n0,0,,no\n-1,4,2,yes\n").unwrap();
    let save = |name: &str, spec: PackageSpec| {
        Arc::new(Package::save(&d.path().join(name), spec).unwrap())
    };
    let base = save("base", spec_for(schema(&["no", "yes"]), 5, 8));
    let same = save("same", spec_for(schema(&["no", "yes"]), 5, 8));
    let retrained = save("retrained", spec_for(schema(&["no", "yes"]), 6, 8));
    let relabeled = save("relabeled", spec_for(schema(&["neg", "pos"]), 5, 8));
    let smaller = save("smaller", spec_for(schema(&["no", "yes"]), 5, 4));
    let opts = CompareOptions {
        data: Some(data.clone()),
        ..Default::default()
    };

    let r = compare(&base, &same, &opts).unwrap();
    assert_eq!(r.verdict, Verdict::Identical, "{r:?}");
    let r = compare(&base, &retrained, &opts).unwrap();
    assert_eq!(r.verdict, Verdict::Compatible, "{r:?}");
    assert!(r.numeric.as_ref().unwrap().max_abs_diff > 0.0);
    assert_eq!(r.numeric.as_ref().unwrap().rows, 3);
    let strict = CompareOptions {
        data: Some(data.clone()),
        max_abs_diff: Some(1e-9),
        max_rows: None,
    };
    assert_eq!(
        compare(&base, &retrained, &strict).unwrap().verdict,
        Verdict::Breaking
    );
    let r = compare(&base, &relabeled, &CompareOptions::default()).unwrap();
    assert_eq!(r.verdict, Verdict::Breaking);
    let r = compare(&base, &smaller, &CompareOptions::default()).unwrap();
    assert_eq!(r.verdict, Verdict::Breaking);
    assert!(r.changes.iter().any(|c| c.area == "bounds"));
}
