//! Review regression: `predict --data result.csv.partial --out result.csv` deleted
//! its input (the fixed temporary name equalled the input path).

use boundml_artifact::{Bounds, Package, PackageSpec, Provenance};
use boundml_core::{Activation, DenseSpec, PlannerKind, Task};
use boundml_data::*;
use std::process::Command;

fn package(dir: &std::path::Path) -> std::path::PathBuf {
    let schema = Schema {
        features: vec!["a".into(), "b".into()],
        label: "y".into(),
        label_kind: LabelKind::Regression,
        missing: MissingPolicy::Reject,
        extra_columns: ExtraColumns::Ignore,
    };
    let dense = DenseSpec {
        inputs: 2,
        hidden: vec![],
        activation: Activation::Relu,
        task: Task::Regression,
    };
    let pre = Preprocessor {
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
    };
    let p = dir.join("m.boundml");
    Package::save(
        &p,
        PackageSpec {
            name: "r".into(),
            schema,
            preprocessing: pre,
            graph: dense.graph().unwrap().def().clone(),
            weights: vec![1.0, 2.0, 0.5],
            bounds: Bounds {
                max_batch: 8,
                tensor_budget_bytes: None,
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
    .unwrap();
    p
}

#[test]
fn predict_input_named_like_the_temporary_output_survives() {
    let d = tempfile::tempdir().unwrap();
    let pkg = package(d.path());
    let input = d.path().join("result.csv.partial");
    let content = "a,b\n1,1\n2,0\n";
    std::fs::write(&input, content).unwrap();
    let out = d.path().join("result.csv");
    let o = Command::new(env!("CARGO_BIN_EXE_boundml"))
        .args([
            "predict",
            pkg.to_str().unwrap(),
            "--data",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(&input).ok().as_deref(),
        Some(content),
        "input lost; status {:?}",
        o.status
    );
}

fn predict(
    pkg: &std::path::Path,
    data: &std::path::Path,
    out: &std::path::Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boundml"))
        .args([
            "predict",
            pkg.to_str().unwrap(),
            "--data",
            data.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap()
}

fn leftovers(dir: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect()
}

#[test]
fn predict_replaces_existing_output_and_keeps_unrelated_files() {
    let d = tempfile::tempdir().unwrap();
    let pkg = package(d.path());
    let data = d.path().join("in.csv");
    std::fs::write(&data, "a,b\n1,1\n").unwrap();
    let out = d.path().join("result.csv");
    std::fs::write(&out, "stale").unwrap();
    // A user file that happens to look like an old fixed temporary name.
    let unrelated = d.path().join("result.csv.partial");
    std::fs::write(&unrelated, "keep me").unwrap();
    assert!(predict(&pkg, &data, &out).status.success());
    assert!(std::fs::read_to_string(&out).unwrap().starts_with("line,y"));
    assert_eq!(std::fs::read_to_string(&unrelated).unwrap(), "keep me");
    assert!(leftovers(d.path()).is_empty(), "{:?}", leftovers(d.path()));
}

#[test]
fn predict_refuses_output_equal_to_input_and_cleans_up_on_failure() {
    let d = tempfile::tempdir().unwrap();
    let pkg = package(d.path());
    let data = d.path().join("in.csv");
    std::fs::write(&data, "a,b\n1,1\n").unwrap();
    let o = predict(&pkg, &data, &data);
    assert_eq!(o.status.code(), Some(3));
    assert_eq!(std::fs::read_to_string(&data).unwrap(), "a,b\n1,1\n");
    // Alias through a different spelling of the same path.
    let alias = d.path().join(".").join("in.csv");
    assert_eq!(predict(&pkg, &data, &alias).status.code(), Some(3));
    // Failure mid-way (bad row) removes only our temporary file; old output kept.
    let bad = d.path().join("bad.csv");
    std::fs::write(&bad, "a,b\n1,1\nx,2\n").unwrap();
    let out = d.path().join("result.csv");
    std::fs::write(&out, "previous").unwrap();
    assert!(!predict(&pkg, &bad, &out).status.success());
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "previous");
    assert!(leftovers(d.path()).is_empty(), "{:?}", leftovers(d.path()));
}

#[test]
fn concurrent_predicts_into_one_directory_do_not_collide() {
    let d = tempfile::tempdir().unwrap();
    let pkg = package(d.path());
    let data = d.path().join("in.csv");
    let mut csv = String::from("a,b\n");
    for i in 0..2000 {
        csv.push_str(&format!("{i},1\n"));
    }
    std::fs::write(&data, &csv).unwrap();
    let handles: Vec<_> = (0..6)
        .map(|i| {
            let (pkg, data, out) = (
                pkg.clone(),
                data.clone(),
                d.path().join(format!("out{i}.csv")),
            );
            std::thread::spawn(move || (predict(&pkg, &data, &out).status.success(), out))
        })
        .collect();
    for h in handles {
        let (ok, out) = h.join().unwrap();
        assert!(ok);
        assert_eq!(std::fs::read_to_string(&out).unwrap().lines().count(), 2001);
    }
    assert_eq!(std::fs::read_to_string(&data).unwrap(), csv);
    assert!(leftovers(d.path()).is_empty());
}
