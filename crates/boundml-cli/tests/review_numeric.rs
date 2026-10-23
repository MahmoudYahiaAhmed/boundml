//! Review regressions: non-finite model outputs must fail closed in the CLI, HTTP
//! and import paths.

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

/// Package whose finite weights overflow to NaN for input (2, 2); no test vectors.
fn nan_package(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let p = package(dir);
    let dest = dir.join(name);
    std::fs::rename(&p, &dest).unwrap();
    // Rewrite weights to [3e38, -3e38, 0] and fix the integrity hashes.
    let tensors = vec![
        (
            "dense0.weight".to_string(),
            [2usize, 1],
            &[3e38f32, -3e38][..],
        ),
        ("dense0.bias".to_string(), [1, 1], &[0.0f32][..]),
    ];
    let bytes = boundml_artifact::fsutil::encode_tensors(&tensors).unwrap();
    std::fs::write(dest.join("weights.safetensors"), &bytes).unwrap();
    let mp = dest.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    m["files"]["weights.safetensors"]["sha256"] =
        boundml_artifact::fsutil::sha256_hex(&bytes).into();
    m["files"]["weights.safetensors"]["bytes"] = bytes.len().into();
    let mb = serde_json::to_vec_pretty(&m).unwrap();
    std::fs::write(&mp, &mb).unwrap();
    std::fs::write(
        dest.join("MANIFEST.sha256"),
        format!("{}\n", boundml_artifact::fsutil::sha256_hex(&mb)),
    )
    .unwrap();
    dest
}

#[test]
fn cli_predict_refuses_non_finite_outputs() {
    let d = tempfile::tempdir().unwrap();
    let pkg = nan_package(d.path(), "nan.boundml");
    let input = d.path().join("in.csv");
    std::fs::write(&input, "a,b\n1,0\n2,2\n").unwrap();
    let out = d.path().join("pred.csv");
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
    let err = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(3), "{err}");
    assert!(err.contains("BML-C007"), "{err}");
    assert!(
        !out.exists(),
        "no prediction file with NaN rows may be written"
    );
    // A finite package still predicts normally through the same command.
    let good = package(&d.path().join("good"));
    let o = Command::new(env!("CARGO_BIN_EXE_boundml"))
        .args([
            "predict",
            good.to_str().unwrap(),
            "--data",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(o.status.success());
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("1.5") && text.contains("6.5"), "{text}");
}

#[test]
fn http_non_finite_output_is_an_error_not_null() {
    use boundml_cli::serve::{AppState, ServeConfig, run};
    use std::io::{Read, Write};
    let d = tempfile::tempdir().unwrap();
    let pkg = std::sync::Arc::new(
        boundml_artifact::Package::load(&nan_package(d.path(), "nan.boundml")).unwrap(),
    );
    let state = AppState::new(
        pkg,
        ServeConfig {
            workers: 1,
            max_rows: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let h = rt.spawn(run(state, listener, async {
        let _ = rx.await;
    }));
    let body = r#"{"rows":[{"a":2,"b":2}]}"#;
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(s, "POST /v1/predict HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 422"), "{resp}");
    assert!(
        resp.contains("BML-C007") && !resp.contains("null"),
        "{resp}"
    );
    let _ = tx.send(());
    rt.block_on(h).unwrap().unwrap();
}

#[test]
fn import_rejects_non_finite_reference_outputs() {
    let d = tempfile::tempdir().unwrap();
    let b = d.path().join("bundle");
    std::fs::create_dir_all(&b).unwrap();
    let tensors = vec![
        ("layer0.weight".to_string(), [1usize, 2], &[1.0f32, 1.0][..]),
        ("layer0.bias".to_string(), [1, 1], &[0.0f32][..]),
    ];
    std::fs::write(
        b.join("weights.safetensors"),
        boundml_artifact::fsutil::encode_tensors(&tensors).unwrap(),
    )
    .unwrap();
    // serde_json cannot carry NaN, so the exporter-side value is the largest finite
    // number whose f32 conversion overflows to +inf.
    let bundle = serde_json::json!({
        "format": "boundml-import", "version": 1, "source": "pytorch", "framework_version": "test",
        "name": "nf", "features": ["a", "b"], "label": "y", "classes": [],
        "layers": [{"type": "dense", "weight": "layer0.weight", "bias": "layer0.bias", "weight_layout": "out_in"}],
        "output_activation": "none",
        "reference": {"inputs": [[1, 1], [2, 2], [0, 1], [1, 0]], "outputs": [[2.0], [1e39], [1.0], [1.0]]}
    });
    std::fs::write(b.join("bundle.json"), serde_json::to_vec(&bundle).unwrap()).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_boundml"))
        .args([
            "import",
            b.to_str().unwrap(),
            "--out",
            d.path().join("m.boundml").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        !o.status.success(),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    assert!(!d.path().join("m.boundml").exists());
}
