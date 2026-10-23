//! End-to-end: the documented walkthrough through the real binary, exit codes, and
//! prediction parity across the in-process SDK, CLI batch prediction and HTTP.

use boundml_artifact::Package;
use boundml_cli::serve::{AppState, ServeConfig};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

fn bml(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_boundml"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).to_string(),
        String::from_utf8_lossy(&o.stderr).to_string(),
    )
}

fn ok(args: &[&str]) -> String {
    let (code, out, err) = bml(args);
    assert_eq!(code, 0, "{args:?}\nstdout: {out}\nstderr: {err}");
    out
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

pub fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let code = resp[9..12].parse().unwrap();
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (code, body)
}

#[test]
fn walkthrough_exit_codes_and_parity() {
    let d = tempfile::tempdir().unwrap();
    let w = d.path().join("proj");
    ok(&["init", p(&w)]);
    let (code, _, _) = bml(&["init", p(&w)]);
    assert_eq!(code, 3, "init refuses to overwrite");
    let cfg = w.join("model.toml");
    ok(&["validate", p(&cfg)]);
    let plan_json = ok(&["--json", "plan", p(&cfg), "--mode", "infer"]);
    let v: serde_json::Value = serde_json::from_str(&plan_json).unwrap();
    assert!(
        v["result"]["report"]["arena_bytes_per_session"]
            .as_u64()
            .unwrap()
            > 0
    );
    let (code, _, err) = bml(&[
        "plan",
        p(&cfg),
        "--mode",
        "train",
        "--tensor-budget",
        "1KiB",
    ]);
    assert_eq!(code, 4, "{err}");
    assert!(err.contains("BML-C005"));

    let ck = w.join("ckpt");
    ok(&[
        "train",
        p(&cfg),
        "--checkpoint",
        p(&ck),
        "--quiet",
        "--max-steps",
        "100",
    ]);
    ok(&[
        "train",
        p(&cfg),
        "--checkpoint",
        p(&ck),
        "--quiet",
        "--resume",
    ]);
    let ev = ok(&[
        "--json",
        "evaluate",
        p(&ck),
        "--data",
        p(&w.join("data/test.csv")),
    ]);
    let ev: serde_json::Value = serde_json::from_str(&ev).unwrap();
    let rmse = ev["result"]["model"]["rmse"].as_f64().unwrap();
    let base = ev["result"]["constant_baseline"]["rmse"].as_f64().unwrap();
    assert!(rmse < 0.25 * base, "rmse {rmse} vs baseline {base}");

    let pkg = w.join("model.boundml");
    ok(&["pack", p(&ck), "--out", p(&pkg)]);
    ok(&["inspect", p(&pkg)]);
    ok(&["verify", p(&pkg)]);
    let preds = w.join("pred.csv");
    let test = w.join("data/test.csv");
    ok(&[
        "predict",
        p(&pkg),
        "--data",
        p(&test),
        "--out",
        p(&preds),
        "--batch",
        "7",
    ]);

    // Parity: CLI CSV output == in-process SDK == HTTP, on the same rows.
    let package = Arc::new(Package::load(&pkg).unwrap());
    let mut sdk = package.predictor(64).unwrap();
    let schema = package.manifest().schema.clone();
    let mut src = boundml_data::CsvSource::open(&test, &schema, true).unwrap();
    let mut row = boundml_data::RawRow::default();
    let mut rdr = csv::Reader::from_path(&preds).unwrap();
    let mut http_rows = Vec::new();
    let mut sdk_out = Vec::new();
    for rec in rdr.records().take(40) {
        let rec = rec.unwrap();
        assert!(src.read_row(&mut row).unwrap());
        let cli: f32 = rec[1].parse().unwrap();
        let s = sdk.predict_one(&row.features).unwrap()[0];
        assert_eq!(
            cli.to_bits(),
            s.to_bits(),
            "CLI vs SDK at line {}",
            row.line
        );
        sdk_out.push(s);
        let obj: serde_json::Map<String, serde_json::Value> = schema
            .features
            .iter()
            .zip(&row.features)
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.map_or(serde_json::Value::Null, |x| serde_json::json!(x)),
                )
            })
            .collect();
        http_rows.push(obj);
    }
    let state = AppState::new(
        Arc::clone(&package),
        ServeConfig {
            workers: 1,
            ..ServeConfig {
                max_rows: 64,
                ..Default::default()
            }
        },
    )
    .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = rt.spawn(boundml_cli::serve::run(state, listener, async {
        let _ = rx.await;
    }));
    let (code, body) = http(
        addr,
        "POST",
        "/v1/predict",
        &serde_json::json!({"rows": http_rows}).to_string(),
    );
    assert_eq!(code, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    for (i, s) in sdk_out.iter().enumerate() {
        let h = v["outputs"][i][0].as_f64().unwrap() as f32;
        assert_eq!(h.to_bits(), s.to_bits(), "HTTP vs SDK row {i}");
    }
    let _ = tx.send(());
    rt.block_on(server).unwrap().unwrap();

    // Update workflow: candidate trained with another seed is compatible; a tight
    // numeric threshold makes it breaking (exit 6).
    let cand_cfg = w.join("cand.toml");
    std::fs::write(
        &cand_cfg,
        std::fs::read_to_string(&cfg)
            .unwrap()
            .replace("seed = 42", "seed = 7"),
    )
    .unwrap();
    let ck2 = w.join("ckpt2");
    ok(&["train", p(&cand_cfg), "--checkpoint", p(&ck2), "--quiet"]);
    let cand = w.join("cand.boundml");
    ok(&["pack", p(&ck2), "--out", p(&cand)]);
    ok(&["compare", p(&pkg), p(&cand), "--data", p(&test)]);
    let (code, out, _) = bml(&[
        "--json",
        "compare",
        p(&pkg),
        p(&cand),
        "--data",
        p(&test),
        "--max-abs-diff",
        "1e-6",
    ]);
    assert_eq!(code, 6, "{out}");

    // Corrupted package: exit 5 with an integrity code.
    let bad = w.join("bad.boundml");
    std::fs::create_dir_all(&bad).unwrap();
    for f in ["manifest.json", "MANIFEST.sha256", "weights.safetensors"] {
        std::fs::copy(pkg.join(f), bad.join(f)).unwrap();
    }
    let wpath = bad.join("weights.safetensors");
    let mut bytes = std::fs::read(&wpath).unwrap();
    let n = bytes.len() - 1;
    bytes[n] ^= 1;
    std::fs::write(&wpath, bytes).unwrap();
    let (code, out, _) = bml(&["--json", "verify", p(&bad)]);
    assert_eq!(code, 5);
    assert!(out.contains("BML-A004"), "{out}");
    let (code, _, _) = bml(&["verify", p(&w.join("nope"))]);
    assert_eq!(code, 3);
    let (code, _, _) = bml(&["frobnicate"]);
    assert_eq!(code, 2);
}
