//! BoundML inference benchmark worker. Protocol (driven by `bench/run_bench.py`):
//! prints `FIRST` right after the first prediction; writes the parity outputs (every
//! test row, raw values in, at batch sizes 1, 7 and 64) to `<parity.json>` and prints
//! `PARITY`; then times, prints one JSON line of results, and waits for a line on
//! stdin so the driver can read peak process memory. Parity is computed before and
//! outside every timed region, and every timed result goes through `black_box`.
//!
//! Reported separately: package load time (integrity checks, validation, plan
//! recomputation, self-test), preprocessing alone, warm core inference on already
//! standardized inputs, and end-to-end raw-to-output prediction. Model-only calls
//! still copy the f32 input into the planned arena: that copy is part of the public
//! `predict_into` API.
//!
//! Usage: `infer <package> <test.csv> <parity.json>`

use boundml_artifact::Package;
use boundml_core::InferenceSession;
use boundml_data::{CsvSource, RawRow};
use serde_json::json;
use std::hint::black_box;
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn timed<F: FnMut(usize)>(n: usize, mut f: F) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n)
        .map(|i| {
            let t = Instant::now();
            f(i);
            t.elapsed().as_nanos() as f64 / 1000.0
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn rows_per_s<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..20 {
        f();
    }
    let t = Instant::now();
    let mut iters = 0u64;
    while t.elapsed() < Duration::from_secs(2) {
        f();
        iters += 1;
    }
    (iters * 256) as f64 / t.elapsed().as_secs_f64()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (pkg_path, csv_path, parity_path) = (&args[1], &args[2], &args[3]);
    let t_load = Instant::now();
    let pkg = Arc::new(Package::load(pkg_path.as_ref())?);
    if !pkg.verify()?.passed {
        return Err("self-test failed".into());
    }
    let mut p = pkg.predictor(256)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let (nf, no) = (p.n_features(), p.output_cols());
    let mut src = CsvSource::open(csv_path.as_ref(), &pkg.manifest().schema, false)?;
    let mut rows: Vec<Vec<Option<f64>>> = Vec::new();
    let mut r = RawRow::default();
    while src.read_row(&mut r)? {
        rows.push(r.features.clone());
    }
    let mut out = vec![0.0f32; 256 * no];
    p.predict_raw(&rows[0], 1, &mut out[..no])?;
    println!("FIRST");
    std::io::stdout().flush()?;

    let mut parity = serde_json::Map::new();
    for bs in [1usize, 7, 64] {
        let mut all = Vec::with_capacity(rows.len() * no);
        for chunk in rows.chunks(bs) {
            let flat: Vec<Option<f64>> = chunk.iter().flatten().copied().collect();
            let n = chunk.len();
            p.predict_raw(&flat, n, &mut out[..n * no])?;
            all.extend_from_slice(&out[..n * no]);
        }
        parity.insert(bs.to_string(), json!(all));
    }
    std::fs::write(
        parity_path,
        serde_json::to_vec(&json!({"outputs_per_row": no, "batches": parity}))?,
    )?;
    println!("PARITY");
    std::io::stdout().flush()?;

    let pre = pkg.manifest().preprocessing.clone();
    let batch_raw: Vec<Option<f64>> = (0..256)
        .flat_map(|i| rows[i % rows.len()].clone())
        .collect();
    let mut batch_x = vec![0.0f32; 256 * nf];
    let mut single_x: Vec<Vec<f32>> = Vec::new();
    for row in &rows {
        let mut x = vec![0.0f32; nf];
        pre.apply(row, &mut x).map_err(|e| e.to_string())?;
        single_x.push(x);
    }
    let prep = timed(5000, |i| {
        let mut x = [0.0f32; 64];
        pre.apply(black_box(&rows[i % rows.len()]), &mut x[..nf])
            .unwrap();
        black_box(&x);
    });

    // Core: same planned execution as `Predictor`, without preprocessing.
    let mut core = InferenceSession::new(Arc::clone(pkg.model()), 256)?;
    let planned = pkg.memory_report(1)?.total_planned_bytes;
    for i in 0..200 {
        core.predict_into(&single_x[i % rows.len()], 1, &mut out[..no])?;
        p.predict_raw(&rows[i % rows.len()], 1, &mut out[..no])?;
    }
    let lat_core = timed(5000, |i| {
        core.predict_into(black_box(&single_x[i % rows.len()]), 1, &mut out[..no])
            .unwrap();
        black_box(&out[..no]);
    });
    let lat_e2e = timed(5000, |i| {
        p.predict_raw(black_box(&rows[i % rows.len()]), 1, &mut out[..no])
            .unwrap();
        black_box(&out[..no]);
    });
    for r in 0..256 {
        pre.apply(
            &batch_raw[r * nf..(r + 1) * nf],
            &mut batch_x[r * nf..(r + 1) * nf],
        )
        .map_err(|e| e.to_string())?;
    }
    let core_rps = rows_per_s(|| {
        core.predict_into(black_box(&batch_x), 256, &mut out)
            .unwrap();
        black_box(&out);
    });
    let e2e_rps = rows_per_s(|| {
        p.predict_raw(black_box(&batch_raw), 256, &mut out).unwrap();
        black_box(&out);
    });
    let q = |v: &[f64]| json!({"p50": pct(v, 0.5), "p95": pct(v, 0.95), "p99": pct(v, 0.99)});
    println!(
        "{}",
        json!({"runtime": "boundml", "threads": 1, "load_s": load_s, "preprocess_us_per_row": q(&prep),
            "latency_core_us": q(&lat_core), "latency_e2e_us": q(&lat_e2e),
            "batch256_core_rows_per_s": core_rps, "batch256_e2e_rows_per_s": e2e_rps,
            "planned_tensor_bytes": planned,
            "unavoidable_copies": "model only: predict_into copies the f32 input into the planned arena"})
    );
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(())
}
