//! Inference benchmark worker. Protocol (driven by bench/run_bench.py):
//! prints `FIRST` right after the first prediction, then one JSON line of results,
//! then waits for a line on stdin so the driver can read peak process memory.
//!
//! Timing includes BoundML's packaged preprocessing (raw values in, outputs out).

use boundml_artifact::Package;
use boundml_data::{CsvSource, RawRow};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (pkg_path, csv_path) = (&args[1], &args[2]);
    let pkg = Arc::new(Package::load(pkg_path.as_ref())?);
    if !pkg.verify()?.passed {
        return Err("self-test failed".into());
    }
    let mut p = pkg.predictor(256)?;
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

    for i in 0..200 {
        p.predict_raw(&rows[i % rows.len()], 1, &mut out[..no])?;
    }
    let mut lat = Vec::with_capacity(5000);
    for i in 0..5000 {
        let t = Instant::now();
        p.predict_raw(&rows[i % rows.len()], 1, &mut out[..no])?;
        lat.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let batch: Vec<Option<f64>> = (0..256)
        .flat_map(|i| rows[i % rows.len()].clone())
        .collect();
    for _ in 0..20 {
        p.predict_raw(&batch, 256, &mut out)?;
    }
    let t = Instant::now();
    let mut iters = 0u64;
    while t.elapsed() < Duration::from_secs(2) {
        p.predict_raw(&batch, 256, &mut out)?;
        iters += 1;
    }
    let rows_per_s = (iters * 256) as f64 / t.elapsed().as_secs_f64();
    let checksum: f64 = out.iter().map(|v| f64::from(*v)).sum();
    println!(
        "{}",
        serde_json::json!({"runtime": "boundml", "features": nf, "latency_us": {"p50": pct(&lat, 0.5), "p95": pct(&lat, 0.95), "p99": pct(&lat, 0.99)},
            "batch256_rows_per_s": rows_per_s, "checksum": checksum})
    );
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(())
}
