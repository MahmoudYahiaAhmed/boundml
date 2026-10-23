//! tract (pure-Rust ONNX runtime) inference benchmark worker — the native-runtime
//! baseline. Same protocol and fields as `infer.rs`. The ONNX file is exported from
//! the same BoundML package weights by `bench/run_bench.py`.
//!
//! Preprocessing is the glue a tract user would write: standardization with the
//! package's fitted statistics, read from its `manifest.json`. Two plans are built
//! with concrete shapes (batch 1 and batch 256), which is tract's fastest mode.
//!
//! Usage: `infer_tract <model.onnx> <package dir> <data.csv>`

use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};
use tract_onnx::prelude::*;

type Plan = TypedRunnableModel;

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn plan(path: &str, batch: usize, nf: usize) -> TractResult<Arc<Plan>> {
    tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, f32::fact([batch, nf]).into())?
        .into_optimized()?
        .into_runnable()
}

fn run(
    state: &mut TypedSimpleState,
    x: &[f32],
    batch: usize,
    nf: usize,
    out: &mut [f32],
) -> TractResult<()> {
    let t = Tensor::from_shape(&[batch, nf], x)?;
    let res = state.run(tvec!(t.into_tvalue()))?;
    let view = res[0].to_plain_array_view::<f32>()?;
    for (o, v) in out.iter_mut().zip(view.iter()) {
        *o = *v;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (onnx, pkg, csv) = (&a[1], std::path::Path::new(&a[2]), &a[3]);
    let t_load = Instant::now();
    let manifest: Value = serde_json::from_slice(&std::fs::read(pkg.join("manifest.json"))?)?;
    let feats: Vec<String> = manifest["schema"]["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let stats = manifest["preprocessing"]["features"].as_array().unwrap();
    let mean: Vec<f64> = stats.iter().map(|f| f["mean"].as_f64().unwrap()).collect();
    let std: Vec<f64> = stats.iter().map(|f| f["std"].as_f64().unwrap()).collect();
    let nf = feats.len();
    let p1 = plan(onnx, 1, nf)?;
    let p256 = plan(onnx, 256, nf)?;
    let mut s1 = TypedSimpleState::new(&p1)?;
    let mut s256 = TypedSimpleState::new(&p256)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let no = match manifest["task"]["kind"].as_str() {
        Some("multiclass") => manifest["task"]["classes"].as_u64().unwrap() as usize,
        _ => 1,
    };

    let mut rdr = csv::Reader::from_path(csv)?;
    let hdr = rdr.headers()?.clone();
    let idx: Vec<usize> = feats
        .iter()
        .map(|f| hdr.iter().position(|h| h == f).unwrap())
        .collect();
    let rows: Vec<Vec<f64>> = rdr
        .records()
        .map(|r| {
            let r = r.unwrap();
            idx.iter()
                .map(|&i| r[i].trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();
    let prep = |raw: &[f64], x: &mut [f32]| {
        for k in 0..nf {
            x[k] = ((raw[k] - mean[k]) / std[k]) as f32;
        }
    };
    let mut x1 = vec![0.0f32; nf];
    let mut out = vec![0.0f32; 256 * no];
    prep(&rows[0], &mut x1);
    run(&mut s1, &x1, 1, nf, &mut out[..no])?;
    println!("FIRST");
    std::io::stdout().flush()?;

    let timed = |n: usize, f: &mut dyn FnMut(usize)| -> Vec<f64> {
        let mut v: Vec<f64> = (0..n)
            .map(|i| {
                let t = Instant::now();
                f(i);
                t.elapsed().as_nanos() as f64 / 1000.0
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let single_x: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| {
            let mut x = vec![0.0; nf];
            prep(r, &mut x);
            x
        })
        .collect();
    let prep_t = timed(5000, &mut |i| {
        let mut x = vec![0.0f32; nf];
        prep(&rows[i % rows.len()], &mut x);
        std::hint::black_box(&x);
    });
    for i in 0..200 {
        run(&mut s1, &single_x[i % rows.len()], 1, nf, &mut out[..no])?;
    }
    let lat_core = timed(5000, &mut |i| {
        run(&mut s1, &single_x[i % rows.len()], 1, nf, &mut out[..no]).unwrap()
    });
    let lat_e2e = timed(5000, &mut |i| {
        prep(&rows[i % rows.len()], &mut x1);
        run(&mut s1, &x1, 1, nf, &mut out[..no]).unwrap()
    });
    let batch_raw: Vec<&Vec<f64>> = (0..256).map(|i| &rows[i % rows.len()]).collect();
    let mut bx = vec![0.0f32; 256 * nf];
    for (r, raw) in batch_raw.iter().enumerate() {
        prep(raw, &mut bx[r * nf..(r + 1) * nf]);
    }
    let rps = |f: &mut dyn FnMut()| {
        for _ in 0..20 {
            f();
        }
        let t = Instant::now();
        let mut it = 0u64;
        while t.elapsed() < Duration::from_secs(2) {
            f();
            it += 1;
        }
        (it * 256) as f64 / t.elapsed().as_secs_f64()
    };
    let core_rps = rps(&mut || run(&mut s256, &bx, 256, nf, &mut out).unwrap());
    let e2e_rps = rps(&mut || {
        for (r, raw) in batch_raw.iter().enumerate() {
            prep(raw, &mut bx[r * nf..(r + 1) * nf]);
        }
        run(&mut s256, &bx, 256, nf, &mut out).unwrap()
    });
    let checksum: f64 = out.iter().map(|v| f64::from(*v)).sum();
    let q = |v: &[f64]| json!({"p50": pct(v, 0.5), "p95": pct(v, 0.95), "p99": pct(v, 0.99)});
    println!(
        "{}",
        json!({"runtime": "tract", "threads": 1, "load_s": load_s, "preprocess_us_per_row": q(&prep_t),
        "latency_core_us": q(&lat_core), "latency_e2e_us": q(&lat_e2e),
        "batch256_core_rows_per_s": core_rps, "batch256_e2e_rows_per_s": e2e_rps, "checksum": checksum})
    );
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(())
}
