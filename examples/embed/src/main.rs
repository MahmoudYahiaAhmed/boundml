//! Embedding a BoundML package in an ordinary Rust application.
//!
//! Depends only on `boundml-artifact` (and through it `boundml-core` and
//! `boundml-data`): no training code, HTTP stack, Python or network.
//!
//! Usage: `cargo run -p boundml-embed-example -- <model.boundml> name=value ...`

use boundml_artifact::{Package, Predictor};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: embed <model.boundml> name=value ...")?;
    let pairs: Vec<(String, Option<f64>)> = args
        .map(|a| {
            let (k, v) = a.split_once('=').ok_or(format!("expected name=value, got {a}"))?;
            let v = if v.is_empty() { None } else { Some(v.parse::<f64>()?) };
            Ok((k.to_string(), v))
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    // 1. Load once: integrity, version, graph, schema and memory plan are validated.
    let package = Arc::new(Package::load(path.as_ref())?);
    let report = package.verify()?;
    if !report.passed {
        return Err("package failed its bundled self-test".into());
    }
    let m = package.manifest();
    println!("loaded '{}' ({:?}), inputs: {}", m.name, m.task, m.schema.features.join(", "));

    // 2. One predictor per thread; each owns its preallocated arena.
    let mut predictor: Predictor = package.predictor(1)?;
    let mut raw = vec![None; predictor.n_features()];
    predictor.fill_named(pairs.iter().map(|(k, v)| (k.as_str(), *v)), &mut raw, 0)?;
    let mut out = vec![0.0f32; predictor.output_cols()];
    predictor.predict_raw(&raw, 1, &mut out)?; // allocation-free after construction
    for (name, v) in m.output_names().iter().zip(&out) {
        println!("{name} = {v}");
    }

    // 3. Concurrency: sessions never share scratch; the package (weights) is shared.
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let pkg = Arc::clone(&package);
            let raw = raw.clone();
            std::thread::spawn(move || -> Result<f32, String> {
                let mut p = pkg.predictor(1).map_err(|e| e.to_string())?;
                let mut o = vec![0.0f32; p.output_cols()];
                p.predict_raw(&raw, 1, &mut o).map_err(|e| e.to_string())?;
                Ok(o[0])
            })
        })
        .collect();
    for t in threads {
        let v = t.join().map_err(|_| "thread panicked")??;
        assert_eq!(v.to_bits(), out[0].to_bits());
    }
    println!("2 concurrent sessions produced identical results");
    Ok(())
}
