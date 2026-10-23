//! Sustained stress test: repeated package loading, concurrent prediction sessions,
//! training steps with checkpoint/resume cycles. Prints one JSON progress line per
//! interval; bench/run_stress.py samples process memory alongside.
//!
//! Usage: `stress <package> <data.csv> <train-config.toml> <work-dir> <seconds>`

use boundml_artifact::Package;
use boundml_data::{CsvSource, RawRow};
use boundml_train::{TrainConfig, TrainEvent, TrainOptions, train};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (pkg_path, csv, cfg_path, work) = (
        PathBuf::from(&a[1]),
        PathBuf::from(&a[2]),
        PathBuf::from(&a[3]),
        PathBuf::from(&a[4]),
    );
    let secs: u64 = a[5].parse()?;
    let pkg = Arc::new(Package::load(&pkg_path)?);
    let mut src = CsvSource::open(&csv, &pkg.manifest().schema, false)?;
    let mut rows: Vec<Option<f64>> = Vec::new();
    let mut r = RawRow::default();
    let mut n = 0usize;
    while src.read_row(&mut r)? {
        rows.extend_from_slice(&r.features);
        n += 1;
    }
    let rows = Arc::new(rows);
    let nf = pkg.manifest().schema.features.len();
    let stop = Arc::new(AtomicBool::new(false));
    let predictions = Arc::new(AtomicU64::new(0));
    let reference = {
        let mut p = pkg.predictor(64)?;
        let mut o = vec![0.0f32; 64 * p.output_cols()];
        let b = n.min(64);
        p.predict_raw(&rows[..b * nf], b, &mut o[..b * p.output_cols()])?;
        Arc::new(o[..b * p.output_cols()].to_vec())
    };
    // 4 concurrent sessions sharing one package; each checks results never drift.
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let (pkg, rows, stop, count, reference) = (
                Arc::clone(&pkg),
                Arc::clone(&rows),
                Arc::clone(&stop),
                Arc::clone(&predictions),
                Arc::clone(&reference),
            );
            std::thread::spawn(move || -> Result<(), String> {
                let mut p = pkg.predictor(64).map_err(|e| e.to_string())?;
                let no = p.output_cols();
                let b = n.min(64);
                let mut o = vec![0.0f32; b * no];
                while !stop.load(Ordering::Relaxed) {
                    p.predict_raw(&rows[..b * nf], b, &mut o)
                        .map_err(|e| e.to_string())?;
                    if o[..] != reference[..] {
                        return Err("prediction drift between sessions".into());
                    }
                    count.fetch_add(b as u64, Ordering::Relaxed);
                }
                Ok(())
            })
        })
        .collect();

    let (cfg, base) = TrainConfig::load(&cfg_path)?;
    let ckpt = work.join("stress-ckpt");
    let started = Instant::now();
    let (mut loads, mut train_cycles, mut steps) = (0u64, 0u64, 0u64);
    let mut last_report = Instant::now();
    let never = || false;
    while started.elapsed() < Duration::from_secs(secs) {
        let p = Package::load(&pkg_path)?;
        if !p.verify()?.passed {
            return Err("self-test failed on reload".into());
        }
        loads += 1;
        let resume = ckpt.exists();
        let mut sink = |_: &TrainEvent| {};
        let out = train(
            &cfg,
            &base,
            TrainOptions {
                checkpoint_dir: ckpt.clone(),
                resume,
                max_steps: Some(steps + 50),
                should_stop: &never,
                on_event: &mut sink,
            },
        )?;
        steps = out.global_step;
        train_cycles += 1;
        if out.completed {
            std::fs::remove_dir_all(&ckpt)?;
            steps = 0;
        }
        if last_report.elapsed() >= Duration::from_secs(5) {
            println!(
                "{}",
                serde_json::json!({"elapsed_s": started.elapsed().as_secs_f64(), "package_loads": loads, "train_resume_cycles": train_cycles, "predicted_rows": predictions.load(Ordering::Relaxed)})
            );
            std::io::stdout().flush()?;
            last_report = Instant::now();
        }
    }
    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join().map_err(|_| "worker panicked")??;
    }
    println!(
        "{}",
        serde_json::json!({"done": true, "elapsed_s": started.elapsed().as_secs_f64(), "package_loads": loads, "train_resume_cycles": train_cycles, "predicted_rows": predictions.load(Ordering::Relaxed)})
    );
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(())
}
