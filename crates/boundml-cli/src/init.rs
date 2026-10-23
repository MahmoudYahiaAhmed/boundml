//! `boundml init`: scaffold a runnable example (config + synthetic CSV data).

use boundml_core::Rng;
use std::fmt::Write as _;
use std::path::Path;

const CONFIG: &str = r#"# BoundML training configuration (paths are relative to this file).
name = "example-regression"
seed = 42

[data]
train = "data/train.csv"
valid = "data/valid.csv"
features = ["temperature", "humidity", "load"]
label = "power"
task = "regression"
missing = "impute_mean"       # or "reject"
extra_columns = "ignore"      # or "reject"
normalize = "standardize"     # or "none"
shuffle_buffer = 256

[model]
hidden = [32, 16]             # [] = linear regression
activation = "relu"           # relu | tanh | sigmoid

[training]
epochs = 30
batch_size = 32
optimizer = { kind = "adamw", lr = 0.005, weight_decay = 0.0001 }
checkpoint_every_steps = 0
tensor_budget = "16MiB"

[deploy]
max_batch = 128
sessions = 2
tensor_budget = "1MiB"
"#;

/// Synthetic data: power = f(temperature, humidity, load) + noise. Deterministic.
fn synth(path: &Path, rows: usize, seed: u64) -> std::io::Result<()> {
    let mut rng = Rng::seed(seed);
    let mut s = String::from("id,temperature,humidity,load,power\n");
    for i in 0..rows {
        let t = rng.uniform(-10.0, 35.0);
        let h = rng.uniform(10.0, 95.0);
        let l = rng.uniform(0.0, 1.0);
        let power = 3.0 + 0.08 * t + 0.02 * (t - 20.0).powi(2) * l + 1.5 * l - 0.01 * h
            + rng.uniform(-0.2, 0.2);
        // About 2% of humidity values are missing to exercise the imputation policy.
        let hs = if rng.next_f32() < 0.02 {
            String::new()
        } else {
            format!("{h:.3}")
        };
        let _ = writeln!(s, "{i},{t:.3},{hs},{l:.4},{power:.4}");
    }
    std::fs::write(path, s)
}

pub fn init(dir: &Path) -> Result<Vec<String>, String> {
    let cfg = dir.join("model.toml");
    if cfg.exists() {
        return Err(format!(
            "BML-T001 configuration error: {} already exists; refusing to overwrite",
            cfg.display()
        ));
    }
    let data = dir.join("data");
    std::fs::create_dir_all(&data).map_err(|e| format!("BML-T001 configuration error: {e}"))?;
    let io = |e: std::io::Error| format!("BML-T001 configuration error: {e}");
    synth(&data.join("train.csv"), 2000, 1).map_err(io)?;
    synth(&data.join("valid.csv"), 400, 2).map_err(io)?;
    synth(&data.join("test.csv"), 400, 3).map_err(io)?;
    std::fs::write(&cfg, CONFIG).map_err(io)?;
    Ok(vec![
        cfg.display().to_string(),
        data.join("train.csv").display().to_string(),
        data.join("valid.csv").display().to_string(),
        data.join("test.csv").display().to_string(),
    ])
}
