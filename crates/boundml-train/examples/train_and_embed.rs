//! End-to-end in Rust: write a small CSV, train from a TOML config, package the
//! model, then load the package and predict.
//!
//! Run: `cargo run --release -p boundml-train --example train_and_embed`

use boundml_artifact::Package;
use boundml_core::Rng;
use boundml_train::{TrainConfig, TrainEvent, TrainOptions, pack, train};
use std::fmt::Write as _;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("boundml-example-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;

    // 1. Data: y = 2*a - b + noise.
    let mut rng = Rng::seed(1);
    let mut csv = String::from("a,b,y\n");
    for _ in 0..500 {
        let (a, b) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
        writeln!(csv, "{a},{b},{}", 2.0 * a - b + rng.uniform(-0.05, 0.05))?;
    }
    std::fs::write(dir.join("train.csv"), csv)?;

    // 2. Config: the same format as `boundml init` writes.
    std::fs::write(
        dir.join("model.toml"),
        r#"
name = "example"
seed = 7
[data]
train = "train.csv"
features = ["a", "b"]
label = "y"
task = "regression"
[model]
hidden = [16]
[training]
epochs = 80
batch_size = 32
optimizer = { kind = "adamw", lr = 0.01 }
[deploy]
max_batch = 16
"#,
    )?;
    let (cfg, base) = TrainConfig::load(&dir.join("model.toml"))?;

    // 3. Train (checkpoint written atomically), then package.
    let never = || false;
    let mut log = |e: &TrainEvent| {
        if let TrainEvent::Epoch(r) = e
            && (r.epoch + 1) % 20 == 0
        {
            println!("epoch {:>2}: train_loss {:.5}", r.epoch + 1, r.train_loss);
        }
    };
    train(
        &cfg,
        &base,
        TrainOptions {
            checkpoint_dir: dir.join("ckpt"),
            resume: false,
            max_steps: None,
            should_stop: &never,
            on_event: &mut log,
        },
    )?;
    pack(&dir.join("ckpt"), &dir.join("model.boundml"), None)?;

    // 4. Embed: load (validated), predict raw feature values by name.
    let package = Arc::new(Package::load(&dir.join("model.boundml"))?);
    let mut predictor = package.predictor(1)?;
    let mut raw = vec![None; predictor.n_features()];
    predictor.fill_named([("a", Some(0.5)), ("b", Some(-0.25))], &mut raw, 0)?;
    let mut out = [0.0f32; 1];
    predictor.predict_raw(&raw, 1, &mut out)?;
    println!(
        "prediction for a=0.5, b=-0.25: {:.3} (true function: 1.250)",
        out[0]
    );

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
