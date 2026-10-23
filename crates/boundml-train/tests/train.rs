use boundml_core::Rng;
use boundml_train::*;
use std::fmt::Write as _;
use std::path::Path;

fn write_data(dir: &Path, name: &str, rows: usize, seed: u64, classify: bool) {
    let mut rng = Rng::seed(seed);
    let mut s = String::from("id,x1,x2,x3,target\n");
    for i in 0..rows {
        let (a, b, c) = (
            rng.uniform(-2.0, 2.0),
            rng.uniform(0.0, 10.0),
            rng.uniform(-1.0, 1.0),
        );
        let noise = rng.uniform(-0.1, 0.1);
        let v = 1.5 * a - 0.3 * b + a * c + noise;
        let label = if classify {
            (if v > -1.5 { "pos" } else { "neg" }).to_string()
        } else {
            format!("{v}")
        };
        writeln!(s, "{i},{a},{b},{c},{label}").unwrap();
    }
    std::fs::write(dir.join(name), s).unwrap();
}

fn config(dir: &Path, classify: bool, extra: &str) -> std::path::PathBuf {
    write_data(dir, "train.csv", 600, 1, classify);
    write_data(dir, "valid.csv", 200, 2, classify);
    write_data(dir, "test.csv", 200, 3, classify);
    let (task, classes) = if classify {
        ("classification", r#"classes = ["neg", "pos"]"#)
    } else {
        ("regression", "")
    };
    let toml = format!(
        r#"
name = "synthetic"
seed = 7

[data]
train = "train.csv"
valid = "valid.csv"
features = ["x1", "x2", "x3"]
label = "target"
task = "{task}"
{classes}
shuffle_buffer = 64

[model]
hidden = [16, 8]
activation = "relu"

[training]
epochs = 12
batch_size = 32
optimizer = {{ kind = "adamw", lr = 0.01, weight_decay = 0.0001 }}
{extra}

[deploy]
max_batch = 64
"#
    );
    let p = dir.join("model.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

fn run(cfg_path: &Path, ckpt: &Path, resume: bool, max_steps: Option<u64>) -> TrainOutcome {
    let (cfg, base) = TrainConfig::load(cfg_path).unwrap();
    let stop = || false;
    let mut sink = |_: &TrainEvent| {};
    train(
        &cfg,
        &base,
        TrainOptions {
            checkpoint_dir: ckpt.to_path_buf(),
            resume,
            max_steps,
            should_stop: &stop,
            on_event: &mut sink,
        },
    )
    .unwrap()
}

#[test]
fn regression_learns_beats_baseline_and_packs() {
    let d = tempfile::tempdir().unwrap();
    let cfg = config(d.path(), false, "");
    let ck = d.path().join("ckpt");
    let out = run(&cfg, &ck, false, None);
    assert!(out.completed);
    let first = out.history.first().unwrap().valid["rmse"];
    let last = out.history.last().unwrap().valid["rmse"];
    assert!(last < first, "valid rmse should improve: {first} -> {last}");

    let eval = evaluate(&ck, &d.path().join("test.csv")).unwrap();
    let base = eval.baseline.unwrap();
    assert!(
        eval.model["rmse"] < 0.5 * base["rmse"],
        "model {:?} vs baseline {:?}",
        eval.model,
        base
    );

    let pkg_dir = d.path().join("model.boundml");
    pack(&ck, &pkg_dir, None).unwrap();
    let from_pkg = evaluate(&pkg_dir, &d.path().join("test.csv")).unwrap();
    assert_eq!(
        from_pkg.model["rmse"], eval.model["rmse"],
        "package and checkpoint agree"
    );
    assert_eq!(from_pkg.baseline.unwrap()["rmse"], base["rmse"]);
}

#[test]
fn classification_learns() {
    let d = tempfile::tempdir().unwrap();
    let cfg = config(d.path(), true, "");
    let ck = d.path().join("ckpt");
    run(&cfg, &ck, false, None);
    let eval = evaluate(&ck, &d.path().join("test.csv")).unwrap();
    let base = eval.baseline.unwrap();
    assert!(eval.model["accuracy"] > 0.9, "{:?}", eval.model);
    assert!(eval.model["accuracy"] > base["accuracy"] + 0.1);
    assert!(eval.model["log_loss"] < base["log_loss"]);
}

#[test]
fn interrupted_and_resumed_run_matches_uninterrupted() {
    let d = tempfile::tempdir().unwrap();
    let cfg = config(d.path(), true, "checkpoint_every_steps = 7");
    let full = d.path().join("full");
    run(&cfg, &full, false, None);

    let part = d.path().join("part");
    // Stop mid-epoch (19 steps per epoch for 600 rows / 32), twice, then finish.
    let o1 = run(&cfg, &part, false, Some(30));
    assert!(!o1.completed);
    let o2 = run(&cfg, &part, true, Some(101));
    assert!(!o2.completed);
    let o3 = run(&cfg, &part, true, None);
    assert!(o3.completed);

    let a = Checkpoint::load(&full).unwrap();
    let b = Checkpoint::load(&part).unwrap();
    assert_eq!(a.params, b.params, "bitwise identical parameters");
    assert_eq!(a.optimizer, b.optimizer);
    assert_eq!(a.state.history, b.state.history);
    assert!(
        part.with_file_name("part.prev").exists(),
        "previous checkpoint is kept"
    );

    // Resuming a completed run is a no-op; resuming with a changed config is refused.
    assert!(run(&cfg, &part, true, None).completed);
    let text = std::fs::read_to_string(&cfg)
        .unwrap()
        .replace("epochs = 12", "epochs = 13");
    std::fs::write(&cfg, text).unwrap();
    let (c2, base) = TrainConfig::load(&cfg).unwrap();
    let stop = || false;
    let mut sink = |_: &TrainEvent| {};
    let e = train(
        &c2,
        &base,
        TrainOptions {
            checkpoint_dir: part.clone(),
            resume: true,
            max_steps: None,
            should_stop: &stop,
            on_event: &mut sink,
        },
    );
    assert!(e.err().unwrap().to_string().contains("BML-T002"));
}

#[test]
fn over_budget_training_is_rejected_before_allocation() {
    let d = tempfile::tempdir().unwrap();
    let cfg = config(d.path(), false, "tensor_budget = \"4KiB\"");
    let (cfg, base) = TrainConfig::load(&cfg).unwrap();
    let stop = || false;
    let mut events = 0;
    let mut sink = |_: &TrainEvent| events += 1;
    let e = train(
        &cfg,
        &base,
        TrainOptions {
            checkpoint_dir: d.path().join("ck"),
            resume: false,
            max_steps: None,
            should_stop: &stop,
            on_event: &mut sink,
        },
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("BML-C005"), "{e}");
    assert_eq!(events, 0);
    assert!(!d.path().join("ck").exists());
}

#[test]
fn invalid_configs_are_rejected() {
    let d = tempfile::tempdir().unwrap();
    let p = config(d.path(), false, "");
    let good = std::fs::read_to_string(&p).unwrap();
    for (from, to) in [
        ("epochs = 12", "epochs = 0"),
        ("task = \"regression\"", "task = \"classification\""),
        ("hidden = [16, 8]", "hidden = [0]"),
        ("kind = \"adamw\"", "kind = \"rmsprop\""),
        ("max_batch = 64", "max_batch = 64\nsurprise = 1"),
        (
            "features = [\"x1\", \"x2\", \"x3\"]",
            "features = [\"x1\", \"x1\"]",
        ),
    ] {
        std::fs::write(&p, good.replace(from, to)).unwrap();
        assert!(TrainConfig::load(&p).is_err(), "{to}");
    }
}
