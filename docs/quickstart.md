# Quickstart

Two paths, both from a release archive or a source install (see [install.md](install.md)):

1. **CLI**: train from CSV, evaluate, package and predict.
2. **Rust**: load the resulting package inside your own program.

Outputs below are copied from a real run (`assets/demo-transcript.txt`). Timings
differ between machines; metrics are deterministic for a given binary and platform.

Measured time to the first prediction with a prebuilt Windows archive: QUICKSTART_TIMING

## Path 1 — CLI

### 1. Create an example project

```bash
boundml init demo
```

This writes `demo/model.toml` and three CSV files (`train`, `valid`, `test`)
generated deterministically by `boundml` itself. The data is synthetic: electrical
`power` as a function of `temperature`, `humidity` and `load`, with about 2% of
humidity values left empty. Open `demo/model.toml`. It declares the features by name,
the label, the missing-value policy (`impute_mean`), the network (`hidden = [32, 16]`),
the optimizer and two tensor budgets.

### 2. Check it before training

```bash
boundml validate demo/model.toml
```

```text
VALIDATE_OUTPUT
```

This checks the config, the CSV headers (columns are matched by name) and both
memory plans against their budgets, without training anything.

### 3. Train and evaluate

```bash
boundml train demo/model.toml --checkpoint demo/ckpt
```

Training prints one line per epoch with the validation metrics. Interrupt it with
Ctrl-C at any time, then continue exactly where it stopped with
`boundml train demo/model.toml --checkpoint demo/ckpt --resume`.

```bash
boundml evaluate demo/ckpt --data demo/data/test.csv
```

```text
EVALUATE_OUTPUT
```

`constant baseline` is what you would get by always predicting the training mean. The
model's RMSE should be far below it.

### 4. Package and predict

```bash
boundml pack demo/ckpt --out demo/model.boundml
boundml verify demo/model.boundml
boundml predict demo/model.boundml --data demo/data/test.csv --out demo/pred.csv --preview 3
```

```text
PACK_VERIFY_PREDICT_OUTPUT
```

`demo/pred.csv` holds one row per input row: the input line number and the predicted
`power`. The raw values went through the package's own fitted preprocessing, so no
scaling code is needed on the caller's side. `boundml inspect demo/model.boundml`
shows the whole contract: features, policies, preprocessing, provenance, self-tests
and planned memory.

### 5. Serve it (optional)

```bash
boundml serve demo/model.boundml
```

```bash
curl -s -X POST localhost:8080/v1/predict -H "content-type: application/json" -d '{"rows":[{"temperature":20,"humidity":50,"load":0.5}]}'
```

```text
SERVE_OUTPUT
```

Stop with Ctrl-C; in-flight requests finish first. The server binds to `127.0.0.1` and
has no authentication or TLS.

### 6. See a failure and fix it

Ask for a plan that cannot fit:

```bash
boundml plan demo/model.toml --mode infer --tensor-budget 64KiB
```

```text
BUDGET_OUTPUT
```

Exit code 4. The message names the largest contributors: the arena is planned for two
sessions of `max_batch = 128`. Halve it in `demo/model.toml`:

```toml
[deploy]
max_batch = 64
```

```bash
boundml plan demo/model.toml --mode infer --tensor-budget 64KiB
```

```text
BUDGET_FIXED_OUTPUT
```

A schema error looks like this. Rename a column and predict:

SCHEMA_ERROR_BLOCK

## Path 2 — Rust

Create a program next to your package (from a checkout of this repository, until the
crates are published):

```bash
cargo new power-app
```

`power-app/Cargo.toml`:

```toml
[dependencies]
boundml-artifact = { path = "<path to this repository>/crates/boundml-artifact" }
```

`power-app/src/main.rs`:

```rust
EMBED_SOURCE
```

```bash
cargo run --release --manifest-path power-app/Cargo.toml -- demo/model.boundml
```

```text
EMBED_OUTPUT
```

The arguments are `name=value` pairs (an empty value means missing). The program loads
the package once, validates it, and then predicts without heap allocation per call.
For several threads, create one `Predictor` per thread from the same
`Arc<Package>`. Never share a `Predictor` between threads.

## Real data

[walkthroughs.md](walkthroughs.md) runs the same workflow on the UCI Breast Cancer
(classification) and Wine Quality (regression) datasets. It reports every metric
against constant and linear baselines, including the cases where the MLP does not
win. The release archive includes the breast-cancer data and configs:

```bash
boundml train examples/breast_cancer/linear.toml --checkpoint bc/ckpt --quiet
boundml evaluate bc/ckpt --data datasets/breast_cancer/test.csv
```
