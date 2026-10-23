# Quickstart

Two paths, both from a release archive or a source install (see [install.md](install.md)):

1. **CLI**: train from CSV, evaluate, package and predict.
2. **Rust**: load the resulting package inside your own program.

Outputs below are copied from a real run (`assets/demo-transcript.txt`). Timings
differ between machines; metrics are deterministic for a given binary and platform.

Measured time to the first prediction with a prebuilt Windows archive: **0.02 s** to extract the archive with Python's `zipfile` (about 0.5 s with PowerShell's `Expand-Archive`), then **0.22 s** for `init`, `train`, `pack` and `predict` together (train 0.17 s), measured by `scripts/release.py` on the machine described in [benchmarks.md](benchmarks.md). Downloading the archive (about 1.6 MB) and reading these instructions take longer than running them. A source build (`cargo install`) adds the compile time measured in [install.md](install.md).

## Path 1 — CLI

### 1. Create an example project

```bash
boundml init demo
```

(If you already ran the README quickstart, `demo/` exists and `init` refuses to
overwrite it with `BML-T001`. Use another directory name, for example
`boundml init demo2`, and adjust the paths below.) In PowerShell, write `.\boundml`
when running from the extracted archive folder.

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
demo/model.toml: valid
  task Regression, 3 features, data headers ok (2 file(s))
  training plan 23.0 KiB (budget 16MiB), inference plan 66.6 KiB (budget 1MiB)
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
metrics on demo/data/test.csv:
  mae            0.1465   (constant baseline 2.2394)
  mse            0.0352   (constant baseline 11.0583)
  r2             0.9968   (constant baseline -0.0071)
  rmse           0.1877   (constant baseline 3.3254)
  rows         400.0000   (constant baseline 400.0000)
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
packed 'example-regression' -> demo/model.boundml (17 synthetic self-test vectors, verified after write)
integrity, version and consistency checks passed; self-test: 17 vector(s), max abs error 0e0 -> PASS
wrote 400 predictions to demo/pred.csv in 0.001s
  line   power
     2  4.5001
     3  6.3667
     4  7.7849
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

```text
serving 'example-regression' on http://127.0.0.1:8080 (workers 2, queue 16, max rows 128, timeout 5s); Ctrl-C to stop
```

The server keeps running in this terminal. Send a request from a **second** terminal.
bash / zsh / Git Bash:

```bash
curl -s -X POST localhost:8080/v1/predict -H "content-type: application/json" -d '{"rows":[{"temperature":20,"humidity":50,"load":0.5}]}'
```

PowerShell, where `curl` is an alias for `Invoke-WebRequest`:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/predict -ContentType "application/json" -Body '{"rows":[{"temperature":20,"humidity":50,"load":0.5}]}' | ConvertTo-Json -Depth 5 -Compress
```

Response:

```text
{"model":"example-regression","output_names":["power"],"outputs":[[4.8455453]]}
```

Stop with Ctrl-C; in-flight requests finish first. The server binds to `127.0.0.1` and
has no authentication or TLS.

### 6. See a failure and fix it

Ask for a plan that cannot fit:

```bash
boundml plan demo/model.toml --mode infer --tensor-budget 64KiB
```

```text
error: BML-C005 tensor budget exceeded: plan needs 68228 bytes but the budget is 65536 bytes; largest contributors: arena x 2 session(s) (65536 bytes), shared weights (2692 bytes), dense0.act (16384 bytes), dense0.matmul (16384 bytes), dense0.pre (16384 bytes)
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
  TOTAL planned tensor bytes             34.6 KiB  (35460 B)
  not covered: package/checkpoint loading buffers; graph and manifest metadata; ...
  within budget of 64.0 KiB (65536 B)
```

A schema error looks like this. Make a copy of the test file whose `humidity` column is
renamed to `humidity_pct` (bash, then PowerShell):

```bash
sed '1s/humidity/humidity_pct/' demo/data/test.csv > demo/data/renamed.csv
```

```powershell
$c = Get-Content demo\data\test.csv; $c[0] = $c[0] -replace 'humidity', 'humidity_pct'; $c | Set-Content demo\data\renamed.csv
```

Then predict:

```bash
boundml predict demo/model.boundml --data demo/data/renamed.csv --out demo/p2.csv
```

```text
error: BML-D004 header of demo/data/renamed.csv: missing feature column(s): humidity
```

Exit code 3, and no output file is written. Columns are matched by name, so the fix
is to restore the header name `humidity` (or retrain with the new name). Column
*order* never matters. See [troubleshooting](troubleshooting.md) for the related header errors.

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
use boundml_artifact::Package;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: power-app <model.boundml> [name=value ...]")?;
    let package = Arc::new(Package::load(path.as_ref())?); // checked: hashes, version, graph, memory plan
    let mut predictor = package.predictor(1)?;

    // Feature values by name; an empty value means "missing".
    let pairs: Vec<(String, Option<f64>)> = args
        .map(|a| {
            let (k, v) = a
                .split_once('=')
                .ok_or(format!("expected name=value, got {a}"))?;
            Ok((
                k.to_string(),
                if v.is_empty() { None } else { Some(v.parse()?) },
            ))
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let mut raw = vec![None; predictor.n_features()];
    predictor.fill_named(pairs.iter().map(|(k, v)| (k.as_str(), *v)), &mut raw, 0)?;

    let mut out = vec![0.0f32; predictor.output_cols()];
    predictor.predict_raw(&raw, 1, &mut out)?; // fitted preprocessing + model
    for (name, value) in package.manifest().output_names().iter().zip(&out) {
        println!("{name} = {value:.3}");
    }
    Ok(())
}
```

```bash
cargo run --release --manifest-path power-app/Cargo.toml -- demo/model.boundml temperature=20 humidity= load=0.5
```

```text
power = 4.822
```

The arguments after the package are `name=value` pairs (an empty value means missing). The program loads
the package once, validates it, and then predicts without heap allocation per call.
For several threads, create one `Predictor` per thread from the same
`Arc<Package>`. Never share a `Predictor` between threads.

## Real data

[walkthroughs.md](walkthroughs.md) runs the same workflow on the UCI Breast Cancer
(classification) and Wine Quality (regression) datasets. It reports every metric
against constant and linear baselines, including the cases where the MLP does not
win. The release archive includes the breast-cancer data and configs. Run these from
the **extracted archive folder** (or a repository checkout), because the config refers
to `../../datasets/` relative to itself:

```bash
boundml train examples/breast_cancer/linear.toml --checkpoint bc/ckpt --quiet
boundml evaluate bc/ckpt --data datasets/breast_cancer/test.csv
```

```text
metrics on datasets/breast_cancer/test.csv:
  accuracy       0.9597   (constant baseline 0.5806)
  f1             0.9495   (constant baseline 0.0000)
  log_loss       0.1379   (constant baseline 0.6870)
  precision      1.0000   (constant baseline 0.0000)
  recall         0.9038   (constant baseline 0.0000)
  rows         124.0000   (constant baseline 124.0000)
```

This is logistic regression on 30 measurements of breast-tissue images, predicting
`diagnosis` (`M` = malignant is the positive class). It demonstrates the workflow on
real data; it is not a medical tool.
