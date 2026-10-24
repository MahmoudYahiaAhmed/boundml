<p align="center">
  <img src="assets/banner.png" width="100%" alt="BoundML — Small models. Simple deployment. Native Rust. Train from CSV, package preprocessing and weights together, and run predictions inside your Rust application.">
</p>

**Small numeric models, packaged with their preprocessing and schema, for
straightforward deployment in Rust applications.**

BoundML trains small dense models (linear, logistic/softmax regression, small MLPs)
from CSV and writes them as one checked package directory: feature schema, fitted
preprocessing, weights, self-tests and a planned tensor-memory budget. Your Rust
application loads that package and predicts from **raw** feature values, with no
Python, ML framework or network at run time. A Python package wraps the same native
code for training and scripting.

**Who it is for:** developers who ship a numeric prediction (an estimate, a score, a
classification from a handful of measured values) inside a Rust binary or local
service, and who want the preprocessing, the feature names and a memory limit to
travel with the model instead of being re-implemented next to it.

**What it is not:** a general deep-learning framework or a faster ONNX runtime.
Large or GPU models, images, text, trees and categorical features are out of scope.
On large batches ONNX Runtime is faster (see [Evidence](#evidence)). **Status:
private beta (0.1.0)**, CPU only, no published release yet:
[what is verified](docs/release-status.md).

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-3DD9B3)](#license)

<p align="center">
  <img src="assets/demo.svg" width="100%" alt="Terminal recording of the real boundml binary: version, init, train, evaluate against a constant baseline, pack, predict with a preview of three predictions, and a memory plan rejected with exit code 4 because it needs 68228 bytes but the budget is 65536 bytes.">
</p>
<sub>Recorded from the real binary. Command typing speed is synthetic and pauses are capped at 1.2 s; output is unedited (<a href="assets/demo-transcript.txt">transcript</a>, <a href="assets/demo.cast">asciinema cast</a>).</sub>

## Choose your path

| You want to… | Use | Start at |
|---|---|---|
| Train from a CSV and predict, from a terminal | `boundml` CLI (one executable) | [Quickstart](#quickstart-csv--model--prediction) |
| Predict inside a Rust program | the `boundml-artifact` crate | [Embed it in Rust](#embed-it-in-a-rust-application) |
| Train or predict from Python (pandas, NumPy) | the `boundml` Python package | [Use it from Python](#use-it-from-python) |

All three run the same Rust code and give bitwise-identical predictions for the same
package (checked in CI). A package trained with any of them works with the others.

## Install

Nothing is on crates.io or PyPI yet, and no GitHub release has been published.
(**`pip install boundml` installs an unrelated PyPI project** with the same name; use
one of the paths below.) Today
there are three ways in, all from this repository:

| What | How | Needs |
|---|---|---|
| CLI from a release archive | build it with `python scripts/release.py --out dist` (or take the `package-smoke` artifact of a CI run), check its `.sha256`, extract it, run `boundml` from the folder | nothing at run time; one executable |
| CLI from source | `cargo install --path crates/boundml-cli --locked` | Rust 1.94+ |
| Python package | `pip install ./python` (builds a wheel), or `pip install` a wheel built by CI or `maturin build` | Rust 1.94+ to build; Python ≥ 3.9 |

Archives and wheels for Windows x86_64, Linux x86_64 and macOS arm64 are built and
tested in CI. Commands for PowerShell and bash, updating and removal:
[docs/install.md](docs/install.md). Once a release exists, the archive becomes the
recommended path.

The commands below are written as `boundml …`. In **PowerShell**, from an extracted
archive folder, type `.\boundml …` instead, or add the folder to `PATH`.

```bash
boundml --version
```
```text
boundml 0.1.0 (revision 4cb8b6c7fe, x86_64-pc-windows-msvc)
```
The revision (source commit) and target identify your build and will differ. A build
from a source archive without `.git` prints `revision unknown`.

## Quickstart: CSV → model → prediction

`boundml init` writes a config and a small synthetic CSV dataset (power use from
temperature, humidity and load; about 2% of humidity values are left blank to
exercise imputation). Nothing is downloaded.

```bash
boundml init demo
boundml train demo/model.toml --checkpoint demo/ckpt --quiet
boundml evaluate demo/ckpt --data demo/data/test.csv
```
```text
training complete after 1890 steps in 0.16s; checkpoint: demo/ckpt
final epoch: train_loss=0.03957  valid: mae=0.1527 mse=0.0366 r2=0.9954 rmse=0.1913
metrics on demo/data/test.csv:
  mae            0.1465   (constant baseline 2.2394)
  mse            0.0352   (constant baseline 11.0583)
  r2             0.9968   (constant baseline -0.0071)
  rmse           0.1877   (constant baseline 3.3254)
  rows         400.0000   (constant baseline 400.0000)
```

The model predicts `power` with an RMSE of 0.19 on held-out rows. Predicting the
training mean for every row, the "constant baseline", gives 3.33.

```bash
boundml pack demo/ckpt --out demo/model.boundml
boundml predict demo/model.boundml --data demo/data/test.csv --out demo/pred.csv --preview 3
```
```text
packed 'example-regression' -> demo/model.boundml (17 synthetic self-test vectors, verified after write)
wrote 400 predictions to demo/pred.csv in 0.001s
  line   power
     2  4.5001
     3  6.3667
     4  7.7849
```

Each prediction is the model's estimate of `power` for one CSV row. `line` is the
row's line number in the input file. The package applied the preprocessing fitted
during training (standardization; mean imputation for the blank humidity cells), so
the rows go in raw.

**When it doesn't fit**, BoundML says so before allocating any tensor storage for the plan:

```bash
boundml plan demo/model.toml --mode infer --tensor-budget 64KiB
```
```text
error: BML-C005 tensor budget exceeded: plan needs 68228 bytes but the budget is 65536 bytes; largest contributors: arena x 2 session(s) (65536 bytes), shared weights (2692 bytes), dense0.act (16384 bytes), dense0.matmul (16384 bytes), dense0.pre (16384 bytes)
```

Exit code 4. Fix it by lowering `deploy.max_batch` (128 → 64 fits) or
`deploy.sessions` in `demo/model.toml`, or by raising the budget. Every error code and its correction is
in [docs/errors.md](docs/errors.md). Full walkthrough with real datasets:
[docs/quickstart.md](docs/quickstart.md).

## Use it from Python

The `boundml` Python package runs the same native code as the CLI. Install it from a
checkout (this builds a wheel; Rust 1.94+ required) or from a built wheel file:

```bash
pip install ./python                                  # from the repository root
pip install boundml-0.1.0-cp39-abi3-win_amd64.whl     # or a wheel built by CI / maturin
```

Predict with the package from the quickstart above. This works from any directory;
only the package path matters:

```python
import boundml

model = boundml.load("demo/model.boundml")
print(model.features)   # ['temperature', 'humidity', 'load']
print(model.predict({"temperature": 20, "humidity": None, "load": 0.5}))   # [4.821724]
```

`predict` also takes a list of dicts, a pandas DataFrame (columns matched by name) or
a NumPy array in `model.features` order, and returns a float32 array of shape
`(rows, outputs)`. Values are raw; `None`/NaN means missing and follows the schema's
missing-value policy. Errors raise `boundml.BoundMLError` with a stable `code`.

Training works the same way. This example uses a config from the repository, so run
it from the repository root (the config's data paths are relative to the config):

```python
import boundml

boundml.train("examples/breast_cancer/linear.toml", "out/bc/ckpt")
print(boundml.evaluate("out/bc/ckpt", "datasets/breast_cancer/test.csv")["model"]["accuracy"])   # 0.959677...
model = boundml.pack("out/bc/ckpt", "out/bc/model.boundml")
```

Details and the full API: [python/README.md](python/README.md).

## Embed it in a Rust application

The `boundml-artifact` crate loads packages and predicts; it has no training, HTTP or
async dependencies. It is not on crates.io yet, so depend on it from a checkout:

```toml
[dependencies]
# a local checkout of this repository:
boundml-artifact = { path = "/path/to/boundml/crates/boundml-artifact" }
# or, with read access to this private repository (Cargo uses your git credentials
# when CARGO_NET_GIT_FETCH_WITH_CLI=true):
# boundml-artifact = { git = "https://github.com/MahmoudYahiaAhmed/boundml" }
```

```rust
use boundml_artifact::Package;
use std::sync::Arc;

fn main() -> Result<(), boundml_artifact::ArtifactError> {
    // Integrity, format version, graph rules and the memory plan are checked here.
    let package = Arc::new(Package::load("demo/model.boundml".as_ref())?);
    // One predictor per worker thread. Each holds one of the package's declared
    // sessions (`deploy.sessions`); creating more than that fails with BML-A010.
    let mut predictor = package.predictor(1)?;

    let mut raw = vec![None; predictor.n_features()]; // features by name; None = missing
    predictor.fill_named([("temperature", Some(20.0)), ("humidity", None), ("load", Some(0.5))], &mut raw, 0)?;
    let mut out = [0.0f32; 1];
    predictor.predict_raw(&raw, 1, &mut out)?; // preprocessing + model, no heap allocation
    println!("power = {}", out[0]); // power = 4.821724
    Ok(())
}
```

Runnable versions in this repository: [`examples/embed`](examples/embed/src/main.rs)
(`cargo run -p boundml-embed-example -- demo/model.boundml temperature=20 humidity= load=0.5`,
which also runs two concurrent sessions within the declared limit) and
[`train_and_embed`](crates/boundml-train/examples/train_and_embed.rs), which trains
from Rust code.

## Where it is useful

- **A desktop or CLI tool that estimates something locally**, such as energy use,
  a risk score or a wait time from a handful of numeric inputs. Ship one Rust binary
  plus a small package directory, with no Python runtime to bundle.
- **A Linux gateway or local service that scores readings** under a fixed memory
  budget. `boundml plan` and package loading reject configurations whose planned
  tensor memory exceeds the declared budget, and prediction allocates nothing per call.
- **A team that retrains regularly** and wants CI to stop incompatible updates.
  `boundml compare current.boundml candidate.boundml --data holdout.csv` exits 6 on
  breaking changes: renamed or removed features under a strict schema, changed
  classes, a smaller `max_batch`, a larger memory plan, or output drift beyond a
  threshold.

## How it works

<p align="center"><img src="assets/workflow.png" width="100%" alt="Workflow: data.csv and model.toml go into boundml train, which fits preprocessing on training rows only and writes a resumable checkpoint. boundml pack produces model.boundml containing schema, fitted preprocessing, graph and weights, bounds, test vectors and SHA-256 checksums. Loading checks integrity, version, graph rules, memory plan against budget and self-tests, then serves the Rust Predictor, boundml predict and boundml serve. boundml compare reports identical, compatible or breaking before an update."></p>

The package is a plain directory (`manifest.json`, `MANIFEST.sha256`,
`weights.safetensors`) with no executable code. On load, the runtime re-validates
everything and **recomputes** the memory plan rather than trusting the file.
Checksums detect corruption; they do not prove who made the package. Details:
[architecture](docs/architecture.md) · [package format](docs/package-format.md).

## Compared with alternatives

| If you need… | Consider | Why BoundML might still fit |
|---|---|---|
| Large or GPU models, many layer types | PyTorch, Burn, Candle | It doesn't; BoundML is deliberately small |
| Fastest CPU inference of an ONNX model | ONNX Runtime, tract | They have faster batch kernels (below). BoundML adds the bundled schema/preprocessing/self-test/memory contract and trains natively |
| Classical ML (trees, SVMs, clustering) in Rust | Linfa, SmartCore, Aprender | BoundML has dense models only; it adds packaging with preprocessing and compatibility checks |
| Memory-planned mobile deployment of PyTorch models | ExecuTorch | BoundML targets Rust applications and tabular inputs with no Python in the workflow |


## Evidence

One machine (Ryzen 7 8700F, Windows 11), the same weights in every runtime, five
runs each. Full tables, methodology and raw JSON: [docs/benchmarks.md](docs/benchmarks.md).

<!-- README-BENCH:BEGIN -->
| 30-feature MLP, breast-cancer data | BoundML | ONNX Runtime (Python) | PyTorch (Python) |
|---|---|---|---|
| Process start → first prediction | **5.5 ms** | 187.0 ms | 1.40 s |
| Peak process memory (OS high-water mark) | **5.7 MiB** | 50.1 MiB | 201.4 MiB |
| 1-row latency, p50, incl. preprocessing | **0.3 µs** | 9.8 µs | 26.9 µs |
| Batch-256 throughput, model only | 9.0 M rows/s | **21.5 M rows/s** | 7.4 M rows/s |
| Batch-256 throughput, incl. preprocessing | 7.0 M rows/s | **10.1 M rows/s** | 4.6 M rows/s |
<!-- README-BENCH:END -->

**ONNX Runtime is faster on large batches.** BoundML's kernels are simple and
single-threaded: model only, it reaches 42–51% of ONNX Runtime's batch-256
throughput (67–70% with preprocessing included). It is ahead on start-up, memory and
single-row calls, and the Python numbers include interpreter overhead. Every reported
run's outputs matched BoundML's for every test row at batch sizes 1, 7 and 64; a run's
timings count only after that check passes. Trained from
identical initial weights and batch order, BoundML and PyTorch reach the same test
metric to four decimals. The MLP did not beat linear models on either dataset
([walkthroughs](docs/walkthroughs.md)). The memory guarantee covers **planned tensor
storage** for a declared batch size and session count, not whole-process memory.
These are measurements of two small models on one machine, not a general accuracy
or speed claim; unfavourable results and limits are kept in the
[benchmark notes](docs/benchmarks.md) and [release status](docs/release-status.md).

## Supported in 0.1

| | |
|---|---|
| Models | linear regression, logistic/softmax regression, MLPs with ReLU/Tanh/Sigmoid |
| Tasks | regression (1 target), binary, multiclass |
| Data | numeric CSV columns mapped by name; standardization; mean imputation or reject |
| Training | SGD (momentum, L2), AdamW, minibatches, exact checkpoint/resume |
| Run | Python package (`boundml`, wheels tested on Windows, Linux and macOS), Rust `Predictor` API, `boundml predict`, local HTTP (`boundml serve`, no auth/TLS) |
| Python | 3.9+ (one abi3 wheel per platform); run on 3.11–3.14 (Windows), 3.12 (Linux) and 3.13 (CI, all three platforms). Linux wheels need glibc 2.34+ |
| Import | dense PyTorch `nn.Sequential` / Keras `Sequential`, checked against the source framework ([migrate/](migrate/README.md)) |
| Platforms | Windows x86_64, Linux x86_64 and macOS arm64: built, tested and archive smoke-tested in CI. No release published yet |
| Not supported | GPU, CNN/RNN/transformers, trees, categorical features, arbitrary ONNX, wasm/no_std |

## Documentation

[Quickstart](docs/quickstart.md) · [Install](docs/install.md) ·
[Troubleshooting](docs/troubleshooting.md) · [Errors](docs/errors.md) ·
[Walkthroughs (real data)](docs/walkthroughs.md) · [Architecture](docs/architecture.md) ·
[Package format & compatibility](docs/package-format.md) · [Data policy](docs/data.md) ·
[Benchmarks](docs/benchmarks.md) · [Release status](docs/release-status.md) ·
[Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of conduct](CODE_OF_CONDUCT.md) ·
[Support](SUPPORT.md) · [Security](SECURITY.md). Tried it on a real workload? A
*Workload report* issue is the most useful feedback right now.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option. Third-party notices: [THIRD_PARTY.md](THIRD_PARTY.md). Demo data: UCI
Machine Learning Repository, CC BY 4.0 ([datasets/README.md](datasets/README.md)).
