# BoundML

Train, package and embed **small dense models** (linear regression, logistic/softmax
regression, MLPs) for numeric tabular data, entirely in Rust. Each model ships with
**one versioned contract**:

- the feature schema
- the fitted preprocessing
- the model graph and weights
- bundled self-test vectors
- a tensor-memory plan that is checked before allocation

The same validated path is used everywhere: training evaluation, `boundml predict`,
the embedded `Predictor` API and the local HTTP server.

> Status: **0.1.0, first release candidate, not production-validated.** It has been
> built and tested on one Windows x64 machine. Linux/macOS CI is configured but has
> never been observed running. No external users have evaluated it yet. See
> [docs/release-status.md](docs/release-status.md).

## Which workflow does it replace?

For the supported model family, the full workflow runs without PyTorch, TensorFlow or
Python: data → training → evaluation → packaging → embedding/serving → update check.

It is **not** a drop-in replacement for either framework. It has no GPU support and
no CNNs, transformers, trees or arbitrary ONNX graphs. A narrow importer brings dense
`nn.Sequential` / Keras `Sequential` models into the same package format, and it is
verified numerically against the source framework.

## What a developer can do now

```bash
cargo build --release
```

```bash
target/release/boundml init demo
```

```bash
target/release/boundml train demo/model.toml --checkpoint demo/out/ckpt
```

```bash
target/release/boundml pack demo/out/ckpt --out demo/out/model.boundml
```

```bash
target/release/boundml serve demo/out/model.boundml
```

| Command | Purpose |
|---|---|
| `init`, `split` | scaffold an example project; deterministic train/valid/test split |
| `validate` | check config, data headers, plans and budgets without training |
| `plan --mode train/infer [--tensor-budget 64MiB]` | itemized planned tensor memory; exit 4 if over budget |
| `train [--resume] [--max-steps N]` | SGD/AdamW training with atomic, exactly resumable checkpoints |
| `evaluate` | metrics next to a constant-predictor baseline |
| `pack`, `inspect`, `verify` | build, inspect and self-test a package |
| `predict` | CSV → CSV batch prediction |
| `compare current candidate --data holdout.csv` | CI-readable compatibility verdict (exit 6 if breaking) |
| `serve` | bounded local HTTP service (`/v1/predict`, `/healthz`, `/readyz`, `/v1/model`) |
| `import` | import a PyTorch/Keras export bundle (see `migrate/`) |

Every command takes `--json` for machine-readable output. Exit codes are documented in
[docs/errors.md](docs/errors.md).

Embedding (depends only on `boundml-artifact`):

```rust
let package = Arc::new(Package::load("model.boundml".as_ref())?); // validated + plan recomputed
let mut p = package.predictor(64)?;                                 // one arena per worker
let mut raw = vec![None; p.n_features()];
p.fill_named([("temperature", Some(20.0)), ("load", Some(0.5))], &mut raw, 0)?;
let mut out = vec![0.0f32; p.output_cols()];
p.predict_raw(&raw, 1, &mut out)?;                                  // no heap allocation
```

## What has been measured (one machine; details in [docs/benchmarks.md](docs/benchmarks.md))

- Gradients agree with central finite differences of an independent f64 reference and
  with PyTorch float64. SGD/AdamW updates agree with PyTorch.
- On two public datasets, models beat constant baselines. The MLP did **not** beat
  linear/logistic regression on either one ([docs/walkthroughs.md](docs/walkthroughs.md)).
- Compared with Python + PyTorch eager on the same model: peak memory 6 MiB vs 202 MiB,
  cold start 11 ms vs 1.6 s. ONNX Runtime has **1.4–1.5× higher** batch throughput
  than BoundML's single-threaded kernels. Native runtimes (tract, RTen, ORT C API)
  were not benchmarked.
- PyTorch and Keras dense models import with max output difference ≤ 1.2e-6 on full
  test sets ([docs/results/migration.json](docs/results/migration.json)).

## What remains unproven

- That Rust teams want this integrated workflow enough to switch. No design-partner
  evaluations have been done ([plan](docs/design-partner-plan.md)).
- Any advantage over a carefully configured Burn, Aprender, ExecuTorch, tract or RTen
  setup. See [docs/decision.md](docs/decision.md) and [docs/research.md](docs/research.md).
- Behaviour on Linux and macOS, or on hardware other than the one test machine.

## Documentation

[Architecture](docs/architecture.md) · [Package & checkpoint format, compatibility policy, support matrix](docs/package-format.md) ·
[Data policy](docs/data.md) · [Errors](docs/errors.md) · [Walkthroughs](docs/walkthroughs.md) ·
[Benchmarks](docs/benchmarks.md) ([pre-declared targets](docs/benchmark-targets.md)) · [Migration](migrate/README.md) ·
[Decision](docs/decision.md) · [Research](docs/research.md) · [Release status](docs/release-status.md) ·
[Contributing](CONTRIBUTING.md) · [Third-party licenses](THIRD_PARTY.md)

License: MIT OR Apache-2.0. Demo datasets: UCI, CC BY 4.0 ([datasets/README.md](datasets/README.md)).
