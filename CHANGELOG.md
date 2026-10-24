# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may contain
breaking changes; they are called out under **Breaking**.

## [Unreleased]

### Added
- `boundml` Python package (`python/`, PyO3 + maturin, one abi3 wheel for Python ≥
  3.9): `train`, `evaluate`, `pack`, `load`, and `Model.predict` from pandas
  DataFrames, dict rows or NumPy arrays, with predictions bitwise identical to the
  CLI. Errors raise `boundml.BoundMLError` with the stable `BML-xxxx` code.

### Fixed
- Python `Model.predict` on a pandas DataFrame or dict rows no longer fails when
  the input has non-numeric columns that are not features (such as a string label
  column); only feature columns are converted, unknown columns follow the schema's
  `extra_columns` policy, and a non-numeric feature value raises `BoundMLError`
  `BML-D005`.
- Benchmark harness: parity files must contain exactly batch sizes 1, 7 and 64 with
  the expected number of finite values; workers are settled on every failure path;
  PyTorch builds its controlled batch order before its training timer.
- Server: a rejected request's body is drained for at most 250 ms and 64 KiB, so an
  overload 503 can reach the client (reproduced Windows reset, error 10053).

### Removed
- Launch drafts (`launch/`), internal planning notes (`docs/decision.md`,
  `docs/research.md`, `docs/design-partner-plan.md`, `docs/benchmark-targets.md`),
  asset-rendering and demo-recording scripts, and the tract/Burn benchmark workspace
  (`bench/baselines`). Benchmarks now compare BoundML with PyTorch and ONNX Runtime.

### Changed
- Benchmarks: one untimed warm-up launch per runtime; peak memory is the OS
  high-water mark on every platform (`ru_maxrss` via `wait4` on Linux and macOS);
  the uncommitted files at measurement time are recorded by name.

## [0.1.0] — first public release (early release)

### Added
- `boundml` CLI with `init`, `split`, `validate`, `plan`, `train` (with `--resume`
  and `--max-steps`), `evaluate`, `pack`, `inspect`, `verify`, `predict` (with
  `--preview`), `compare`, `serve`, `import`, `version`. `--json` output and
  documented exit codes for every command.
- Models: linear regression, logistic and softmax regression, and multilayer dense
  networks (ReLU/Tanh/Sigmoid). Tasks: regression, binary and multiclass.
- Native CPU computation in Rust: graph IR, kernels, reverse-mode autodiff, SGD
  (momentum, L2) and AdamW. No ML framework or BLAS dependency.
- Static tensor-memory planning (lifetime-based reuse) with budgets that are checked
  before any tensor storage is allocated. Allocation-free prediction and training
  steps after setup.
- Schema-checked CSV input mapped by column name, explicit missing-value and
  extra-column policies, preprocessing fitted on training data only, and bounded
  deterministic shuffling with exactly resumable checkpoints.
- Versioned model packages (`format_version` 1): schema, fitted preprocessing,
  graph, f32 safetensors weights, bounds, provenance, self-test vectors and SHA-256
  integrity. Strict load-time validation, with the memory plan recomputed by the
  runtime.
- Embeddable `Predictor` API (`boundml-artifact` crate) with no training or HTTP
  dependencies.
- Bounded local HTTP server: request, row, queue and worker limits, timeouts,
  overload responses and graceful shutdown.
- `compare`: an identical / compatible / breaking verdict for model updates,
  CI-friendly via exit code 6.
- PyTorch `nn.Sequential` and Keras `Sequential` dense-model import, verified against
  the source framework's outputs.

### Fixed before release (pre-launch audit)
- `compare` now reports a removed feature as breaking when the candidate rejects
  extra columns, and reports a numeric comparison that cannot run instead of
  aborting.
- Packages are limited to 1 GiB of planned inference tensor memory and 256 sessions,
  whether or not they declare a budget. The server's default workers and rows are
  capped.
- Inference graphs must be row-wise, so a prediction cannot depend on the other rows
  in its batch.
- The CSV record size limit is enforced while reading. `split` refuses to overwrite
  its input and writes atomically. `predict` writes its output atomically.
- The HTTP admission check now runs before the request body is read, and duplicate
  JSON keys are rejected.
- Resume also checks the validation file. JSON floats round-trip exactly.

### Fixed before release (source review)
- **Breaking:** inference fails closed with `BML-C007` when any output is not
  finite (CLI exit 3, HTTP 422, no partial output file). `verify` fails when a
  self-test output is non-finite, and packages with non-finite test vectors are
  rejected. `compare` treats non-finite outputs as breaking and rejects invalid
  thresholds (`BML-A009`).
- Temporary files are created exclusively under unique names next to the
  destination and committed by rename. A failed run never deletes or replaces an
  existing file it did not create, and `predict`/`split` refuse an output path that
  is the input.
- **Breaking:** the declared `bounds.sessions` is enforced per loaded package: a
  `Predictor` beyond the allowance fails with `BML-A010` (exit 4) before allocating.
  `Predictor` stages preprocessing directly into the planned arena. Training plans
  also budget the validation arena and batch staging buffers, and report the
  data-loader buffer.
- Package loading validates the graph, tensor declarations, consistency, test
  vectors and the declared weight size before reading the weight payload.
- Preprocessing checks exact input and output lengths. The server keeps a compute
  slot until a timed-out computation really finishes, replaces a predictor after a
  panic, and reports `usable_workers` in `/readyz` (503 when none is usable).
- **Breaking:** an invalid server configuration (`BML-S005`, e.g. more `--workers`
  than the package's sessions) exits with 3 (invalid input) instead of 1. The embed
  example releases its first predictor before starting two more, so it stays within
  the declared sessions.
- Benchmarks are gated on elementwise output parity across runtimes, keep every
  run's record, and compare training from identical initial weights and batch
  order. Earlier results are preserved under `docs/results/history/`.

### Known limitations
See [docs/release-status.md](docs/release-status.md): CPU only, single-threaded
kernels, only Windows x86_64 built and tested, and no external user evaluations yet.

[Unreleased]: ../../compare/v0.1.0...HEAD
[0.1.0]: ../../releases/tag/v0.1.0
