# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may contain
breaking changes; they are called out under **Breaking**.

## [Unreleased]

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

### Known limitations
See [docs/release-status.md](docs/release-status.md): CPU only, single-threaded
kernels, only Windows x86_64 built and tested, and no external user evaluations yet.

[Unreleased]: ../../compare/v0.1.0...HEAD
[0.1.0]: ../../releases/tag/v0.1.0
