# Contributing

## Toolchain

Rust 1.94.0 is pinned in `rust-toolchain.toml`, and dependencies are locked in
`Cargo.lock`. Python is needed only for the reference fixtures, migration and
benchmarks: 3.13 with `torch`, `keras`/`tensorflow`, `onnxruntime`, `safetensors`,
`numpy` and `psutil`.

## Checks to run before a pull request

```bash
cargo fmt --all --check
```

```bash
cargo clippy --workspace --all-targets --locked -- -D warnings
```

```bash
cargo test --workspace --locked
```

```bash
cargo doc --workspace --no-deps --locked
```

If you change kernels, autodiff or optimizers, regenerate and commit the PyTorch
fixture:

```bash
python scripts/torch_reference.py
```

If you change the import path:

```bash
python migrate/check_migration.py
```

## Release and presentation tooling

| Task | Command |
|---|---|
| Build, package, checksum and smoke-test a release archive | `python scripts/release.py --out dist` |
| Re-record the README demo from the real binary | `python scripts/record_demo.py` |
| Re-render banner, diagram and social preview (needs Chrome/Edge and Pillow) | `python scripts/render_assets.py` |
| Benchmarks (see docs/benchmarks.md for prerequisites) | `python bench/run_bench.py --runs 5` then `python bench/report.py` |
| Stress test | `python bench/run_stress.py 300` |

Releases are cut by pushing a `v*` tag. `release.yml` builds and smoke-tests the
archives and creates a **draft** GitHub release for a maintainer to review. crates.io
publication is the separate, manual `publish-crates.yml` workflow.

## Rules

- Each new operator needs forward and backward kernels, an independent f64 reference
  in `crates/boundml-core/tests/reference/`, and a finite-difference test. Kernels must
  not allocate and must report any scratch they need to the planner.
- A change to package fields requires a `format_version` bump and an update to
  `docs/package-format.md`.
- Each new error needs a stable `BML-` code, documented in `docs/errors.md`.
- `unsafe` is confined to `crates/boundml-core/src/arena.rs`.
- Do not report a performance or compatibility claim without a reproducible command
  and logged results under `docs/results/`.
