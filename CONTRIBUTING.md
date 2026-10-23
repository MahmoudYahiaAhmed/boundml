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
