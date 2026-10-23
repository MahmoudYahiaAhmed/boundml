# Release status — 0.1.0 (early release)

**Label: early release (Windows x86_64 beta).** Complete for its scoped features. The
release archive is built and smoke-tested for Windows x86_64 only. Not
production-validated, and no external users yet.

## Verified locally (Windows 11 x64, Ryzen 7 8700F, Rust 1.94.0)

| Area | Evidence |
|---|---|
| Gradients, losses, optimizers | f64 finite-difference reference and a PyTorch float64 fixture (`crates/boundml-core/tests/`) |
| Fan-out / shared parameters, both planners agree | `gradcheck.rs` |
| No heap allocation in prediction and training steps | counting-allocator tests (core and `Predictor`) |
| Budgets rejected before allocation; 1 GiB package ceiling | planner, package and CLI tests (exit code 4/5) |
| Row-wise inference graphs only | graph and package tests |
| Exact resume after interruption | bitwise test, 2 mid-epoch interruptions |
| CSV policies, bounded record size, safe split | `boundml-data/tests/data.rs` |
| Package corruption/version/size/consistency rejection | 16 malformed-package cases + audit repros |
| `compare` verdicts | identical / compatible / breaking fixtures, incl. schema-policy cases |
| CLI ↔ SDK ↔ HTTP parity | bitwise, `boundml-cli/tests/e2e.rs` |
| Server limits | overload, early reject before body read, timeout, malformed and duplicate keys, shutdown |
| Source-review fixes (non-finite outputs, temp files, session allowance, metadata preflight, buffer lengths, server recovery) | `review_*.rs` tests in core, artifact and CLI; each failed on `e567a00` and passes now |
| PyTorch 2.12 / Keras 3.14 import | `migrate/check_migration.py` → `docs/results/migration.json` |
| Real-data demonstrations | 2 UCI datasets, `docs/walkthroughs.md` |
| Benchmarks vs PyTorch and ONNX Runtime, parity-gated, controlled training | `docs/benchmarks.md`, raw `docs/results/bench.json`; driver checks in `bench/test_run_bench.py`; earlier records in `docs/results/history/` |
| 5-minute stress test (4 concurrent sessions within the declared allowance) | `docs/results/stress.json` |
| Release archive smoke test (extracted archive, path with spaces) | `scripts/release.py` → `dist/smoke-x86_64-pc-windows-msvc.json`, copied to `docs/results/` after the build (an archive cannot contain the record of its own checksum) |
| Newcomer rehearsal (source copy and archive, paths with spaces) | independent docs-only run; every gap found was fixed or documented |

Local checks: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test
--workspace --locked`, `cargo doc` with `-D warnings`, `cargo package` for
`boundml-core`.

Also checked on Linux, **not** in CI: WSL2 Ubuntu 24.04 on the same machine (gcc 13.3,
binutils 2.42, Rust 1.94.0, no compiler or linker overrides). `cargo test --workspace
--locked` passed 100 of 100 tests, on both ext4 and the Windows-mounted filesystem;
`cargo build --release`, `clippy -D warnings` and `fmt --check` passed. An external
reviewer's default Linux build failed while archiving `boundml-train` with `failed
to map object file: memory map must have a non-zero length`. It passed with
`CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_TEST_CODEGEN_UNITS=1` and
`CARGO_PROFILE_TEST_DEBUG=0`. That error was **not reproduced** here. These cases were
tested on the unchanged tree:

- target directory on a 1 GiB filesystem: fails with `No space left on device`;
- rebuilding after enlarging it, with and without incremental compilation: 101/101;
- default build under a 2 GiB memory cap: 101/101;
- default build under a 768 MiB cap: killed by the cap;
- rebuilding in the same target directory after that: 101/101.

A full disk and a killed build both left zero-length objects behind, but each next
build replaced them. The project has no build scripts or codegen settings involved.
The most likely cause is the reviewer's environment, for example a sandboxed
filesystem where a newly written object's size is not yet visible when rustc maps
it, which fewer parallel jobs and codegen units would make rarer. That is not
confirmed. Closing it needs a reproduction in that environment.

## Not verified — depends on things outside this machine

| Item | Depends on |
|---|---|
| Linux and macOS builds, tests and archives in CI (the default GitHub runners) | first observed run of `ci.yml` / `release.yml` on GitHub; the WSL run above does not stand in for it |
| The reviewer's Linux object-file error | a reproduction in the reviewer's environment (filesystem type, container runtime, limits); not reproduced locally under disk or memory exhaustion |
| CI status of any kind | adding a remote and pushing branch `main` (the branch `ci.yml` triggers on); **no workflow has ever run** |
| crates.io packages (`cargo install boundml-cli`) | publishing in order core → data → artifact → train → cli; dependent dry-runs fail until their dependencies are on crates.io |
| Private vulnerability reporting route | enabling it in repository settings |
| Demand / fit for real workloads | 3–5 outside developers deploying their own models |
| Fuzzing, `cargo audit`/`cargo deny` | not run |

## Known limitations

CPU only, with single-threaded scalar kernels. ONNX Runtime has higher large-batch
throughput. Dense models on numeric features only. `boundml train`
re-parses the CSV every epoch. Checksums do not authenticate. The HTTP server has no
auth or TLS. The memory bound covers planned tensor storage, not the whole process.

## Compatibility expectations

Package format v1 is read by every 0.1.x runtime. A new format requires a version
bump, and older runtimes refuse it with `BML-A003`. Within 0.x, CLI flags and the
Rust API may change in minor releases, listed under **Breaking** in the changelog.
Details: [package-format.md](package-format.md#compatibility-and-version-policy).
