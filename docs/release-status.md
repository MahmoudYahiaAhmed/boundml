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
| Benchmarks incl. native runtime (tract) and Burn, parity-gated, controlled training | `docs/benchmarks.md`, raw `docs/results/bench.json`; driver checks in `bench/test_run_bench.py`; earlier records in `docs/results/history/` |
| 5-minute stress test (4 concurrent sessions within the declared allowance) | `docs/results/stress.json` |
| Release archive smoke test (extracted archive, path with spaces) | `scripts/release.py` → `dist/smoke-x86_64-pc-windows-msvc.json`, copied to `docs/results/` after the build (an archive cannot contain the record of its own checksum) |
| Newcomer rehearsal from a fresh clone | `launch/readiness-report.md` |

Local checks: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test
--workspace --locked`, `cargo doc` with `-D warnings`, `cargo package` for
`boundml-core`.

Also checked on Linux, **not** in CI: WSL2 Ubuntu 24.04 on the same machine (gcc 13.3,
binutils 2.42, Rust 1.94.0, no compiler or linker overrides). `cargo test --workspace
--locked` passed 100 of 100 tests, on both ext4 and the Windows-mounted filesystem;
`cargo build --release`, `clippy -D warnings` and `fmt --check` passed. An external
reviewer saw a zero-length object-file error from the default Linux build, which went
away with compiler overrides. That error was **not reproduced** here, so its cause is
still unknown. The usual causes are a full disk or temporary directory, an
interrupted build that left a truncated object, or a filesystem- or linker-specific
problem. Their exact environment is needed to close it.

## Not verified — depends on things outside this machine

| Item | Depends on |
|---|---|
| Linux and macOS builds, tests and archives in CI (the default GitHub runners) | first observed run of `ci.yml` / `release.yml` on GitHub; the WSL run above does not stand in for it |
| The reviewer's Linux object-file error | the reviewer's environment details; not reproduced locally |
| CI status of any kind | adding a remote and pushing branch `main` (the branch `ci.yml` triggers on); **no workflow has ever run** |
| crates.io packages (`cargo install boundml-cli`) | publishing in order core → data → artifact → train → cli; dependent dry-runs fail until their dependencies are on crates.io |
| Private vulnerability reporting route | enabling it in repository settings |
| Demand / fit for real workloads | external design-partner evaluations (`docs/design-partner-plan.md`) |
| Fuzzing, `cargo audit`/`cargo deny` | not run |

## Known limitations

CPU only, with single-threaded scalar kernels. tract has higher large-batch
throughput on the wider model. Dense models on numeric features only. `boundml train`
re-parses the CSV every epoch. Checksums do not authenticate. The HTTP server has no
auth or TLS. The memory bound covers planned tensor storage, not the whole process.

## Compatibility expectations

Package format v1 is read by every 0.1.x runtime. A new format requires a version
bump, and older runtimes refuse it with `BML-A003`. Within 0.x, CLI flags and the
Rust API may change in minor releases, listed under **Breaking** in the changelog.
Details: [package-format.md](package-format.md#compatibility-and-version-policy).
