# Release status — 0.1.0 (private beta)

**Label: private beta.** Complete for its scoped features; nothing published. Built,
tested and archive smoke-tested on Windows x86_64, Linux x86_64 and macOS arm64 in
GitHub Actions (latest: [run 40139782934](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40139782934) on branch
`readiness/private-beta`, commit `4cb8b6c`, all 10 jobs passed; first green run:
[40032964857](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40032964857)
on `main`, `fbad6db`), and locally on Windows 11 and WSL2 Ubuntu 24.04. Not
production-validated, and no outside users yet. Publication steps and owner
decisions: [publication-plan.md](publication-plan.md).

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

## Verified in CI (GitHub Actions, default runners)

| Job | ubuntu-latest | windows-latest | macos-latest (arm64) |
|---|---|---|---|
| `rust`: fmt, clippy `-D warnings`, `cargo test --workspace`, doc, walkthrough + embed example | pass | pass | pass |
| `package-smoke`: release archive built, extracted into a path with spaces, first-use steps | pass | pass | pass |
| `python`: wheel built, installed, 10 tests incl. bitwise parity with the CLI and DataFrames with non-numeric extra columns | pass | pass | pass |
| benchmark-harness checks (20 tests: parity gate, worker lifecycle; ResourceWarning as error) | pass | pass | pass |
| `migration`: PyTorch and Keras import checked against the source frameworks | pass | | |

The first run found two problems, both fixed before this green run: an overload 503
lost to a TCP reset on Windows (`c8297a0`, with a regression test), and a runner
toolchain conflict (`fbad6db`). The reviewer's Linux object-file error did not occur
on the default Linux runner.

## Installation paths verified from built artifacts (2026-10-25)

Fresh directories whose paths contain spaces, outside the repository, with fresh
virtual environments. Scripts and logs: `out/verify/` (local, not committed).

| Path | Platform | Result |
|---|---|---|
| CLI from the extracted archive: version, init, validate, train, evaluate, pack, verify, predict, budget failure (exit 4), schema error (exit 3, no output file), `--out` = `--data` refused, real-data training | Windows 11 (archive built from `4cb8b6c`); WSL2 Ubuntu 24.04 (archive built from `b35e9ff`) | all 24 checks pass; documented numbers reproduced (RMSE 0.1877, first prediction 4.5001, accuracy 0.9597) |
| Rust: README snippet in a new Cargo project (path dependency and private git dependency); a third predictor with `deploy.sessions = 2` refused (BML-A010); `examples/embed` | Windows 11 | pass; `power = 4.821724` |
| Python wheel in a clean venv: metadata and tags, README examples, predictions bitwise equal to the CLI's for dict rows and DataFrames, Python-trained package equal through the CLI | Windows 11, Python 3.11, 3.12, 3.13, 3.14; WSL2 Ubuntu, Python 3.12 | pass |

## Not verified — depends on things outside this machine

| Item | Depends on |
|---|---|
| The reviewer's Linux object-file error | a reproduction in the reviewer's environment (filesystem type, container runtime, limits); not reproduced locally under disk or memory exhaustion |
| Tagged release workflow (`release.yml`) and published archives | pushing a `v*` tag; the same build and smoke test already pass in `ci.yml` |
| PyPI wheels | a distribution name: **`boundml` on PyPI belongs to an unrelated project**, so `pip install boundml` installs something else; then publishing. Wheels are built and tested in CI but not uploaded |
| Linux wheels for glibc < 2.34 (older than Ubuntu 22.04) | a manylinux2014 container build; current wheels are `manylinux_2_34` |
| Python 3.9 and 3.10 | running the tests on them; the abi3 wheel claims 3.9+ but only 3.11–3.14 were run |
| Conduct and security contacts | the owner supplying a private conduct contact; vulnerability reporting needs a public repository |
| crates.io packages (`cargo install boundml-cli`) | publishing in order core → data → artifact → train → cli; dependent dry-runs fail until their dependencies are on crates.io |
| Private vulnerability reporting route | making the repository public, then enabling it in repository settings |
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
