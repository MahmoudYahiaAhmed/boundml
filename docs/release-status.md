# Release status — 0.1.0 (2026-10-23)

Legend:
- **Tested locally**: executed and passing on Windows 11 x64 (Ryzen 7 8700F) with
  Rust 1.94.0.
- **CI configured**: in `.github/workflows/ci.yml`; **no CI run has been observed**.
- **Unverified**: implemented but not executed in the stated setting.

## Capability status

| Capability | Implemented | Tested locally | CI configured | Notes |
|---|---|---|---|---|
| Graph IR, shape/broadcast rules, checked arithmetic | yes | yes (unit + property tests) | yes | |
| Kernels for all listed ops, stable losses | yes | yes (f64 reference, extreme logits) | yes | single-threaded scalar loops |
| Reverse-mode AD incl. fan-out | yes | yes (finite differences, PyTorch float64 fixture) | yes | |
| SGD (momentum, L2) / AdamW | yes | yes (independent reference + PyTorch fixture) | yes | |
| Deterministic seeding, exact resume | yes | yes (bitwise test) | yes | same binary + CPU only |
| CSV schema, policies, streaming shuffle | yes | yes | yes | |
| Train-only standardization, mean imputation | yes | yes | yes | |
| Memory planner (conservative + reuse), budget rejection before allocation | yes | yes (property test, budget tests, CLI exit 4) | yes | bound = planned tensor storage only |
| Allocation-free predict / training step / `predict_raw` | yes | yes (counting allocator) | yes | |
| Package format v1, strict load validation, self-tests | yes | yes (16 corruption/unsupported cases) | yes | hashes ≠ signatures |
| Atomic checkpoints with `.prev` recovery | yes | yes | yes | crash between renames handled by message, not automatic fallback |
| `compare` with verdicts | yes | yes (fixtures: identical / compatible / breaking) | yes | |
| CLI (all commands, `--json`, exit codes) | yes | yes (e2e test) | yes | |
| Local HTTP server limits (body, rows, queue, workers, timeout, shutdown) | yes | yes (server tests) | yes | slow-client protection = request timeout only; no auth/TLS |
| CLI ↔ SDK ↔ HTTP prediction parity | yes | yes (bitwise) | yes | |
| Embedding example (no training/HTTP deps) | yes | yes (manual run) | partially (built in CI) | |
| PyTorch import | yes | yes (`check_migration.py`, PyTorch 2.12.1) | yes (separate job) | dense Sequential only |
| Keras import | yes | yes (Keras 3.14 / TF 2.21) | yes (separate job) | dense Sequential only |
| Linux build/tests | source is portable | **no** | yes | unverified |
| macOS build/tests | source is portable | **no** | yes | unverified |
| Real-data demonstrations | yes | yes (2 datasets) | no | results in `docs/walkthroughs.md` |
| Benchmarks vs PyTorch / ONNX Runtime | yes | yes (5 runs) | no | one machine; native runtimes not compared |
| Sustained stress test | yes | yes (300 s, `bench/run_stress.py`) | no | one machine |
| Design-partner evaluation | plan only | **no** | — | no demand evidence |

## Acceptance matrix

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Fresh checkout builds with pinned toolchain and lockfile | passed locally (`cargo build --locked`); fresh-clone CI not observed | `Cargo.lock`, `rust-toolchain.toml` |
| 2 | Every op passes independent numerical tests | passed | `crates/boundml-core/tests/gradcheck.rs` |
| 3 | Gradients vs central differences and PyTorch | passed (atol 5e-4 + rtol 5e-3 vs FD; 1e-4 + 1e-3 vs PyTorch f64) | `gradcheck.rs`, `torch_reference.rs` |
| 4 | Fan-out / shared parameters accumulate | passed | `gradcheck.rs::fan_out_and_all_ops` |
| 5 | SGD/AdamW match references | passed (1e-6) | `optim_reference.rs`, `torch_reference.rs` |
| 6 | Training beats trivial baseline on held-out data | passed on both datasets; MLP ≤ linear on both (reported) | `docs/walkthroughs.md` |
| 7 | Interrupted + resumed = uninterrupted | passed (bitwise, 2 interruptions mid-epoch) | `boundml-train/tests/train.rs` |
| 8 | Preprocessing/predictions agree in-process, packaged, CLI, HTTP | passed (bitwise) | `boundml-cli/tests/e2e.rs` |
| 9 | Migrated models match source frameworks | passed (≤ 1.2e-6 on full test sets, 6 models) | `docs/results/migration.json` |
| 10 | Invalid schemas, batches, ops, corrupted/oversized artifacts fail cleanly | passed | `boundml-data/tests`, `boundml-artifact/tests/package.rs`, `server.rs` |
| 11 | Over-budget plans rejected before allocation, accurate accounting | passed | budget tests; `plan` report |
| 12 | Allocation-free core paths; reuse does not corrupt live tensors | passed | `alloc_free.rs` (core + artifact); planner property test; Conservative = Reuse bitwise in gradcheck |
| 13 | Sustained stress with duration and memory | passed: 300 s, 9,594 package reloads + self-tests, 9,594 train/checkpoint/resume cycles, 5.73 billion rows predicted by 4 concurrent sessions with zero result drift; working set 7.8 → 8.2 MiB (+0.8% from 2nd to 4th quarter), peak 8.5 MiB | `docs/results/stress.json` |
| 14 | New-user walkthrough with real commands | passed by the author on this machine; not tried by a new user | `docs/walkthroughs.md`, CI smoke job (not observed) |
| 15 | Compare reports breaking vs compatible on fixtures | passed | `package.rs::compare_classifies_changes`, e2e |
| 16 | Server overload, malformed, timeout, shutdown bounded | passed | `boundml-cli/tests/server.rs` |

## Checks run for this release (local)

`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` (clean),
`cargo test --workspace --locked` (61 tests, 0 failures), `cargo doc --workspace --no-deps` (no warnings), `cargo build --release`,
`python migrate/check_migration.py` (all passed), `python bench/run_bench.py --runs 5`,
`python bench/run_stress.py 300`.

Not run: `cargo audit` / `cargo deny` (not installed), fuzzing (malformed-input tests
cover the package parser instead), coverage measurement, and any non-Windows platform.

## Claims: what kind of evidence backs them

| Claim | Enforced by API | Established by analysis | Tested | Assumption |
|---|---|---|---|---|
| Planned tensor bytes ≤ budget | yes (checked before allocation) | planner intervals | yes | allocator/OS behaviour outside the bound |
| No aliasing of live buffers | runtime assertion | interval analysis | property test | — |
| No heap allocation in hot paths | — | by construction | counting allocator | other threads not counted |
| Package integrity | hash checks | — | corruption tests | hashes do not authenticate |
| Resume exactness | config/data hash checks | deterministic design | bitwise test | same binary and CPU |
| Server work bounded | semaphores, limits | design | server tests | slow clients are bounded by the request timeout only |

## Future (outside this release)

GPU/SIMD/threaded kernels, wasm/no_std targets, categorical features, trees, early
stopping and LR schedules, signed packages, an ONNX import subset, online
retraining loop (candidate C), and a sidecar contract for tract/RTen packages
(candidate B).
