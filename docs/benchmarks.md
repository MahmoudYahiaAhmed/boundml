# Benchmarks

Every number on this page comes from `docs/results/bench.json`, produced by the
commands below, and the tables are generated from it by `bench/report.py`. The
original targets were committed before the first measurement
([benchmark-targets.md](benchmark-targets.md)). The native-runtime (tract) and Burn
comparisons were added after the pre-launch audit and have no pre-declared
threshold.

## Reproduce

```bash
cargo build --release --workspace
cargo build --release --manifest-path bench/baselines/Cargo.toml
python -I datasets/prepare.py --download
boundml split datasets/breast_cancer/all.csv --out datasets/breast_cancer --valid 0.15 --test 0.2 --seed 2026
boundml split datasets/wine_red/all.csv --out datasets/wine_red --valid 0.15 --test 0.2 --seed 2026
boundml train examples/breast_cancer/mlp.toml --checkpoint out/breast_cancer/mlp/ckpt --quiet
boundml train examples/wine_quality/mlp.toml --checkpoint out/wine_quality/mlp/ckpt --quiet
python bench/run_bench.py --runs 5
python bench/report.py
```

(The splits and `prepare.py` are only needed if `datasets/` was removed; the split
CSVs are in the repository.) Python baselines need `torch`, `onnxruntime`,
`safetensors`, `numpy` and `psutil`.

## Setup

| | |
|---|---|
| Hardware | AMD Ryzen 7 8700F, 8 cores / 16 threads, 32 GB RAM; CPU only; the single machine available |
| OS | Windows 11 Pro 10.0.26200 |
| BoundML | rustc 1.94.0, `--release`, `lto = "thin"`, `codegen-units = 1`, default target CPU (no `-C target-cpu=native`), single-threaded kernels |
| tract | `tract-onnx` 0.23.9, in-process Rust, plans optimized for concrete shapes (batch 1 and batch 256), one thread |
| Burn | 0.21.0, ndarray backend + autodiff |
| ONNX Runtime | 1.20.1, Python API, CPUExecutionProvider, default session options (multi-threaded) |
| PyTorch | 2.12.1+cpu eager, `inference_mode`, default 8 threads (training also measured with 1) |
| Python | 3.13.9 |
| Models | breast cancer: MLP 30 → 16 → 1 (ReLU, sigmoid output), 529 parameters; wine quality: MLP 11 → 32 → 16 → 1 (ReLU), 929 parameters; f32 |
| Data | UCI test splits (124 and 328 rows); a fixed 256-row batch for throughput is built by repeating test rows |

All runtimes execute **the same weights**. The ONNX and PyTorch models are rebuilt
from the BoundML package (`bench/pymodel.py`), and output checksums agree to 4+
decimal places.

## What each column measures

- **Process start → first prediction**: wall time from `Popen` until the worker
  prints `FIRST`. This includes OS process creation, interpreter start-up and imports
  (Python), model loading, reading the test CSV, and one prediction.
- **Framework import** (Python only): `import torch` / `import onnxruntime`, measured
  inside the worker. Interpreter start-up alone is reported once, above the tables.
- **Model load**: in-process time to load and prepare the model. For BoundML that
  covers integrity hashes, strict validation, memory-plan recomputation, the arena
  allocation and the bundled self-test. For tract it is ONNX parsing plus optimizing
  two plans; for Python, building the module or session.
- **Preprocessing alone**: standardizing one raw row with the fitted statistics
  (BoundML `Preprocessor::apply`; NumPy for Python; equivalent Rust glue for tract).
- **Core inference**: pre-standardized input → output.
- **Preprocess + infer**: raw values → output.
- **Peak process working set**: whole-process peak (`psutil` `peak_wset`), read after
  all measurements. It includes the runtime, the loaded CSV and everything else.
- **Planned tensor bytes** (BoundML only): the planner's total for one session at the
  package's `max_batch` (weights + arena). This is the quantity the budget bounds. It
  is a small part of the whole process.
- Training time covers the training loop only. `boundml train` streams and parses the
  CSV every epoch and evaluates the validation file after each epoch; the in-memory
  rows (BoundML core, Burn, PyTorch) hold standardized tensors in memory and skip
  validation.

## Results

<!-- BENCH:BEGIN -->
Measured 2026-10-23 on AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 logical CPUs), Windows-11-10.0.26200-SP0; 5 runs per measurement, medians shown. Python interpreter start-up (`python -I -c pass`): **47.9 ms**.

### Breast cancer (MLP 30-16-1, sigmoid output)

**Start-up and memory**

| Runtime | Process start → first prediction | Framework import | Model load | Peak process working set | Planned tensor bytes |
|---|---|---|---|---|---|
| BoundML | 9.1 ms | — | 0.6 ms | 6.3 MiB | 49,156 B |
| tract (native Rust) | 11.2 ms | — | 2.6 ms | 9.5 MiB | not reported |
| PyTorch eager (Python) | 1.49 s | 1.44 s | 3.5 ms | 202.4 MiB | not reported |
| ONNX Runtime (Python API) | 211.9 ms | 150.2 ms | 9.3 ms | 51.4 MiB | not reported |

**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)

| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |
|---|---|---|---|---|---|---|
| BoundML | 1 | 0.0 / 0.1 / 0.1 | 0.3 / 0.3 / 0.4 | 0.3 / 0.4 / 0.4 | 8.85 M | 7.06 M |
| tract (native Rust) | 1 | 0.1 / 0.1 / 0.1 | 1.7 / 1.9 / 2.0 | 1.7 / 1.8 / 2.5 | 28.82 M | 15.60 M |
| PyTorch eager (Python) | 8 | 1.8 / 1.8 / 1.9 | 23.5 / 27.0 / 47.3 | 27.5 / 32.0 / 51.6 | 6.04 M | 4.12 M |
| ONNX Runtime (Python API) | ort-default | 2.6 / 3.4 / 3.6 | 7.9 / 8.2 / 11.9 | 9.9 / 11.0 / 15.7 | 17.57 M | 10.02 M |

**Training** (same architecture, optimizer settings, batch size and epochs; test split)

| Trainer | Train time | Test metric | Peak process working set |
|---|---|---|---|
| `boundml train` (CSV streamed every epoch + validation) | 387.2 ms | accuracy 0.9597 | 7.5 MiB |
| BoundML core, in-memory | 9.1 ms | accuracy 0.9597 | 5.7 MiB |
| Burn 0.21 ndarray | 48.6 ms | accuracy 0.9597 | 6.9 MiB |
| PyTorch, 8 threads | 463.6 ms | accuracy 0.9597 | 273.4 MiB |
| PyTorch, 1 thread | 369.9 ms | accuracy 0.9597 | 273.5 MiB |

Sizes: `boundml` binary 4.0 MB; package 28.0 KB (includes schema, preprocessing, self-tests); ONNX file 2.5 KB; installed PyTorch 516 MB, ONNX Runtime 37 MB.

### Wine quality (MLP 11-32-16-1, regression)

**Start-up and memory**

| Runtime | Process start → first prediction | Framework import | Model load | Peak process working set | Planned tensor bytes |
|---|---|---|---|---|---|
| BoundML | 8.8 ms | — | 0.5 ms | 6.2 MiB | 69,252 B |
| tract (native Rust) | 12.3 ms | — | 3.2 ms | 9.8 MiB | not reported |
| PyTorch eager (Python) | 1.50 s | 1.44 s | 3.5 ms | 202.5 MiB | not reported |
| ONNX Runtime (Python API) | 192.5 ms | 137.8 ms | 8.9 ms | 51.3 MiB | not reported |

**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)

| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |
|---|---|---|---|---|---|---|
| BoundML | 1 | 0.0 / 0.1 / 0.1 | 0.4 / 0.4 / 0.5 | 0.4 / 0.4 / 0.4 | 6.56 M | 6.14 M |
| tract (native Rust) | 1 | 0.1 / 0.1 / 0.1 | 3.1 / 3.3 / 4.3 | 3.1 / 3.2 / 3.3 | 16.41 M | 14.25 M |
| PyTorch eager (Python) | 8 | 1.8 / 1.9 / 1.9 | 28.0 / 31.5 / 42.9 | 31.8 / 37.2 / 53.8 | 4.89 M | 4.01 M |
| ONNX Runtime (Python API) | ort-default | 1.8 / 1.8 / 1.9 | 8.1 / 8.4 / 12.0 | 10.1 / 10.3 / 12.9 | 13.11 M | 8.91 M |

**Training** (same architecture, optimizer settings, batch size and epochs; test split)

| Trainer | Train time | Test metric | Peak process working set |
|---|---|---|---|
| `boundml train` (CSV streamed every epoch + validation) | 221.9 ms | rmse 0.6635 | 7.3 MiB |
| BoundML core, in-memory | 28.4 ms | rmse 0.6592 | 5.8 MiB |
| Burn 0.21 ndarray | 83.9 ms | rmse 0.6225 | 6.9 MiB |
| PyTorch, 8 threads | 805.9 ms | rmse 0.6616 | 274.0 MiB |
| PyTorch, 1 thread | 660.0 ms | rmse 0.6616 | 273.5 MiB |

Sizes: `boundml` binary 4.0 MB; package 17.5 KB (includes schema, preprocessing, self-tests); ONNX file 4.4 KB; installed PyTorch 516 MB, ONNX Runtime 37 MB.

<!-- BENCH:END -->

## Reading the results

- **Batch throughput: tract and ONNX Runtime win.** On a 256-row batch (model only),
  tract is about 3.3× faster than BoundML on the 30-feature model and 2.5× faster on
  the 11-feature model. ONNX Runtime is about 2.0× faster on both. With preprocessing
  included, tract stays about 2.2–2.3× ahead. Pre-declared target T3 is met as
  originally defined (raw input including preprocessing), but the model-only ratio
  sits right at the 50% threshold. BoundML uses straightforward single-threaded loops. It
  does not use packed GEMM, explicit SIMD, `target-cpu=native` or threads. If you
  score large batches and throughput dominates, use tract or ONNX Runtime.
- **Single-row calls, start-up and memory: BoundML is ahead** on these models. One
  row through preprocessing and the model takes under half a microsecond, with no
  allocation, and the whole process stays under ~10 MiB. tract's per-call overhead
  (tensor creation and plan state) dominates at one row. The Python runtimes pay
  interpreter and framework costs.
- **Training: BoundML's core is fast on these tiny models, but `boundml train` is
  not.** Most of the CLI's time goes to re-parsing the CSV every epoch and running
  validation, a design choice that bounds memory for large files. Caching small
  datasets in memory is on the [roadmap](../ROADMAP.md).
- **Quality** is the same across trainers for breast cancer. On wine, Burn reached a
  lower test RMSE than BoundML and PyTorch with the same hyperparameters, plausibly
  because of its different default weight initialization and weight decay on biases
  (see `bench/baselines/src/bin/train_burn.rs`). All are within the run-to-run
  spread of small-data MLPs. A linear model beats every MLP on this data
  ([walkthroughs](walkthroughs.md)).

### Why choose BoundML when another runtime is faster

Pick BoundML when the costs it removes matter more than batch throughput: one Rust
binary with no Python or ONNX toolchain; preprocessing and the feature schema shipped
inside the model package and enforced on every call; a memory plan checked against a
declared budget before allocation; self-tests after every copy or upgrade; and a
CI-readable `compare` before each model update.

### When another tool is the better fit

- High-throughput batch scoring, or models beyond small MLPs: **tract**, **ONNX
  Runtime**.
- Training larger networks, GPUs, or many layer types: **Burn**, **Candle**,
  **PyTorch**.
- Trees or classical ML on tabular data: **Linfa**, **SmartCore**, **Aprender**, or
  XGBoost/LightGBM via their runtimes.
- PyTorch models on mobile with ahead-of-time memory planning: **ExecuTorch**.

## Pre-declared targets, re-evaluated

The thresholds are unchanged from [benchmark-targets.md](benchmark-targets.md),
evaluated per dataset with the measurement the target names.

<!-- TARGETS:BEGIN -->
| ID | Dataset | Threshold | Measured | Verdict |
|---|---|---|---|---|
| T1 memory | Breast cancer | ≤ 20% of PyTorch peak | 3.1% | **met** |
| T2 cold start | Breast cancer | ≤ 10% of PyTorch | 0.6% | **met** |
| T3 throughput | Breast cancer | ≥ 50% of ORT batch-256, raw input incl. preprocessing (as originally measured) | 70% (model only: 50%) | **met** |
| T4 training | Breast cancer | `boundml train` ≤ 2× PyTorch 1-thread | 1.05× | **met** |
| T5 quality | Breast cancer | within 2 pp of PyTorch | +0.0 pp accuracy | **met** |
| T1 memory | Wine quality | ≤ 20% of PyTorch peak | 3.1% | **met** |
| T2 cold start | Wine quality | ≤ 10% of PyTorch | 0.6% | **met** |
| T3 throughput | Wine quality | ≥ 50% of ORT batch-256, raw input incl. preprocessing (as originally measured) | 69% (model only: 50%) | **met** |
| T4 training | Wine quality | `boundml train` ≤ 2× PyTorch 1-thread | 0.34× | **met** |
| T5 quality | Wine quality | within 10% RMSE of PyTorch | RMSE 0.6635 vs 0.6616 (+0.3%) | **met** |
<!-- TARGETS:END -->

## Limits of this evidence

- One Windows machine. No Linux, macOS or ARM measurements.
- Tiny models: the results say nothing about wide or deep networks, where optimized
  kernels dominate.
- The Python baselines include interpreter overhead per call. They represent a Python
  deployment, not the kernels' speed.
- The ONNX Runtime C API, RTen and ExecuTorch were not measured.
- These results do not show that anyone wants this workflow. That needs the
  evaluations in [design-partner-plan.md](design-partner-plan.md).
