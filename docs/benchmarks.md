# Benchmarks

Every number on this page comes from `docs/results/bench.json`, produced by the
commands below, and the tables are generated from it by `bench/report.py`. The
original targets were committed before the first measurement
([benchmark-targets.md](benchmark-targets.md)). The native-runtime (tract) and Burn
comparisons were added after the pre-launch audit and have no pre-declared
threshold.

The records from the previous measurement (commit `e567a00`) are kept unchanged in
[results/history/2026-10-23-e567a00](results/history/2026-10-23-e567a00/README.md),
together with the method limitations that led to this re-measurement.

## Reproduce

```bash
cargo build --release --workspace
cargo build --release --manifest-path bench/baselines/Cargo.toml
python -I datasets/prepare.py --download
boundml split datasets/breast_cancer/all.csv --out datasets/breast_cancer --valid 0.15 --test 0.2 --seed 2026
boundml split datasets/wine_red/all.csv --out datasets/wine_red --valid 0.15 --test 0.2 --seed 2026
boundml train examples/breast_cancer/mlp.toml --checkpoint out/breast_cancer/mlp/ckpt --quiet
boundml train examples/wine_quality/mlp.toml --checkpoint out/wine_quality/mlp/ckpt --quiet
python -m unittest bench/test_run_bench.py
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
from the BoundML package (`bench/pymodel.py`); `bench.json` records the SHA-256 of
the package manifest, weights, ONNX file and test CSV.

**Parity gate.** Before any timing, every worker writes its outputs for every test
row at batch sizes 1, 7 and 64, starting from raw values. The driver checks each
output against BoundML at batch 1: it must be finite and within `1e-5 + 1e-4 × |ref|`.
Any mismatch, missing output, non-zero worker exit or step timeout (300 s) aborts the
whole benchmark without writing results. The largest error seen is reported per
runtime. `bench/test_run_bench.py` checks that these failures abort.

**Runs.** Each run starts a fresh worker process; runtimes are interleaved within a
run. Every run's command, exit status, timings, memory and parity error is kept in
`bench.json`; tables show the median, and ranges show the minimum and maximum over runs.
Every timed Rust result goes through `std::hint::black_box`.

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
- **Core inference** (model only): pre-standardized f32 input → output. Copies that the
  runtime's API forces are part of the call and listed under each table: BoundML's
  `predict_into` copies the input into its planned arena, and tract's
  `Tensor::from_shape` copies it into an owned tensor. Python inputs are pre-built
  float32 arrays, so no dtype conversion runs inside the timed call.
- **Preprocess + infer**: raw values → output.
- **Peak process memory**: on Windows, the OS-reported peak working set
  (`psutil` `peak_wset`, a high-water mark) read after all measurements. On other
  platforms `psutil` has no peak counter, so the driver records the resident set size
  (inference: read once after the measurements; training: the maximum of samples
  taken every 5 ms), which can miss short peaks. Each record names its metric. Both
  include the runtime, the loaded CSV and everything else.
- **Planned tensor bytes** (BoundML only): the planner's total for one session at the
  package's `max_batch` (weights + arena). This is the quantity the budget bounds. It
  is a small part of the whole process.
- **Training** time covers the training loop only. Two kinds of rows are kept apart:
  - *CLI training* (`boundml train`): the shipped end-to-end path. It streams and
    parses the CSV every epoch through a bounded shuffle buffer and evaluates the
    validation file after each epoch. Its batch order differs from the other rows.
  - *Controlled in-memory training*: `train_core` (BoundML core) writes a control file
    with the standardized train/test tensors, the initial weights (after output-bias
    initialization) and every epoch's permutation, then trains from it. PyTorch and
    Burn load the same file. All three use the same architecture, loss, AdamW
    hyperparameters, batch size and epochs. Remaining differences: optimizer
    implementations and floating-point order, and **Burn applies weight decay to
    biases** (its AdamW has no per-parameter groups; BoundML and PyTorch decay
    weights only).

## Results

<!-- BENCH:BEGIN -->
Measured 2026-10-23 on AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 logical CPUs), Windows-11-10.0.26200-SP0; 5 runs per measurement; medians shown, with the min–max over runs in parentheses where given. Commit `925fb83` (with uncommitted changes). Python interpreter start-up (`python -I -c pass`): **40.8 ms**.

Timing ran only after every runtime's outputs matched BoundML's for every test row at batch sizes 1, 7, 64 (every output finite and |a - b| <= atol + rtol * |b|, atol 1e-05, rtol 0.0001). Memory is `peak_wset`: OS-reported peak working set (high-water mark), read after all measurements.

### Breast cancer (MLP 30-16-1, sigmoid output)

**Start-up and memory**

| Runtime | Process start → first prediction | Framework import | Model load | Peak process memory | Planned tensor bytes | Max parity error |
|---|---|---|---|---|---|---|
| BoundML | 11.4 ms (9.0 ms–53.7 ms) | — | 0.6 ms | 6.3 MiB (6.3 MiB–6.3 MiB) | 49,156 B | 0.0e+00 |
| tract (native Rust) | 13.4 ms (11.7 ms–237.7 ms) | — | 3.0 ms | 10.1 MiB (10.1 MiB–10.1 MiB) | not reported | 4.8e-07 |
| PyTorch eager (Python) | 1.57 s (1.47 s–1.66 s) | 1.52 s | 3.8 ms | 203.0 MiB (202.4 MiB–203.4 MiB) | not reported | 1.5e-07 |
| ONNX Runtime (Python API) | 213.9 ms (191.7 ms–263.4 ms) | 159.8 ms | 9.8 ms | 51.5 MiB (51.4 MiB–51.6 MiB) | not reported | 1.2e-07 |

**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)

| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |
|---|---|---|---|---|---|---|
| BoundML | 1 | 0.1 / 0.1 / 0.1 | 0.3 / 0.4 / 0.4 | 0.4 / 0.4 / 0.5 | 8.72 M (8.52 M–9.04 M) | 6.45 M (5.92 M–6.82 M) |
| tract (native Rust) | 1 | 0.1 / 0.1 / 0.1 | 1.8 / 1.9 / 2.6 | 1.8 / 1.9 / 2.8 | 27.51 M (25.96 M–28.37 M) | 13.76 M (13.71 M–14.94 M) |
| PyTorch eager (Python) | 8 | 3.7 / 3.8 / 4.3 | 31.8 / 36.8 / 46.0 | 27.8 / 34.0 / 52.3 | 6.87 M (6.07 M–7.21 M) | 4.02 M (3.83 M–4.22 M) |
| ONNX Runtime (Python API) | ort-default | 2.4 / 3.9 / 4.8 | 7.5 / 7.6 / 10.6 | 10.2 / 15.1 / 19.3 | 20.62 M (19.70 M–20.89 M) | 9.06 M (8.44 M–9.28 M) |

Input copies inside the timed calls: BoundML: model only: predict_into copies the f32 input into the planned arena; tract (native Rust): every call: Tensor::from_shape copies the f32 input into an owned tract tensor; PyTorch eager (Python): model only: none in user code (float32 inputs pre-built); ONNX Runtime (Python API): model only: none in user code (float32 inputs pre-built).

**Training** (test split). The controlled rows load one control file written by BoundML's in-memory trainer: the same standardized data, initial weights and per-epoch batch order. The CLI row is a separate end-to-end measurement with its own streamed batch order.

| Trainer | Train time (min–max) | Test metric (all runs) | Peak process memory |
|---|---|---|---|
| `boundml train` CLI (own init and streamed batch order; CSV re-read every epoch + validation) | 402.0 ms (391.3 ms–444.8 ms) | accuracy 0.9597 | 7.5 MiB |
| BoundML core, in-memory (writes the control file) | 9.1 ms (8.9 ms–9.3 ms) | accuracy 0.9597 | 8.3 MiB |
| Burn 0.21 ndarray, controlled (decays biases too) | 45.0 ms (44.8 ms–48.1 ms) | accuracy 0.9597 | 9.1 MiB |
| PyTorch, default threads, controlled | 428.3 ms (418.5 ms–575.9 ms) | accuracy 0.9597 | 274.3 MiB |
| PyTorch, 1 thread, controlled | 359.2 ms (351.7 ms–423.0 ms) | accuracy 0.9597 | 273.8 MiB |

Sizes: `boundml` binary 4.1 MB; package 28.0 KB (includes schema, preprocessing, self-tests); ONNX file 2.5 KB; installed PyTorch 516 MB, ONNX Runtime 37 MB.

### Wine quality (MLP 11-32-16-1, regression)

**Start-up and memory**

| Runtime | Process start → first prediction | Framework import | Model load | Peak process memory | Planned tensor bytes | Max parity error |
|---|---|---|---|---|---|---|
| BoundML | 12.5 ms (10.3 ms–13.1 ms) | — | 0.5 ms | 6.3 MiB (6.3 MiB–6.3 MiB) | 69,252 B | 0.0e+00 |
| tract (native Rust) | 15.9 ms (13.4 ms–17.9 ms) | — | 3.5 ms | 10.4 MiB (10.3 MiB–10.4 MiB) | not reported | 9.5e-07 |
| PyTorch eager (Python) | 1.60 s (1.50 s–1.73 s) | 1.53 s | 3.9 ms | 203.0 MiB (202.7 MiB–203.2 MiB) | not reported | 9.5e-07 |
| ONNX Runtime (Python API) | 239.7 ms (209.5 ms–294.9 ms) | 166.8 ms | 10.8 ms | 51.5 MiB (51.5 MiB–51.9 MiB) | not reported | 9.5e-07 |

**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)

| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |
|---|---|---|---|---|---|---|
| BoundML | 1 | 0.0 / 0.1 / 0.1 | 0.4 / 0.4 / 0.5 | 0.4 / 0.5 / 0.5 | 6.35 M (6.04 M–6.63 M) | 5.37 M (5.13 M–6.06 M) |
| tract (native Rust) | 1 | 0.1 / 0.1 / 0.1 | 3.2 / 3.3 / 5.0 | 3.2 / 3.3 / 3.4 | 15.74 M (15.42 M–16.00 M) | 12.10 M (11.74 M–12.62 M) |
| PyTorch eager (Python) | 8 | 3.7 / 3.8 / 4.2 | 38.6 / 43.4 / 55.1 | 32.8 / 35.9 / 56.8 | 4.35 M (3.94 M–4.45 M) | 3.37 M (1.98 M–3.51 M) |
| ONNX Runtime (Python API) | ort-default | 2.4 / 3.6 / 4.3 | 8.1 / 9.1 / 14.0 | 11.0 / 12.5 / 19.3 | 13.36 M (12.25 M–13.64 M) | 7.96 M (7.70 M–8.13 M) |

Input copies inside the timed calls: BoundML: model only: predict_into copies the f32 input into the planned arena; tract (native Rust): every call: Tensor::from_shape copies the f32 input into an owned tract tensor; PyTorch eager (Python): model only: none in user code (float32 inputs pre-built); ONNX Runtime (Python API): model only: none in user code (float32 inputs pre-built).

**Training** (test split). The controlled rows load one control file written by BoundML's in-memory trainer: the same standardized data, initial weights and per-epoch batch order. The CLI row is a separate end-to-end measurement with its own streamed batch order.

| Trainer | Train time (min–max) | Test metric (all runs) | Peak process memory |
|---|---|---|---|
| `boundml train` CLI (own init and streamed batch order; CSV re-read every epoch + validation) | 266.4 ms (251.3 ms–274.2 ms) | rmse 0.6635 | 7.3 MiB |
| BoundML core, in-memory (writes the control file) | 30.0 ms (28.9 ms–30.3 ms) | rmse 0.6592 | 9.1 MiB |
| Burn 0.21 ndarray, controlled (decays biases too) | 89.2 ms (85.5 ms–101.3 ms) | rmse 0.6680 | 9.6 MiB |
| PyTorch, default threads, controlled | 904.7 ms (885.7 ms–930.8 ms) | rmse 0.6592 | 275.7 MiB |
| PyTorch, 1 thread, controlled | 745.3 ms (686.3 ms–818.9 ms) | rmse 0.6592 | 275.3 MiB |

Sizes: `boundml` binary 4.1 MB; package 17.5 KB (includes schema, preprocessing, self-tests); ONNX file 4.4 KB; installed PyTorch 516 MB, ONNX Runtime 37 MB.

<!-- BENCH:END -->

## Reading the results

- **Batch throughput: tract and ONNX Runtime win.** On a 256-row batch (model only),
  tract is about 3.2× faster than BoundML on the 30-feature model and 2.5× faster on
  the 11-feature model; ONNX Runtime is about 2.4× and 2.1× faster. With
  preprocessing included, tract stays about 2.1–2.3× ahead and ONNX Runtime about
  1.4–1.5×. Pre-declared target T3 is met only as originally defined (raw input
  including preprocessing: 71% and 68% of ONNX Runtime). **Model only, BoundML reaches
  42% and 48%, below the 50% threshold.** The previous measurement reported about 50%
  model only; that figure included a float64→float32 conversion inside ONNX Runtime's
  timed call, which this run removes. BoundML uses straightforward single-threaded
  loops: no packed GEMM, explicit SIMD, `target-cpu=native` or threads. If you score
  large batches and throughput dominates, use tract or ONNX Runtime.
- **Single-row calls, start-up and memory: BoundML is ahead** on these models. One
  row through preprocessing and the model takes about 0.4 µs, with no allocation, and
  the whole process peaks at about 6 MiB. tract's per-call overhead (tensor creation
  and plan state) dominates at one row. The Python runtimes pay interpreter and
  framework costs.
- **Training: BoundML's core is fast on these tiny models, but `boundml train` is
  not.** From identical initial weights and batch order, BoundML core trains in 9 ms
  and 30 ms, against 359 ms and 745 ms for single-threaded PyTorch. The CLI takes
  402 ms and 266 ms. Most of that goes to re-parsing the CSV every epoch and running
  validation, a design choice that bounds memory for large files. Caching small
  datasets in memory is on the [roadmap](../ROADMAP.md).
- **Quality.** In the controlled comparison, BoundML core and PyTorch reach the same
  test metric to four decimals on both datasets (accuracy 0.9597; RMSE 0.6592). That
  is evidence that the two compute the same training. Burn reaches the same accuracy
  and a higher RMSE on wine (0.6680); it decays biases too, and its optimizer
  implementation differs. The CLI's RMSE (0.6635) differs because it uses its own
  streamed batch order. The previous measurement's lower Burn RMSE (0.6225) came from
  Burn's own initialization and is not comparable. A linear model beats every MLP on
  this data ([walkthroughs](walkthroughs.md)).
- **Spread.** Ranges are over 5 fresh processes. A few cold starts were outliers
  (BoundML up to 53.7 ms and tract up to 237.7 ms on one run each, against medians of
  11–16 ms), so read the start-up medians, not single runs.
- The `bench.json` environment says the tree had uncommitted changes: those were
  documentation files only. The benchmark code and workers are at the recorded commit.

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
| T2 cold start | Breast cancer | ≤ 10% of PyTorch | 0.7% | **met** |
| T3 throughput | Breast cancer | ≥ 50% of ORT batch-256, raw input incl. preprocessing (as originally measured) | 71% (model only: 42%) | **met**; the model-only ratio is **below** 50%, so a model-only reading of T3 is not met |
| T4 training | Breast cancer | `boundml train` ≤ 2× PyTorch 1-thread | 1.12× | **met** |
| T5 quality | Breast cancer | within 2 pp of PyTorch | +0.0 pp accuracy | **met** |
| T1 memory | Wine quality | ≤ 20% of PyTorch peak | 3.1% | **met** |
| T2 cold start | Wine quality | ≤ 10% of PyTorch | 0.8% | **met** |
| T3 throughput | Wine quality | ≥ 50% of ORT batch-256, raw input incl. preprocessing (as originally measured) | 68% (model only: 48%) | **met**; the model-only ratio is **below** 50%, so a model-only reading of T3 is not met |
| T4 training | Wine quality | `boundml train` ≤ 2× PyTorch 1-thread | 0.36× | **met** |
| T5 quality | Wine quality | within 10% RMSE of PyTorch | RMSE 0.6635 vs 0.6592 (+0.7%) | **met** |
<!-- TARGETS:END -->

## Limits of this evidence

- One Windows machine. No Linux, macOS or ARM measurements. Peak-memory figures use
  the Windows high-water counter; on other platforms the driver can only sample RSS.
- Tiny models: the results say nothing about wide or deep networks, where optimized
  kernels dominate.
- The Python baselines include interpreter overhead per call. They represent a Python
  deployment, not the kernels' speed.
- The ONNX Runtime C API, RTen and ExecuTorch were not measured.
- These results do not show that anyone wants this workflow. That needs the
  evaluations in [design-partner-plan.md](design-partner-plan.md).
