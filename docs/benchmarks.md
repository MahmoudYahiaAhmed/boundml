# Benchmarks

Every number on this page comes from `docs/results/bench.json`, produced by the
commands below, and the tables are generated from it by `bench/report.py`. The
targets T1–T5 below were committed before the first measurement and are unchanged.

The records from the previous measurement (commit `e567a00`) are kept unchanged in
[results/history/2026-10-23-e567a00](results/history/2026-10-23-e567a00/README.md),
together with the method limitations that led to this re-measurement. Those older
records also include tract and Burn, which are no longer part of the benchmark.

## Reproduce

```bash
cargo build --release --workspace
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
| ONNX Runtime | 1.20.1, Python API, CPUExecutionProvider, default session options (multi-threaded) |
| PyTorch | 2.12.1+cpu eager, `inference_mode`, default 8 threads (training also measured with 1) |
| Python | 3.13.9 |
| Models | breast cancer: MLP 30 → 16 → 1 (ReLU, sigmoid output), 529 parameters; wine quality: MLP 11 → 32 → 16 → 1 (ReLU), 929 parameters; f32 |
| Data | UCI test splits (124 and 328 rows); a fixed 256-row batch for throughput is built by repeating test rows |

All runtimes execute **the same weights**. The ONNX and PyTorch models are rebuilt
from the BoundML package (`bench/pymodel.py`); `bench.json` records the SHA-256 of
the package manifest, weights, ONNX file and test CSV.

**Parity gate.** Each worker process writes its outputs for every test row at batch
sizes 1, 7 and 64, starting from raw values, *before its own timed section*, then
signals the driver and continues into timing. The driver validates those outputs
while the worker runs: exactly the batch sizes 1, 7 and 64; exactly `rows × outputs`
values each, with `rows` counted from the test CSV and `outputs` taken from the
package's task; every candidate and reference value finite; and every value within
`1e-5 + 1e-4 × |ref|` of BoundML's batch-1 output. The warm-up launch of every
runtime goes through the same check. A run's timings become eligible for reporting
only after its parity has validated *and* the worker has exited with status 0. Any
validation failure, malformed file or results line, non-zero exit or step timeout
(300 s) kills that worker, closes its pipes and aborts the whole benchmark before
`bench.json` is written. The largest error seen is reported per runtime.
`bench/test_run_bench.py` (run in CI on Linux, Windows and macOS) checks each of
these failure paths with scripted workers.

**Runs.** Each run starts a fresh worker process; runtimes are interleaved within a
run, after one untimed warm-up launch per runtime (file-system cache, first library
load), which still has to pass the parity gate. Every run's command, exit status, timings, memory and parity error is kept in
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
  allocation and the bundled self-test. For Python it is building the module or
  session.
- **Preprocessing alone**: standardizing one raw row with the fitted statistics
  (BoundML `Preprocessor::apply`; NumPy for Python).
- **Core inference** (model only): pre-standardized f32 input → output. Copies that the
  runtime's API forces are part of the call and listed under each table: BoundML's
  `predict_into` copies the input into its planned arena. Python inputs are pre-built
  float32 arrays, so no dtype conversion runs inside the timed call.
- **Preprocess + infer**: raw values → output.
- **Peak process memory**: the OS-reported high-water mark of the worker process: the
  peak working set on Windows, `ru_maxrss` from `wait4` on Linux and macOS. The driver
  now reads it once, after the worker exits (on Windows through the process handle),
  so short peaks are included. It covers the runtime, the loaded CSV and everything
  else. **The results below were recorded with the earlier driver**, which on Windows
  sampled the peak working set every 5 ms while the process ran. Inference workers
  wait for the driver at the end of a run, so their samples include the full peak.
  Training processes exit on their own, so their recorded peaks can be slightly
  *under*-reported if memory grew in the last few milliseconds before exit. CI found
  this race on 2026-10-25 in a 50 MB allocation test.
- **Planned tensor bytes** (BoundML only): the planner's total for one session at the
  package's `max_batch` (weights + arena). This is the quantity the budget bounds. It
  is a small part of the whole process.
- **Training** time covers the training loop only. Two kinds of rows are kept apart:
  - *CLI training* (`boundml train`): the shipped end-to-end path. It streams and
    parses the CSV every epoch through a bounded shuffle buffer and evaluates the
    validation file after each epoch. Its batch order differs from the other rows.
  - *Controlled in-memory training*: `train_core` (BoundML core) writes a control file
    with the standardized train/test tensors, the initial weights (after output-bias
    initialization) and every epoch's permutation, then trains from it. PyTorch loads the
    same file. Both use the same architecture, loss, AdamW hyperparameters (weight
    decay on weights only), batch size and epochs. Remaining differences: optimizer
    implementation details and floating-point order.

## Results

<!-- BENCH:BEGIN -->
Measured 2026-10-23 on AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 logical CPUs), Windows-11-10.0.26200-SP0; 5 runs per measurement; medians shown, with the min–max over runs in parentheses where given; one untimed warm-up launch per runtime precedes them. Commit `31f5e25` (recorded as `79ef2ab`, its ID before the 2026-10-24 history rewrite) (uncommitted changes to: `docs/benchmarks.md`). Python interpreter start-up (`python -I -c pass`): **38.8 ms**.

Every reported run's outputs matched BoundML's for every test row at batch sizes 1, 7, 64 (every output finite and |a - b| <= atol + rtol * |b|, atol 1e-05, rtol 0.0001), validated before its timings were accepted. Memory is `peak_wset`: OS-reported peak working set (high-water mark) of the worker process.

### Breast cancer (MLP 30-16-1, sigmoid output)

**Start-up and memory**

| Runtime | Process start → first prediction | Framework import | Model load | Peak process memory | Planned tensor bytes | Max parity error |
|---|---|---|---|---|---|---|
| BoundML | 5.5 ms (5.3 ms–7.4 ms) | — | 0.6 ms | 5.7 MiB (5.7 MiB–5.7 MiB) | 49,156 B | 0.0e+00 |
| PyTorch eager (Python) | 1.40 s (1.37 s–1.55 s) | 1.36 s | 3.5 ms | 201.4 MiB (201.3 MiB–201.5 MiB) | not reported | 1.5e-07 |
| ONNX Runtime (Python API) | 187.0 ms (183.9 ms–189.6 ms) | 141.0 ms | 8.9 ms | 50.1 MiB (50.0 MiB–50.3 MiB) | not reported | 1.2e-07 |

**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)

| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |
|---|---|---|---|---|---|---|
| BoundML | 1 | 0.1 / 0.1 / 0.1 | 0.3 / 0.4 / 0.4 | 0.3 / 0.4 / 0.4 | 8.95 M (8.71 M–9.06 M) | 7.04 M (6.86 M–7.08 M) |
| PyTorch eager (Python) | 8 | 3.6 / 3.7 / 4.1 | 31.1 / 32.1 / 36.9 | 26.9 / 37.3 / 44.2 | 7.42 M (6.91 M–7.57 M) | 4.59 M (3.21 M–4.66 M) |
| ONNX Runtime (Python API) | ort-default | 2.3 / 2.7 / 4.1 | 7.1 / 7.6 / 10.1 | 9.8 / 10.4 / 12.7 | 21.53 M (21.27 M–21.98 M) | 10.09 M (9.35 M–10.41 M) |

Input copies inside the timed calls: BoundML: model only: predict_into copies the f32 input into the planned arena; PyTorch eager (Python): model only: none in user code (float32 inputs pre-built); ONNX Runtime (Python API): model only: none in user code (float32 inputs pre-built).

Training rows re-measured on 2026-10-25 at commit `42abd73` (5 runs, `python bench/run_bench.py --training-only --runs 5`); inference above is from the run described at the top.

**Training** (test split). The controlled rows load one control file written by BoundML's in-memory trainer: the same standardized data, initial weights and per-epoch batch order. The CLI row is a separate end-to-end measurement with its own streamed batch order.

| Trainer | Train time (min–max) | Test metric (all runs) | Peak process memory |
|---|---|---|---|
| `boundml train` CLI (own init and streamed batch order; CSV re-read every epoch + validation) | 678.4 ms (591.5 ms–729.9 ms) | accuracy 0.9597 | 6.8 MiB |
| BoundML core, in-memory (writes the control file) | 9.2 ms (8.8 ms–11.0 ms) | accuracy 0.9597 | 7.7 MiB |
| PyTorch, default threads, controlled | 513.0 ms (463.0 ms–563.5 ms) | accuracy 0.9597 | 273.0 MiB |
| PyTorch, 1 thread, controlled | 380.7 ms (374.3 ms–382.9 ms) | accuracy 0.9597 | 272.5 MiB |

Sizes: `boundml` binary 4.1 MB; package 28.0 KB (includes schema, preprocessing, self-tests); ONNX file 2.5 KB; installed PyTorch 516 MB, ONNX Runtime 37 MB.

### Wine quality (MLP 11-32-16-1, regression)

**Start-up and memory**

| Runtime | Process start → first prediction | Framework import | Model load | Peak process memory | Planned tensor bytes | Max parity error |
|---|---|---|---|---|---|---|
| BoundML | 5.7 ms (5.4 ms–6.6 ms) | — | 0.5 ms | 5.6 MiB (5.6 MiB–5.7 MiB) | 69,252 B | 0.0e+00 |
| PyTorch eager (Python) | 1.38 s (1.33 s–1.40 s) | 1.34 s | 3.0 ms | 202.0 MiB (201.6 MiB–202.2 MiB) | not reported | 9.5e-07 |
| ONNX Runtime (Python API) | 207.6 ms (183.0 ms–224.8 ms) | 156.0 ms | 9.2 ms | 50.3 MiB (50.0 MiB–50.4 MiB) | not reported | 9.5e-07 |

**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)

| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |
|---|---|---|---|---|---|---|
| BoundML | 1 | 0.0 / 0.1 / 0.1 | 0.4 / 0.4 / 0.5 | 0.4 / 0.5 / 0.5 | 6.76 M (6.55 M–6.81 M) | 6.27 M (6.19 M–6.28 M) |
| PyTorch eager (Python) | 8 | 2.3 / 3.7 / 4.1 | 27.5 / 40.6 / 53.3 | 31.5 / 35.6 / 51.8 | 4.97 M (4.13 M–5.58 M) | 3.92 M (3.81 M–4.18 M) |
| ONNX Runtime (Python API) | ort-default | 2.4 / 3.7 / 4.0 | 7.8 / 8.1 / 11.5 | 10.5 / 11.1 / 13.5 | 13.35 M (12.34 M–14.02 M) | 9.39 M (9.18 M–9.59 M) |

Input copies inside the timed calls: BoundML: model only: predict_into copies the f32 input into the planned arena; PyTorch eager (Python): model only: none in user code (float32 inputs pre-built); ONNX Runtime (Python API): model only: none in user code (float32 inputs pre-built).

Training rows re-measured on 2026-10-25 at commit `42abd73` (5 runs, `python bench/run_bench.py --training-only --runs 5`); inference above is from the run described at the top.

**Training** (test split). The controlled rows load one control file written by BoundML's in-memory trainer: the same standardized data, initial weights and per-epoch batch order. The CLI row is a separate end-to-end measurement with its own streamed batch order.

| Trainer | Train time (min–max) | Test metric (all runs) | Peak process memory |
|---|---|---|---|
| `boundml train` CLI (own init and streamed batch order; CSV re-read every epoch + validation) | 348.4 ms (344.2 ms–539.8 ms) | rmse 0.6635 | 6.7 MiB |
| BoundML core, in-memory (writes the control file) | 29.0 ms (28.7 ms–29.7 ms) | rmse 0.6592 | 8.4 MiB |
| PyTorch, default threads, controlled | 864.6 ms (843.2 ms–1.28 s) | rmse 0.6592 | 273.9 MiB |
| PyTorch, 1 thread, controlled | 692.1 ms (674.3 ms–726.4 ms) | rmse 0.6592 | 273.7 MiB |

Sizes: `boundml` binary 4.1 MB; package 17.5 KB (includes schema, preprocessing, self-tests); ONNX file 4.4 KB; installed PyTorch 516 MB, ONNX Runtime 37 MB.

<!-- BENCH:END -->

## Reading the results

- **Batch throughput: ONNX Runtime wins.** On a 256-row batch, model only, ONNX
  Runtime is about 2.4× faster than BoundML on the 30-feature model and 2.0× faster on
  the 11-feature model. With preprocessing included it is about 1.4–1.5× faster.
  Pre-declared target T3 ("batch-256 throughput ≥ 50% of ONNX Runtime's") does not
  say whether preprocessing is included, so both readings are reported and neither is
  chosen after the fact. Including preprocessing, BoundML reaches 70% and 67% (met).
  **Model only, it reaches 42% (not met) and 51%**, which is at the threshold and
  within ONNX Runtime's run-to-run spread, so treat that one as borderline. BoundML
  uses straightforward single-threaded loops: no packed GEMM, explicit SIMD,
  `target-cpu=native` or threads. If you score large batches and throughput
  dominates, use ONNX Runtime or tract.
- **Single-row calls, start-up and memory: BoundML is ahead** on these models. One
  row through preprocessing and the model takes about 0.3–0.4 µs, with no
  allocation. The process starts and predicts in about 5.5 ms and peaks under 6 MiB.
  The Python runtimes pay interpreter and framework costs: ONNX Runtime takes about
  190–210 ms to start and 50 MiB; PyTorch takes 1.4 s and 200 MiB.
- **Training: BoundML's core is fast on these tiny models, but `boundml train` is
  not.** From identical initial weights and batch order, BoundML core trains in 9 ms
  and 29 ms, against 381 ms and 692 ms for single-threaded PyTorch. The CLI takes
  678 ms and 348 ms. Most of that goes to re-parsing the CSV every epoch and running
  validation, a design choice that bounds memory for large files. Caching small
  datasets in memory is on the [roadmap](../ROADMAP.md).
- **The training rows were re-measured on 2026-10-25** after a fix to the PyTorch
  timer boundary: it now builds its permutation tensors before its timer starts, as
  BoundML's in-memory trainer always did. Every training row ran slower in this
  session than in the previous one, including the BoundML CLI, which the fix does
  not touch (breast cancer: 367 ms then, 678 ms now). That points to a different
  machine state, not to the code change. Rows within one session are measured
  interleaved, so they remain comparable with each other; compare across sessions
  with care. As a result, T4 on breast cancer moved from 1.02× to **1.78×** against
  its 2× limit. It is still met, but with less margin. The previous record is kept
  in [results/history/2026-10-24-pre-b3](results/history/2026-10-24-pre-b3/README.md).
- **Quality.** In the controlled comparison, BoundML core and PyTorch reach the same
  test metric to four decimals on both datasets (accuracy 0.9597; RMSE 0.6592), which
  is evidence that the two compute the same training. The CLI's RMSE (0.6635)
  differs because it uses its own streamed batch order. A linear model beats every
  MLP on this data ([walkthroughs](walkthroughs.md)).
- **Spread.** Ranges are over 5 fresh processes, after one untimed warm-up launch per
  runtime. The single cold-start outliers of the previous measurement (up to 53.7 ms)
  no longer appear; BoundML's start-up range is 5.3–7.4 ms.
- `bench.json` names the files that were uncommitted when it was measured; this time
  only `docs/benchmarks.md` (prose). The benchmark code is at the recorded commit.

### Why choose BoundML when another runtime is faster

Pick BoundML when the costs it removes matter more than batch throughput: one Rust
binary or Python wheel with no ONNX toolchain or ML framework; preprocessing and the feature schema shipped
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

The thresholds were committed before the first measurement and are unchanged
(their original wording is in Git history, `docs/benchmark-targets.md`). Each is
evaluated per dataset with the measurement the target names.

<!-- TARGETS:BEGIN -->
| ID | Dataset | Threshold | Measured | Verdict |
|---|---|---|---|---|
| T1 memory | Breast cancer | ≤ 20% of PyTorch peak | 2.8% | **met** |
| T2 cold start | Breast cancer | ≤ 10% of PyTorch | 0.4% | **met** |
| T3 throughput | Breast cancer | ≥ 50% of ORT batch-256 rows/s (the pre-declared text does not say whether preprocessing is included, so both readings are shown) | incl. preprocessing 70% (**met**); model only 42% (**not met**) | **met** only if preprocessing is included |
| T4 training | Breast cancer | `boundml train` ≤ 2× PyTorch 1-thread | 1.78× | **met** |
| T5 quality | Breast cancer | within 2 pp of PyTorch | +0.0 pp accuracy | **met** |
| T1 memory | Wine quality | ≤ 20% of PyTorch peak | 2.8% | **met** |
| T2 cold start | Wine quality | ≤ 10% of PyTorch | 0.4% | **met** |
| T3 throughput | Wine quality | ≥ 50% of ORT batch-256 rows/s (the pre-declared text does not say whether preprocessing is included, so both readings are shown) | incl. preprocessing 67% (**met**); model only 51% (**met**) | **met under both readings** |
| T4 training | Wine quality | `boundml train` ≤ 2× PyTorch 1-thread | 0.50× | **met** |
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
- These results do not show that anyone wants this workflow. That needs outside
  developers deploying their own models.
