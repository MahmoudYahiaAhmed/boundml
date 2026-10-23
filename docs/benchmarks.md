# Benchmarks

Targets were committed before any measurement (`docs/benchmark-targets.md`,
commit "docs: pre-declare benchmark targets before measuring"). Raw results:
`docs/results/bench.json`. Re-run: `python bench/run_bench.py --runs 5`.

## Environment

| | |
|---|---|
| Machine | AMD Ryzen 7 8700F (8 cores / 16 threads), 32 GB RAM, CPU only |
| OS | Windows 11 Pro 10.0.26200 |
| BoundML | rustc 1.94.0, `--release`, `lto = "thin"`, `codegen-units = 1`, default target CPU (no `-C target-cpu=native`), single-threaded kernels |
| PyTorch | 2.12.1+cpu, eager, `torch.inference_mode()`, default 8 threads (training also measured with 1) |
| ONNX Runtime | 1.20.1 Python API, CPUExecutionProvider, default session options; model exported from the same weights with `torch.onnx.export` (opset default, dynamic batch) |
| Python | 3.13.9 |
| Runs | 5 per measurement, median reported |

All three runtimes execute **the same weights**: the PyTorch and ONNX models are
rebuilt from the BoundML package (`bench/pymodel.py`). Output checksums over the same
batch agree to 4+ decimal places (`checksum` fields in `bench.json`).

## Methodology

- **Workload**: the MLPs from `examples/breast_cancer/mlp.toml` (30 → 16 → 1,
  sigmoid) and `examples/wine_quality/mlp.toml` (11 → 32 → 16 → 1), f32.
- **Inputs include preprocessing**: every timed call goes from raw feature values to
  model outputs. BoundML applies its packaged standardization; the Python workers apply
  the same mean/std with NumPy inside the timed region.
- **Cold start**: the driver measures wall time from `Popen` to the worker printing
  `FIRST`. That interval includes process creation, interpreter/runtime start-up,
  imports, package/model loading (BoundML: integrity checks + self-test), reading the
  test CSV, and one prediction.
- **Warm latency**: 200 warm-up calls, then 5000 single-row calls, each timed
  individually (`Instant` / `perf_counter_ns`). The Python numbers include Python
  call overhead. They measure what a Python deployment pays, **not** kernel speed.
- **Throughput**: one fixed 256-row batch (test rows repeated), 20 warm-up calls, then
  as many calls as fit in 2 s.
- **Peak memory**: the process's peak working set (`psutil` `peak_wset`), read
  after the measurements while the worker waits on stdin.
- **Training**: BoundML time is the `seconds` reported by `boundml train` (the loop
  only, excluding process start). It **includes** streaming the CSV from disk every
  epoch and the per-epoch validation pass. The PyTorch baseline (`bench/py_train.py`)
  runs the same architecture, initialization family, output-bias initialization,
  AdamW groups, batch size and epochs. Its data is held in memory as tensors and it
  does **no** per-epoch validation, so its loop does less work.

## Results

### Inference (median of 5)

| Dataset | Runtime | Cold start | Peak working set | p50 / p95 / p99 latency (1 row) | Batch-256 throughput |
|---|---|---|---|---|---|
| Breast cancer | **BoundML** | **0.011 s** | **6.1 MiB** | **0.3 / 0.4 / 0.4 µs** | 6.96 M rows/s |
| | PyTorch eager | 1.555 s | 202.2 MiB | 26.9 / 38.2 / 50.4 µs | 4.19 M rows/s |
| | ONNX Runtime | 0.225 s | 50.7 MiB | 10.4 / 13.6 / 17.6 µs | **9.87 M rows/s** |
| Wine quality | **BoundML** | **0.011 s** | **5.9 MiB** | **0.4 / 0.5 / 0.6 µs** | 6.04 M rows/s |
| | PyTorch eager | 1.641 s | 202.0 MiB | 31.4 / 39.5 / 60.7 µs | 3.31 M rows/s |
| | ONNX Runtime | 0.229 s | 50.8 MiB | 10.8 / 13.6 / 17.2 µs | **9.13 M rows/s** |

### Training (median of 5; test metric on the held-out split)

| Dataset | Runtime | Train time | Test metric |
|---|---|---|---|
| Breast cancer (80 epochs, batch 32) | BoundML | 0.430 s | accuracy 0.9597 |
| | PyTorch, 8 threads | 0.499 s | accuracy 0.9597 |
| | PyTorch, 1 thread | 0.383 s | accuracy 0.9597 |
| Wine quality (45 epochs, batch 32) | BoundML | 0.244 s | RMSE 0.6635 |
| | PyTorch, 8 threads | 0.960 s | RMSE 0.6616 |
| | PyTorch, 1 thread | 0.743 s | RMSE 0.6616 |

Every run gave the same result: identical metrics across the 5 runs for each
runtime, because both runtimes are seeded.

### Sizes

| Item | Bytes |
|---|---|
| `boundml` CLI binary (train + serve + all commands, Windows x64) | 3.9 MB |
| Package (breast cancer / wine) incl. manifest, schema, preprocessing, self-tests | 28 KB / 17 KB |
| ONNX model file (weights + graph only) | 2.5 KB / 4.4 KB |
| PyTorch installed package | 516 MB |
| ONNX Runtime installed package | 37 MB |

## Targets

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| T1 memory | BoundML ≤ 20% of PyTorch peak | 6.1 / 202.2 MiB = 3.0% (5.9 / 202.0 = 2.9%) | **met** |
| T2 cold start | BoundML ≤ 10% of PyTorch | 0.011 / 1.555 s = 0.7% (0.011 / 1.641 = 0.7%) | **met**; also 20× faster than ORT-via-Python |
| T3 throughput | BoundML ≥ 50% of ORT batch-256 | 70% (breast cancer), 66% (wine) | **met**, but ORT is faster |
| T4 training | BoundML ≤ 2× PyTorch 1-thread | 1.12× (slower) on breast cancer; 0.33× (faster) on wine | **met** |
| T5 quality | ≤ 2 pp accuracy / ≤ 10% RMSE difference | 0.0 pp; RMSE +0.3% | **met** |
| T6 workflow | qualitative | see below | reported |

### What the numbers do and do not show

- The memory and cold-start advantages mostly measure *removing the Python
  interpreter and framework* from deployment. Any compiled runtime would show a
  similar effect, and the comparison is not against such runtimes. tract, RTen,
  ExecuTorch and ONNX Runtime's C API were **not** benchmarked, so this is not
  evidence that BoundML beats native runtimes on memory or start-up.
- The single-row latency gap is mostly Python call overhead in the baselines.
- **ONNX Runtime has 1.4–1.5× higher batch throughput.** BoundML's kernels are scalar
  loops on one thread. It also does not use `target-cpu=native`, packed GEMM or
  threading. If throughput on large batches matters, this is a real regression.
- Training: BoundML is slower than single-threaded PyTorch on the 30-feature dataset,
  even though its timing also covers CSV streaming and validation. It is faster on the
  wine data, where PyTorch's per-step overhead dominates. These are tiny models; the
  results say nothing about larger networks, where PyTorch's optimized kernels would
  dominate.
- No hardware other than this one Windows machine was measured.

### T6: manual steps to ship and update a model (qualitative)

| Step | BoundML | Typical PyTorch → ONNX Runtime deployment |
|---|---|---|
| Feature order / names enforced at prediction | in package; checked on every request | application code |
| Fitted preprocessing shipped with model | in package | separate artifact (pickle/JSON) + reimplementation in serving code |
| Memory bound checked before allocation | `boundml plan` / load-time budget | not available in this workflow |
| Self-test after copy/deploy | `boundml verify` | write your own |
| Pre-update compatibility report | `boundml compare --data` | write your own |
| Commands from data to served model | `validate, train, pack, verify, serve` | training script, export script, preprocessing export, serving code |

This table compares workflows, not effort measured on real users. No design-partner
evaluation has been run (`docs/design-partner-plan.md`).
