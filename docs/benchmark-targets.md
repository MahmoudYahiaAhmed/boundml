# Pre-declared benchmark targets

Written and committed on 2026-10-23 **before** any benchmark was run. The results
document (`docs/benchmarks.md`) must report each target as met / not met against
these thresholds, unchanged. Hardware: the single machine available to this work
(AMD Ryzen 7 8700F, 8 cores / 16 threads, 32 GB RAM, Windows 11, CPU only).

Workloads: the same dense models on the same public datasets (UCI Breast Cancer
Wisconsin Diagnostic — binary; UCI Wine Quality red — regression), f32, CPU.

| ID | Claim tested | Comparison | Threshold to call it "met" |
|----|--------------|------------|-----------------------------|
| T1 | Lower deployed memory | Peak working set of a process that loads the model and predicts the test set: BoundML (Rust binary) vs Python + PyTorch eager | BoundML <= 20% of PyTorch's |
| T2 | Lower cold start | Wall time from process launch to first prediction: BoundML CLI vs Python + PyTorch eager vs Python + ONNX Runtime | BoundML <= 10% of PyTorch's; reported vs ORT without threshold |
| T3 | No unacceptable throughput regression vs a native runtime | Batch-256 inference throughput (rows/s), in-process, warm: BoundML vs ONNX Runtime (Python API, same model) | BoundML >= 50% of ORT's |
| T4 | Native training speed acceptable | Wall time to train the same MLP for the same epochs/batch size: BoundML vs PyTorch CPU (default threads and 1 thread) | BoundML <= 2x PyTorch (1 thread) |
| T5 | Equivalent model quality | Test metric of the same architecture trained by BoundML vs PyTorch (same split, epochs, optimizer settings) | Within 2 percentage points accuracy (classification) / within 10% RMSE (regression) |
| T6 | Simpler update workflow | Count of manual steps to ship and update a model with preprocessing + schema check + memory bound | Reported qualitatively; no threshold |

Warm single-row latency (p50/p95/p99) is reported for all runtimes with no
threshold, because the ONNX Runtime and PyTorch numbers include Python call overhead
and are therefore not a like-for-like kernel comparison.
