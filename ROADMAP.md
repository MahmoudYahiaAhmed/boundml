# Roadmap

What happens next depends on feedback from real workloads (see the *Workload report*
issue template). Nothing below is a commitment or a date.

## Next (0.2)

- **Observed CI on Linux and macOS**, and release archives for both, verified by
  their smoke tests.
- **Faster `boundml train` on small data**: optionally keep the parsed training set
  in memory when it fits a declared budget. Today the per-epoch CSV parse dominates
  training time ([benchmarks](docs/benchmarks.md)).
- **Faster batch inference**: blocked/SIMD matmul. tract is currently 2.5–3.2× faster
  on 256-row batches (model only).
- Early stopping on a validation metric, and a learning-rate schedule.
- `boundml compare` thresholds for metric regressions, not only output drift.

## Later

- Categorical features (one-hot with a fixed vocabulary in the schema).
- Package signing with a documented key model (today hashes only detect corruption).
- `no_std` / WebAssembly inference runtime for the same package format.
- Optional multi-threaded batch inference within one session.
- A contract sidecar (schema + preprocessing + test vectors + compare) for models run
  by other Rust runtimes such as tract or RTen.

## Not planned

GPU training, CNNs/transformers, arbitrary ONNX import, distributed training. Other
tools already do these well (see [docs/decision.md](docs/decision.md)).
