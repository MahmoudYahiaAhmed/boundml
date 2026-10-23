# boundml-core

The computational core of [BoundML](../../README.md): a small static graph IR over
f32 matrices, CPU kernels, reverse-mode differentiation, a lifetime-based
tensor-memory planner with budgets that are checked before allocation, and
SGD/AdamW. It does no I/O; its only dependencies are `serde` and `thiserror`.

Most users want `boundml-artifact` (load and run packages) or the `boundml` CLI.
Semantics and memory boundaries: [docs/architecture.md](../../docs/architecture.md).
