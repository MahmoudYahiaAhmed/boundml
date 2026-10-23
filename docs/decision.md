# Product decision

Date: 2026-10-23. Inputs: `docs/research.md` (primary sources, accessed 2026-10-23),
the environment available (Windows x64, 16 logical CPUs, no GPU, Rust 1.94, PyTorch
2.12, TensorFlow/Keras 2.21/3.14, ONNX Runtime 1.20). Scores are engineering
judgments on a 1–5 scale, not market data.

## Candidates

Weights: user pain 25%, measurable advantage 20%, feasibility 20%, adoption friction 15%,
reason to use Rust 10%, ability to demonstrate honestly 10%.

| Candidate | Pain | Advantage | Feasibility | Adoption | Rust | Demonstrable | Weighted |
|---|---|---|---|---|---|---|---|
| **A. BoundML** — train/package/embed small dense tabular models with one contract (schema, fitted preprocessing, graph, test vectors, planned memory) | 3.5 | 3 | 4.5 | 3.5 | 4 | 4.5 | **3.7** |
| B. Preprocessing/contract layer for existing Rust runtimes (tract/RTen/Burn): wrap ONNX models with schema + fitted preprocessing + self-test + diff, no training | 3.5 | 2.5 | 4.5 | 4 | 3 | 4 | 3.6 |
| C. On-device retraining loop for edge gateways (incremental training of a deployed dense model with re-verification and rollback) | 3 | 3 | 2.5 | 2.5 | 4 | 2.5 | 2.9 |

### A. BoundML (selected)
- **Who**: Rust developers who ship small prediction models inside desktop apps,
  local services or Linux gateways.
- **Today**: they train in Python (scikit-learn/PyTorch). Then they export to ONNX
  and run tract/RTen/ORT, or hand-port coefficients. Preprocessing and feature order
  are re-implemented in Rust and checked by hand.
- **What improves**: one Rust toolchain covers data → training → package → embedding.
  The package carries the feature contract and the fitted preprocessing. Loading
  verifies integrity and self-tests, memory is planned and bounded before allocation,
  and `compare` gives a CI-readable verdict before an update.
- **Why switch**: fewer artifacts and less hand-written glue for the narrow
  dense-model case, and no Python in the training or deployment loop.
- **Closest alternatives**: Aprender (pure-Rust classical ML, versioned `.apr`,
  validate/diff; fitted preprocessing and schema were not found inside the artifact in
  the files checked), ExecuTorch (memory planning + bundled self-tests, but Python
  authoring, C++ runtime, no tabular schema), Burn (Rust training/inference; no
  bundled preprocessing or budget documented), TF Transform (fitted preprocessing in
  the SavedModel; Python/TF).
- **What could invalidate it**: Aprender or Burn adding schema + preprocessing to
  their formats; users not caring about memory bounds; the dense-model family being
  too narrow (tabular users often prefer gradient-boosted trees); teams preferring to
  train in Python anyway.
- **Original implementation**: graph IR, kernels, autodiff, memory planner, package
  contract, compare, training loop. **Reused**: serde, csv, safetensors, sha2, clap,
  axum/tokio, toml.

### B. Contract layer over existing runtimes
- Strong on adoption, because users keep their training stack. It would not satisfy
  the requirement to replace the training workflow, though, and the contract value
  is the same as A's packaging half. It is a good **upstream contribution** target:
  BoundML's `Schema` + `Preprocessor` + test vectors + `compare` could be offered as a
  sidecar format for tract/RTen packages.

### C. On-device retraining
- Real pain for some edge products, but validation needs real devices and drifting
  data that this environment does not have. It is also hard to demonstrate honestly.
  It is deferred; BoundML's resumable checkpoints are a foundation for it.

## Decision

Build A within the frozen first-release scope. The research found no single tool that
documents the whole integrated workflow. Each component, though, has prior art
(ExecuTorch planning and bundled I/O, TF Transform, TFLM arenas, Aprender
validate/diff), and none of these ideas is claimed as new. The product claim is the
integration and the enforcement at its boundaries. Whether that is enough reason to
switch is **unvalidated** until external users evaluate it
(`docs/design-partner-plan.md`).
