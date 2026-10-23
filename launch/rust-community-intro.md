# Short introduction for a Rust community (draft, e.g. r/rust or users.rust-lang.org)

**BoundML 0.1 — train small dense models from CSV and embed them with a checked contract**

I've been working on a small, deliberately narrow ML tool for Rust apps: linear and
logistic models and small MLPs on numeric tabular data. It's a CLI plus a library
(`boundml-artifact`) for loading packages.

What might interest Rust folks:
- The native computation is all in the repo: graph IR, kernels, reverse-mode autodiff,
  SGD/AdamW. No BLAS and no framework. Gradients are checked against finite differences
  and a PyTorch float64 fixture.
- A static lifetime-based arena planner. Budgets are checked *before* allocation, and a
  counting global allocator proves prediction and training steps don't allocate.
- The only `unsafe` is one function that hands out disjoint arena slices after
  runtime-checked disjointness.
- Packages bundle the schema and fitted preprocessing. Load-time validation recomputes
  the memory plan instead of trusting the file.
- `compare` gives a breaking/compatible verdict for model updates.

Honest limits:
- single-threaded scalar kernels, so tract is 2.5–3.3× faster on 256-row batches;
- only Windows has been built and tested by me;
- early release, no users yet.

Benchmarks (including where it loses) and the verification status are in the repo.
Feedback on the API and on real workloads is very welcome: [link]
