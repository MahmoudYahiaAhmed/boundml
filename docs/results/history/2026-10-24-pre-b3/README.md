# Benchmark record before the B3 timing fix

`bench.json` here is the complete 5-run record (inference and training) as it stood
before the training rows were re-measured.

- **Measured:** 2026-10-23, with the driver at the commit now named `31f5e25` (`feat: add
  the boundml Python package`). The file records that commit as `79ef2ab…`, its ID
  before the repository history was rewritten on 2026-10-24 to remove local paths.
  The rewrite changed no benchmark code.
- **Limitation in its training rows:** the PyTorch trainer converted the controlled
  per-epoch permutations into tensors *after* starting its timer, while BoundML's
  `train_core` prepared its batch order before its timer. The PyTorch training times
  therefore included that conversion.
- **What replaced it:** `docs/results/bench.json` keeps these inference results
  unchanged and replaces only the training rows, re-measured with
  `python bench/run_bench.py --training-only --runs 5` (see `environment.training_rerun`
  there).
