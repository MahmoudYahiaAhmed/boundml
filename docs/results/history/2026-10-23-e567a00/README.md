# Historical results (commit e567a00, 2026-10-23)

Raw results recorded before the external source review's reliability fixes. They are
kept unchanged as historical evidence. The current tables in `docs/benchmarks.md`
are generated from newer results in `docs/results/`.

Known limitations of these records, identified by the review:
- benchmark inference results store only per-runtime medians (no per-run records or
  dispersion);
- there was no elementwise output-parity gate (only one batch checksum);
- timed Rust inference calls were not individually made observable with `black_box`;
- Python "model-only" paths still converted inputs to f32 inside the timed region;
- training comparisons used different initial weights and batch orders across
  frameworks;
- the stress worker created 4 predictors from a package that declared 2 sessions,
  which was unenforced at the time.
