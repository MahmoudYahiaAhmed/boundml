# Starter issues (drafts — not filed)

Each is scoped so a newcomer can finish it in a few hours with the existing tests as a guide.

## 1. `boundml inspect --json` should include the output column names
**Labels:** good first issue, enhancement
`commands::inspect` returns the manifest and memory report. Clients currently
recompute output names (`p_<class>`) themselves. Add `"outputs": manifest.output_names()`
to the JSON, plus an assertion in `crates/boundml-cli/tests/e2e.rs`.

## 2. Show elapsed time per epoch in `boundml train`
**Labels:** good first issue
`TrainEvent::Epoch` is printed in `commands::train`. Add the wall time of each epoch
(measure in `trainer.rs`, carry it in `EpochRecord` as an optional field so old
checkpoints still load) and print it next to `train_loss`.

## 3. Document and test `predict` on a CSV without the label column
**Labels:** good first issue, documentation
This already works (`CsvSource::open(.., require_label = false)`), but no test covers
it. Add an e2e case and a sentence to `docs/walkthroughs.md`.

## 4. Add a `--threshold` option for binary `predict`
**Labels:** enhancement
`predict` always uses 0.5 for the `predicted` column. Add `--threshold <p>`
(validated to be in 0..1), use it in the `predicted` column only, and record it in the
JSON output.

## 5. Cache the parsed training set when it fits in a budget
**Labels:** enhancement, performance
The benchmarks show `boundml train` time dominated by re-parsing the CSV every epoch.
Add an optional `data.cache_in_memory = "8MiB"` that keeps preprocessed rows in memory
when they fit. Resume semantics must stay exact (extend `tests/train.rs`).

## 6. Run the release smoke test on Linux and macOS
**Labels:** ci, help wanted
`scripts/release.py` has only run on Windows. Run it on Linux/macOS (or via the
`package-smoke` CI job), fix any platform issue, and update `docs/install.md`'s
status table with the observed result.
