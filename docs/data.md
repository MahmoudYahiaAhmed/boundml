# Data handling policy

| Situation | Behaviour |
|---|---|
| Feature mapping | by **column name** from the header; file column order does not matter |
| Feature column absent from the header | error `BML-D004` naming the missing columns |
| Duplicate header names | error `BML-D004` |
| Columns that are neither features nor label | `extra_columns = "ignore"` (default) skips them; `"reject"` → `BML-D004` |
| Label column absent | error for train/evaluate; allowed for `predict` |
| Empty cell in a feature | *missing* → `missing = "reject"` (default): `BML-D006` with file/line/column; `"impute_mean"`: replaced by the **training** mean |
| `NaN`, `inf`, `-inf` literals, non-numeric text | `BML-D005` (use an empty cell to mean missing) |
| Missing or unknown label | `BML-D005` |
| Value that overflows f32 after standardization | `BML-D005` / `BML-A008` |
| Constant feature in training data | allowed. It is flagged `constant: true` and uses `std = 1`, so the feature is centered only |
| Feature never observed in training | `BML-D007` |
| Ragged rows, records > 1 MiB (including an unterminated quote), > 100k columns | `BML-D002` / `BML-D004`. The size check fails *while reading*, so memory stays bounded by 1 MiB plus a 64 KiB parser buffer |
| Classification labels | strings matched exactly (after trimming) against `classes`; the index is the class id; with 2 classes the positive class is `classes[1]` |

**Fitting.** `Preprocessor::fit` makes one streaming pass over the training file only
(Welford, f64, population standard deviation). The fitted statistics are stored in
checkpoints and packages and reused unchanged for validation, test, CLI `predict`,
the embedding API and HTTP. There is no code path that refits at inference.

**Streaming and shuffling.** Training reads the CSV every epoch, holding at most
`shuffle_buffer` rows. Rows are drawn uniformly from the buffer and replaced from the
file, which gives a local shuffle; `shuffle_buffer = 1` keeps file order. If you need
a global shuffle, set `shuffle_buffer` ≥ the number of rows, which uses memory
proportional to the file, or pre-shuffle the file. The shuffle RNG for epoch `e` is
derived from `seed` and `e`. Checkpoints record the RNG state at the start of the
epoch and the rows already emitted, and resume replays the draws to the exact same
position. Validation and evaluation read in file order.

**Splitting.** `boundml split` assigns each row to train/valid/test independently
with a seeded RNG (streaming, approximate proportions, deterministic). Outputs are
written to temporary files and renamed only on success, and an `--out` directory that
would overwrite the input file is refused.

**Outside the package.** Anything the package does not represent must be applied
identically by the caller: derived features, unit conversions, categorical
encodings, windows. The package's schema only guarantees that named numeric columns
arrive with the expected names.
