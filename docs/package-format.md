# Package and checkpoint formats

## Inference package (format `boundml-package`, version 1)

A package is a **directory** (conventionally `name.boundml/`) containing exactly
these files:

| File | Content |
|---|---|
| `manifest.json` | the contract (below), UTF-8 JSON, at most 16 MiB |
| `MANIFEST.sha256` | hex SHA-256 of `manifest.json` plus a newline |
| `weights.safetensors` | f32 rank-2 tensors, one per graph parameter, named as in the graph; at most 1 GiB |

A directory was chosen over an archive. That removes path-traversal, symlink and
decompression-bomb risks, and leaves the files inspectable with ordinary tools. Ship
it as you like (tar, zip, OCI layer); integrity is checked after extraction.

### `manifest.json`

| Field | Meaning |
|---|---|
| `format`, `format_version` | `"boundml-package"`, `1` |
| `min_runtime_version` | oldest BoundML runtime that may load it (`0.1.0`) |
| `name` | 1–128 bytes |
| `task` | `{"kind":"regression"}`, `{"kind":"binary"}`, `{"kind":"multiclass","classes":K}` |
| `schema` | `features` (ordered names), `label`, `label_kind` (`regression` or `classes`), `missing` (`reject`/`impute_mean`), `extra_columns` (`ignore`/`reject`) |
| `preprocessing` | `normalize` (`standardize`/`none`), `missing`, per-feature `{name, mean, std, count, missing, constant}`, `rows_fitted` — fitted on training data only |
| `graph` | `GraphDef`: `params` (`name, rows, cols, decay`), `nodes` (`name`, `op`, operand ids), `output`, `loss` |
| `tensors` | per parameter: `name`, `shape [r,c]`, `dtype "f32"`, `layout "row_major"` |
| `bounds` | `max_batch`, `tensor_budget_bytes` (optional), `sessions`, `planner` |
| `provenance` | `created_unix`, `created_by`, `source` (`boundml-train`, `import:pytorch`, `import:keras`), `seed`, `training_config_sha256`, `data` (role, file name, sha256, rows), `metrics`, `notes` |
| `files` | `{"weights.safetensors": {"sha256", "bytes"}}` — no other entries allowed |
| `test_vectors` (optional) | `origin` (`synthetic` or `source-framework:<name>-<version>`), raw `inputs` (schema order, `null` = missing), `expected` outputs, `atol`, `rtol`; at most 1024 rows |

Unknown fields are rejected at every level (`deny_unknown_fields`).

### Load-time validation (`Package::load`)

1. The path must be a real directory. Each file must be a regular file (symlinks are
   rejected), and its size is checked against its limit *before* reading.
2. `manifest.json` must match `MANIFEST.sha256`. Then it is parsed strictly.
3. `format`, `format_version` and `min_runtime_version` are checked (`BML-A003`).
4. The declared weight size must be within the limit and must equal the bytes read.
   The hash must match (`BML-A004`).
5. Safetensors are parsed (offsets validated by the `safetensors` crate). Tensors
   must be f32, rank 2 and finite. The tensor set must equal the graph's parameters
   with identical shapes (`BML-A006`).
6. The graph is re-validated (`BML-C001`/`BML-C003`). Schema, preprocessing, task and
   input/output widths must agree.
7. The inference memory plan is **recomputed** for `bounds` with the installed
   runtime, and the budget is enforced (`BML-C005`). Independently of any declared
   budget, the planned inference tensor memory for all declared sessions must not
   exceed 1 GiB (`BML-A005`), and `sessions` is limited to 1..=256. Graph or
   consistency violations in a loaded package are reported as `BML-A006` (exit 5),
   with the underlying rule in the message. The graph must also be row-wise (see
   `docs/architecture.md`).
8. Test vectors have their widths and tolerances checked, and every input and expected
   value must be finite. A vector whose evaluation fails (for example a non-finite
   output) or whose difference is not provably within tolerance fails the self-test. `verify()` runs them
   through the full preprocessing + model path. `boundml verify`, `predict`, `serve`
   and the embedding example all run `verify()` after loading.

`Package::save` validates the same things, self-tests in memory **before** writing
(a failing package is never written), writes atomically, then reloads from disk and
self-tests again.

## Training checkpoint (format `boundml-checkpoint`, version 1)

A checkpoint is not an inference package. It is a directory with:

| File | Content |
|---|---|
| `state.json` | config (full), config SHA-256, schema, fitted preprocessing, model spec, label statistics, optimizer step, progress (`epoch`, `batches_in_epoch`, `global_step`, reader position = epoch RNG state + rows emitted, running epoch loss), per-epoch history, data file hashes, tensor-file hash |
| `state.sha256` | hex SHA-256 of `state.json` |
| `tensors.safetensors` | `param.<name>` and `opt<k>.<name>` (AdamW first/second moments, SGD momentum) |

Writes go to a temp directory first, and every file is fsynced. The previous
checkpoint is renamed to `<dir>.prev`, then the temp directory is renamed into
place. If the process dies between the two renames, `<dir>.prev` is complete, and
`Checkpoint::load` tells you to use it. Resume requires the identical configuration
(by hash) and an unchanged training file (by hash). It reproduces the remaining data
order exactly by replaying the shuffle draws.

## Compatibility and version policy

- **Package format**: a runtime reads exactly the package format versions it
  declares (currently 1). Adding fields, ops or dtypes requires a new
  `format_version`. Older runtimes then refuse the package with `BML-A003` rather
  than misreading it. Within 0.x, a runtime release may drop an old format version
  only with a note in the changelog; 1.x will keep reading every 1.x format.
- **`min_runtime_version`**: writers set the lowest runtime that implements every
  feature the package uses (currently `0.1.0`).
- **Checkpoints**: tied to the format version and to the exact configuration. They
  are not a long-term interchange format. Use packages for that.
- **Numerical compatibility**: the same package on the same binary and CPU gives
  bitwise-identical outputs. Across BoundML versions or CPUs, outputs are expected
  to agree within the package's test-vector tolerance, and `boundml verify` checks
  this after an upgrade.
- **Model updates**: `compare` marks a change *breaking* when current clients or
  deployments could fail or change meaning. That covers a changed task, label or
  classes, a new required feature, a removed feature when the candidate rejects
  extra columns, a lower `max_batch`, exceeding the current budget, stricter policies,
  a numeric difference over `--max-abs-diff`, or a requested numeric check that cannot
  run on the data. Fewer planned sessions or a removed budget are warnings. Run `boundml compare current.boundml candidate.boundml --data
  holdout.csv [--max-abs-diff X]` in CI. Exit code 6 means breaking (see
  `docs/errors.md`).

### Supported model, operator and platform matrix

| | Supported in 0.1 | Not supported |
|---|---|---|
| Models | linear regression, logistic regression, softmax regression, MLPs (any number of dense layers, one hidden activation type via `DenseSpec`; per-layer activations via import/GraphBuilder) | CNN, RNN, attention, embeddings, trees, arbitrary ONNX |
| Tasks | regression (1 target), binary, multiclass (2–4096 classes) | multi-output regression, multilabel, ranking |
| Operators | listed in `docs/architecture.md` | everything else (rejected at parse) |
| Dtypes | f32 tensors (f64 statistics) | f16/bf16/int8, quantization |
| Training | SGD (momentum, L2), AdamW, minibatch, bounded shuffle, checkpoint/resume | LR schedules, early stopping, distributed, mixed precision |
| Data | numeric CSV features, numeric or string-class labels, standardization, mean imputation | categorical encoding, text, images, Parquet |
| Execution | CPU, single-threaded kernels; concurrency via sessions | GPU/Metal/WebGPU, SIMD-specialized or threaded kernels |
| Platforms | Windows x64 (built and tested locally); Linux/macOS configured in CI, **not observed** | wasm, embedded/no_std, mobile |
