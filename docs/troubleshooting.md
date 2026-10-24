# Troubleshooting

Every error starts with a stable code. The full list is in [errors.md](errors.md).
Add `--json` to any command to get the code, message and exit code as JSON.

## "BML-D004 header of … missing feature column(s): …"

The CSV header does not contain a column named in `data.features` (or in the
package's schema). Columns are matched **by name**, exactly, after trimming spaces.
Fix the header or the config. Column order does not matter.

`unexpected column(s) …` appears when the schema has `extra_columns = "reject"`.
Either drop those columns or set `extra_columns = "ignore"` before training.

## "BML-D006 … missing value in feature …"

An empty cell under `missing = "reject"`. Either fill the data, or retrain with
`missing = "impute_mean"`. Imputation uses the *training* mean, which is stored in
the package. `NaN`/`inf` written as text are always rejected (`BML-D005`). Leave the
cell empty to mean "missing".

## "BML-D005 … unknown class; expected one of …"

A label value is not in `data.classes`. Labels are matched exactly, including case.

## "BML-C005 tensor budget exceeded: plan needs N bytes but the budget is M bytes"

The planned tensor memory for the configuration is larger than the declared budget.
The message lists the largest contributors. Options:

- lower `deploy.max_batch` (inference) or `training.batch_size` (training);
- lower `deploy.sessions` (each session has its own arena);
- use smaller hidden layers;
- raise `tensor_budget` if the memory is actually available.

`boundml plan <config> --mode infer` (or `--mode train`) shows the full breakdown
without allocating anything. BoundML never shrinks a batch or a model to make a plan
fit.

## "BML-A004 integrity check failed …"

A package file was modified or truncated after `boundml pack` (for example by a
partial copy or a line-ending conversion). Copy the package directory again in binary
mode. Checksums detect corruption; they do not prove who made the package.

## "BML-A005 planned inference tensor memory … exceeding the limit"

The package declares bounds (`max_batch` × `sessions`) whose planned memory exceeds
the 1 GiB safety ceiling. Repack with smaller `deploy.max_batch` / `deploy.sessions`.

## "BML-A003 unsupported artifact version"

The package was written by a newer BoundML than the one loading it. Upgrade the
runtime (see [install.md](install.md)).

## "BML-A007 self-test failed"

The package's bundled test vectors no longer reproduce. After a runtime upgrade,
this means numerical behaviour changed beyond the package tolerance; report it as a
bug. During `boundml import`, it means the imported model does not match the source
framework's outputs. Check that the layer list and preprocessing match the source
model.

## "BML-T002 … configuration differs from the checkpoint's"

`--resume` requires the identical config file. Start a new checkpoint directory, or
undo the config change. `BML-T003` is the same rule applied to the training and
validation files.

## The server returns 503

`BML-S001` means all workers are busy and the queue is full. Raise `--queue` or
`--workers` (workers are limited to the package's `deploy.sessions`), or retry after
the `Retry-After` interval. `BML-S003` means the server is shutting down.

Over capacity, the server rejects a request *before* buffering its body. So that the
503 can reach the client, it first reads and discards up to 64 KiB of the body, for
at most 250 ms in total, then answers with `Connection: close`. That fixed a
reproduced Windows case where a TCP reset destroyed the 503 (error 10053). It is
**not** a delivery guarantee. A client that sends a larger body, or sends it more
slowly than that, can still see the connection reset instead of a readable 503;
treat a reset during overload like a 503 and retry. Rejected requests do not take an
admission slot while draining, so each holds its connection for up to 250 ms; the
draining itself never buffers the body.

## The HTTP request returns 422

Read the message:
- a feature is missing under the `reject` policy;
- an unknown field arrived under `extra_columns = "reject"`;
- a duplicated key;
- a non-numeric value;
- more rows than `--max-rows` (default: the package `max_batch`, at most 256).

## Training is slower than expected

`boundml train` re-reads and parses the CSV every epoch, which keeps memory bounded,
and evaluates the validation file after each epoch. On small datasets this dominates
the run time: see the *pipeline* vs *core* training rows in
[benchmarks.md](benchmarks.md). Fewer epochs or no `valid` file reduce it.

## Windows: "boundml is not recognized"

The folder containing `boundml.exe` is not on `PATH`. Use the full path, or follow
the `PATH` step in [install.md](install.md) and open a new terminal.
