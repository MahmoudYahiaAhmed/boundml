# Error codes and exit codes

Every error message starts with a stable code. With `--json`, the CLI prints
`{"ok": false, "error": {"code", "message", "exit_code"}}` on stdout. The HTTP API
returns `{"error": {"code", "message"}}`.

## CLI exit codes

| Code | Meaning |
|---|---|
| 0 | success (for `compare`: identical or compatible) |
| 1 | internal error |
| 2 | usage error (unknown command or flag) |
| 3 | invalid input: configuration, data, schema, shape, missing file, invalid request row |
| 4 | tensor budget exceeded (`BML-C005`) |
| 5 | artifact rejected: malformed, unsupported version, integrity failure, too large, inconsistent |
| 6 | check failed: self-test failed (`BML-A007`) or `compare` verdict is breaking |

## Codes

| Code | Where | Meaning / action |
|---|---|---|
| BML-C001 | core | invalid shape or dimension; fix the model spec or data width |
| BML-C002 | core | size arithmetic overflow; dimensions are far too large |
| BML-C003 | core | invalid graph (order, arity, unused node, bad loss); fix the graph |
| BML-C004 | core / server | batch outside `1..=max_batch`; send smaller batches or plan a larger `max_batch` |
| BML-C005 | core | tensor budget exceeded; the message lists requested and available bytes and the largest contributors. Reduce batch, sessions or model width, or raise the budget. Nothing is changed automatically to fit. |
| BML-C006 | core | buffer length mismatch (API misuse) |
| BML-C007 | core | non-finite input, target, weight, loss or gradient; clean the data or lower the learning rate |
| BML-C008 | core | class index not an integer in `0..K` |
| BML-C009 | core | invalid optimizer hyperparameters |
| BML-D001 | data | cannot read a data file |
| BML-D002 | data | malformed CSV (ragged rows, encoding, record > 1 MiB) |
| BML-D003 | data | invalid schema (duplicate/empty names, < 2 classes) |
| BML-D004 | data | CSV header does not satisfy the schema (missing feature/label, duplicate column, rejected extra column) |
| BML-D005 | data | invalid cell value (not a number, NaN/inf literal, unknown class, missing label) with file, line and column |
| BML-D006 | data | missing feature value under the `reject` policy |
| BML-D007 | data | no rows, or a feature never observed in training |
| BML-D008 | data | preprocessing does not match the schema |
| BML-D009 | data | resume position invalid (data file shorter than recorded) |
| BML-T001 | train | configuration error |
| BML-T002 | train | checkpoint missing, corrupted, wrong version, or config changed since it was written |
| BML-T003 | train | training data changed since the checkpoint |
| BML-A001 | artifact | I/O error |
| BML-A002 | artifact | malformed artifact (bad JSON, unknown field or op, wrong file type, invalid safetensors) |
| BML-A003 | artifact | unsupported format version or runtime too old |
| BML-A004 | artifact | integrity (hash/size) mismatch: corrupted or modified file |
| BML-A005 | artifact | declared or actual file size over the limit, or planned inference tensor memory over the 1 GiB package ceiling |
| BML-A006 | artifact | internally inconsistent: tensor names/shapes, widths, tolerances, or a graph rule violated (the underlying `BML-C`/`BML-D` code is in the message) |
| BML-A007 | artifact | self-test failed: outputs differ from bundled test vectors |
| BML-A008 | artifact | invalid input row (missing under `reject`, duplicate or unknown field, non-finite) |
| BML-M001 | import | invalid or unsupported import bundle (including atol > 1e-3, rtol > 1e-2, or fewer than 4 reference rows) |
| BML-S001 | server | at capacity (HTTP 503 with `Retry-After: 1`) |
| BML-S002 | server | malformed JSON request (400) |
| BML-S003 | server | shutting down (503) |
| BML-S004 | server | internal worker error (500) |
| BML-S005 | server | invalid server configuration |
| BML-S006 | server | cannot bind the address |

HTTP statuses: 200 ok, 400 malformed JSON, 404 unknown route, 408 request timeout,
413 body too large, 422 invalid rows or batch size, 503 overload or shutdown.
