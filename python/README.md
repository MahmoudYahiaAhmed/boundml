# boundml (Python)

Train and run small numeric models whose **schema, fitted preprocessing, self-tests
and memory plan travel inside one package**. The computation is native Rust, the same
code as the `boundml` CLI: no PyTorch, no ONNX toolchain, and predictions identical
to the CLI, the HTTP server and the Rust API.

Early beta (0.1): dense models (linear, logistic/softmax, MLP) on numeric CSV
features, CPU only. Wheels are built and tested in CI on Windows x86_64, Linux
x86_64 and macOS arm64; they are not on PyPI yet.

## Install

From a checkout of the repository (Rust 1.94 and Python ≥ 3.9):

```bash
pip install maturin
pip install ./python
```

## Use

```python
import boundml

# Train from a TOML config (examples/breast_cancer/linear.toml in the repository).
result = boundml.train("examples/breast_cancer/linear.toml", "out/bc/ckpt")
print(result["history"][-1])

# Metrics on held-out data, next to a constant baseline.
print(boundml.evaluate("out/bc/ckpt", "datasets/breast_cancer/test.csv"))

# Package it: weights + schema + preprocessing + self-tests, verified after writing.
model = boundml.pack("out/bc/ckpt", "out/bc/model.boundml")

# Later, anywhere:
model = boundml.load("out/bc/model.boundml")
model.verify()                       # {"vectors": ..., "passed": True, ...}
model.predict(df)                    # pandas DataFrame, columns matched by name
model.predict({"radius_mean": 14.1, "texture_mean": None, ...})   # one row
model.predict(x)                     # NumPy (rows, n_features) in model.features order
```

Inputs are **raw values**: the package applies the preprocessing fitted during
training. `None`/NaN means missing and follows the schema's missing-value policy.
`predict` returns a float32 array of shape `(rows, len(model.outputs))`.

Errors raise `boundml.BoundMLError` with a stable `code` such as `BML-D004` (schema
mismatch) or `BML-C005` (memory budget exceeded); see `docs/errors.md`.

| Function | Does |
|---|---|
| `boundml.train(config, checkpoint, resume=False, max_steps=None)` | train; Ctrl-C saves a resumable checkpoint |
| `boundml.evaluate(path, data_csv)` | metrics for a checkpoint or package, with baseline |
| `boundml.pack(checkpoint, out, name=None)` | write a package and return it as a `Model` |
| `boundml.load(path)` | load and validate a package |
| `Model.predict(data)` / `.verify()` / `.memory_report(sessions)` | predict, self-test, planned memory |
| `Model.features` / `.outputs` / `.manifest` | the package's contract |

A `Model` holds one of the package's declared sessions while it has predicted; it is
thread-safe, and predictions release the GIL.

License: MIT OR Apache-2.0.
