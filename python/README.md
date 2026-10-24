# boundml (Python)

Train and run small numeric models whose **schema, fitted preprocessing, self-tests
and memory plan travel inside one package**. The computation is native Rust, the same
code as the `boundml` CLI: no PyTorch, no ONNX toolchain, and predictions identical
to the CLI, the HTTP server and the Rust API.

Early beta (0.1): dense models (linear, logistic/softmax, MLP) on numeric CSV
features, CPU only. Wheels are built and tested in CI on Windows x86_64, Linux
x86_64 and macOS arm64; they are not on PyPI yet.

## Install

Not on PyPI yet. **Do not run `pip install boundml`**: that name on PyPI belongs to an
unrelated project (a branch-and-bound solver toolbox), so it installs something else.
Install from a checkout (builds a wheel; needs Rust 1.94+) or from a
wheel file built by CI or by `maturin build --release`:

```bash
pip install ./python                                   # from the repository root
pip install boundml-0.1.0-cp39-abi3-win_amd64.whl      # or a built wheel
```

One abi3 wheel per platform serves Python 3.9 and later. Verified on Python 3.11–3.14
(Windows), 3.12 (Linux) and 3.13 (CI on Windows, Linux and macOS); 3.9 and 3.10 have
not been run. Linux wheels built on current runners need glibc 2.34 or newer.

## Use

Run this from the repository root: the example config and the data it names live
there (config paths are relative to the config file).

```python
import boundml
import pandas as pd

# Train from a TOML config, then compare with a constant baseline on held-out data.
result = boundml.train("examples/breast_cancer/linear.toml", "out/bc/ckpt")
print(result["completed"], result["global_step"])
metrics = boundml.evaluate("out/bc/ckpt", "datasets/breast_cancer/test.csv")
print(metrics["model"]["accuracy"], metrics["baseline"]["accuracy"])   # 0.9596... 0.5806...

# Package it: weights + schema + fitted preprocessing + self-tests, verified after writing.
model = boundml.pack("out/bc/ckpt", "out/bc/model.boundml")

# Later, from anywhere: load and predict from raw values.
model = boundml.load("out/bc/model.boundml")
print(model.verify()["passed"])                     # True
df = pd.read_csv("datasets/breast_cancer/test.csv") # extra columns (id, diagnosis) are ignored
p = model.predict(df)                               # float32, shape (124, 1): P(class "M")
print(model.outputs, p.shape, p[:3, 0])
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
