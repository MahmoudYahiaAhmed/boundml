# boundml-rs (Python)

**Small numeric models, packaged with their preprocessing and schema.** Train linear,
logistic/softmax or small MLP models from CSV, and get one checked package with the
feature schema, fitted preprocessing, weights, self-tests and a planned memory budget.
Load it and predict from **raw** values. The computation is native Rust (no PyTorch,
no ONNX toolchain), the same code as the BoundML CLI and Rust library, so predictions
are bitwise identical across all three.

**Early public beta (0.1.0).** Dense models on numeric features, CPU only. Project,
CLI and Rust library: <https://github.com/MahmoudYahiaAhmed/boundml>.

## Install

```bash
pip install boundml-rs
```

```python
import boundml_rs
```

The distribution is `boundml-rs` and the import name is `boundml_rs`. The project
named `boundml` on PyPI is unrelated.

| | Supported and tested |
|---|---|
| Python | 3.11, 3.12, 3.13, 3.14 (each tested in CI on every platform below) |
| Platforms | Windows x86_64; Linux x86_64 with glibc 2.34 or newer (for example Ubuntu 22.04+, Debian 12+); macOS on Apple silicon (arm64) |
| Not built | Intel macOS, ARM Linux, musl (Alpine), 32-bit; there is no source distribution yet |
| Dependencies | NumPy 2.x; pandas optional (for DataFrame input) |

## Use

Training takes a TOML config and CSV files. The CLI release archives include a real
example: UCI breast-cancer data, a 30-feature binary classification problem. Download
an archive from the [v0.1.0 release](https://github.com/MahmoudYahiaAhmed/boundml/releases/tag/v0.1.0),
extract it, and run this from the extracted folder:

```python
import boundml_rs
import pandas as pd

# Train from a TOML config, then compare with a constant baseline on held-out data.
result = boundml_rs.train("examples/breast_cancer/linear.toml", "bc/ckpt")
print(result["completed"], result["global_step"])                     # True 960
metrics = boundml_rs.evaluate("bc/ckpt", "datasets/breast_cancer/test.csv")
print(metrics["model"]["accuracy"], metrics["baseline"]["accuracy"])  # 0.9596... 0.5806...

# Package it: weights + schema + fitted preprocessing + self-tests, verified after writing.
model = boundml_rs.pack("bc/ckpt", "bc/model.boundml")

# Later, from anywhere: load and predict from raw values.
model = boundml_rs.load("bc/model.boundml")
print(model.verify()["passed"])                       # True
df = pd.read_csv("datasets/breast_cancer/test.csv")   # extra columns (id, diagnosis) are ignored
p = model.predict(df)                                 # float32, shape (124, 1): P(class "M")
print(model.outputs, p.shape, p[:3, 0])               # ['p_M'] (124, 1) [0.9998099  0.99994636 0.9985293 ]
```

`predict` accepts a pandas DataFrame (columns matched by name), one dict or a list of
dicts (keys are feature names), or a NumPy array of shape `(rows, n_features)` in
`model.features` order. `None` or NaN means missing and follows the schema's
missing-value policy. The result is a float32 array of shape `(rows, len(model.outputs))`.

Errors raise `boundml_rs.BoundMLError` with a stable `code`, for example `BML-D004`
(schema mismatch), `BML-D005` (a value that is not a number) or `BML-C005` (memory
budget exceeded). Every code is explained in
[docs/errors.md](https://github.com/MahmoudYahiaAhmed/boundml/blob/main/docs/errors.md).

| Function | Does |
|---|---|
| `boundml_rs.train(config, checkpoint, resume=False, max_steps=None)` | train; Ctrl-C saves a resumable checkpoint |
| `boundml_rs.evaluate(path, data_csv)` | metrics for a checkpoint or package, with a constant baseline |
| `boundml_rs.pack(checkpoint, out, name=None)` | write a package and return it as a `Model` |
| `boundml_rs.load(path)` | load and validate a package |
| `Model.predict(data)` / `.verify()` / `.memory_report(sessions)` | predict, self-test, planned memory |
| `Model.features` / `.outputs` / `.manifest` | the package's contract |

A `Model` holds one of the package's declared sessions once it has predicted. It is
thread-safe, and predictions release the GIL.

## Limits

Dense models only (linear, logistic/softmax, MLPs with ReLU/Tanh/Sigmoid) on numeric
features; no categorical features, trees, images, text or GPU. Single-threaded
kernels: ONNX Runtime is about 2–2.4× faster on large batches. On the two bundled
datasets, the tested MLPs did not beat the linear baselines. Measurements and their
limits: [docs/benchmarks.md](https://github.com/MahmoudYahiaAhmed/boundml/blob/main/docs/benchmarks.md).

License: MIT OR Apache-2.0.
