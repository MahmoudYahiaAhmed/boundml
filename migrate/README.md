# Migrating dense models from PyTorch or Keras

BoundML imports a deliberately narrow model family: sequential stacks of dense
layers with ReLU/Tanh/Sigmoid hidden activations and an identity (regression),
sigmoid (binary) or softmax (multiclass) output. This is a bridge for that
family only; it is not a general PyTorch/TensorFlow/ONNX importer.

| Source | Supported | Rejected with an error |
|--------|-----------|------------------------|
| PyTorch `nn.Sequential` | `Linear` (with or without bias), `ReLU`, `Tanh`, `Sigmoid`; final `Sigmoid` with 1 output (binary) or `Softmax(dim=-1)` with ≥ 3 outputs (multiclass) | everything else (BatchNorm, Dropout, Conv, custom modules, non-Sequential models) |
| Keras `Sequential` | `Dense` with `linear/relu/tanh/sigmoid` activation, final `sigmoid` (1 unit) or `softmax` (≥ 3 units), `Activation` layers, `InputLayer` | everything else |

Classifiers must end in the probability activation. A model that outputs raw logits
(for example one trained with `CrossEntropyLoss` and no final `Softmax`) is rejected;
append `nn.Softmax(dim=-1)` / `nn.Sigmoid()` before exporting. A two-class softmax
model is also rejected: export it as one sigmoid output instead.

Weight layout is handled explicitly: PyTorch `Linear.weight` is `[out, in]` and is
transposed on import; Keras kernels are `[in, out]`. Weights are stored as f32.

## Steps

```bash
pip install torch safetensors numpy   # or keras/tensorflow
```

```python
import sys; sys.path.insert(0, "migrate")
from export_torch import export_sequential
export_sequential(model, features=["a", "b"], label="y", out_dir=pathlib.Path("bundle"),
                  reference_inputs=raw_rows, classes=["neg", "pos"],
                  standardization=(train_mean, train_std))   # optional
```

```bash
boundml import bundle --out model.boundml
boundml verify model.boundml
```

`reference_inputs` are raw feature rows; the exporter records the source
framework's outputs for them. `boundml import` fails (exit code 6, `BML-A007`) and
writes nothing unless the imported model reproduces those outputs within the
bundle's tolerance (default `atol=1e-5`, `rtol=1e-4`; f32 with a different
summation order than the source framework). Bundles may not loosen this beyond
`atol=1e-3` / `rtol=1e-2`, and must contain at least 4 reference rows. The reference rows become the
package's test vectors with origin `source-framework:<name>-<version>`.

Preprocessing: if the source pipeline standardized inputs, pass the training mean
and population standard deviation; they become the package's fitted
preprocessing. Other transforms (log, one-hot, clipping) are not representable and
must stay outside the package — document them.

## Verification gate

`python migrate/check_migration.py` trains small models in both frameworks on the
demo datasets, imports them, runs `boundml predict` on the full test files and
compares with the source frameworks. Results: `docs/results/migration.json`.
