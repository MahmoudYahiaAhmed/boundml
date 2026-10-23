"""Export a Keras `Sequential` of Dense layers to a BoundML import bundle.

Supported: keras.layers.Dense with activation linear/relu/tanh/sigmoid, plus
softmax (multiclass) or sigmoid (binary) on the final Dense; Activation layers with
the same functions; InputLayer. Anything else raises UnsupportedModelError.
"""

from __future__ import annotations

import pathlib

import keras
import numpy as np

from bundle import BundleBuilder, UnsupportedModelError


def _act_name(fn) -> str:
    return getattr(fn, "__name__", str(fn))


def export_sequential(
    model: keras.Sequential,
    features: list[str],
    label: str,
    out_dir: pathlib.Path,
    reference_inputs: np.ndarray,
    classes: list[str] | None = None,
    standardization: tuple[np.ndarray, np.ndarray] | None = None,
    name: str = "keras-model",
) -> pathlib.Path:
    if not isinstance(model, keras.Sequential):
        raise UnsupportedModelError("only keras.Sequential models are supported")
    classes = classes or []
    b = BundleBuilder("keras", keras.__version__, name, features, label, classes)
    layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.Dense):
            kernel, *rest = layer.get_weights()  # kernel [in, out]
            b.dense(kernel, rest[0] if rest else None, "in_out")
            act = _act_name(layer.activation)
        elif isinstance(layer, keras.layers.Activation):
            act = _act_name(layer.activation)
        else:
            raise UnsupportedModelError(f"layer {i} ({type(layer).__name__}) is not supported; see migrate/README.md")
        if act == "linear":
            continue
        if i == last and act == "softmax":
            b.output_activation = "softmax"
        elif i == last and act == "sigmoid" and len(classes) == 2:
            b.output_activation = "sigmoid"
        else:
            b.activation(act)
    x = np.asarray(reference_inputs, dtype=np.float64)
    xs = x if standardization is None else (x - standardization[0]) / standardization[1]
    y = model.predict(xs.astype(np.float32), verbose=0)
    return b.write(out_dir, x, y, standardization)
