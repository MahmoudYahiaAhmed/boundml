"""Export a PyTorch `nn.Sequential` of dense layers to a BoundML import bundle.

Supported: nn.Linear, nn.ReLU, nn.Tanh, nn.Sigmoid (hidden), and as the final module
nn.Sigmoid (binary probability) or nn.Softmax(dim=-1 / 1) (multiclass probabilities).
Anything else raises UnsupportedModelError naming the module.

    from export_torch import export_sequential
    export_sequential(model, features, label, out_dir, reference_inputs,
                      classes=["B", "M"], standardization=(mean, std))

`reference_inputs` are RAW feature rows; if `standardization` is given it is applied
before calling the model, exactly as BoundML's packaged preprocessing will.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from bundle import BundleBuilder, UnsupportedModelError

HIDDEN = {torch.nn.ReLU: "relu", torch.nn.Tanh: "tanh", torch.nn.Sigmoid: "sigmoid"}


def export_sequential(
    model: torch.nn.Sequential,
    features: list[str],
    label: str,
    out_dir: pathlib.Path,
    reference_inputs: np.ndarray,
    classes: list[str] | None = None,
    standardization: tuple[np.ndarray, np.ndarray] | None = None,
    name: str = "torch-model",
) -> pathlib.Path:
    if not isinstance(model, torch.nn.Sequential):
        raise UnsupportedModelError("only torch.nn.Sequential models are supported")
    classes = classes or []
    b = BundleBuilder("pytorch", torch.__version__, name, features, label, classes)
    modules = list(model)
    last = len(modules) - 1
    for i, m in enumerate(modules):
        if isinstance(m, torch.nn.Linear):
            w = m.weight.detach().cpu().numpy()  # [out, in]
            bias = None if m.bias is None else m.bias.detach().cpu().numpy()
            b.dense(w, bias, "out_in")
        elif i == last and isinstance(m, torch.nn.Softmax):
            if m.dim not in (-1, 1):
                raise UnsupportedModelError(f"Softmax over dim {m.dim}; only the feature dimension is supported")
            b.output_activation = "softmax"
        elif i == last and isinstance(m, torch.nn.Sigmoid) and len(classes) == 2:
            b.output_activation = "sigmoid"
        elif type(m) in HIDDEN:
            b.activation(HIDDEN[type(m)])
        else:
            raise UnsupportedModelError(f"module {i} ({type(m).__name__}) is not supported; see migrate/README.md")
    x = np.asarray(reference_inputs, dtype=np.float64)
    xs = x if standardization is None else (x - standardization[0]) / standardization[1]
    model.eval()
    with torch.no_grad():
        y = model(torch.as_tensor(xs, dtype=torch.float32)).numpy()
    return b.write(out_dir, x, y, standardization)
