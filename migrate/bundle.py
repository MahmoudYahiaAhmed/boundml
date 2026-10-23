"""Shared writer for BoundML import bundles (bundle.json + weights.safetensors).

A bundle is an explicit architecture description plus weights. It contains no
pickled objects or code. `boundml import <bundle> --out model.boundml` validates it
and fails unless the imported model reproduces the reference outputs.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
from safetensors.numpy import save_file


class UnsupportedModelError(ValueError):
    """Raised for layers or configurations outside the supported dense subset."""


@dataclass
class BundleBuilder:
    source: str
    framework_version: str
    name: str
    features: list[str]
    label: str
    classes: list[str] = field(default_factory=list)
    layers: list[dict] = field(default_factory=list)
    tensors: dict[str, np.ndarray] = field(default_factory=dict)
    output_activation: str = "none"

    def dense(self, weight: np.ndarray, bias: np.ndarray | None, layout: str) -> None:
        i = sum(1 for l in self.layers if l["type"] == "dense")
        wname, bname = f"layer{i}.weight", f"layer{i}.bias"
        self.tensors[wname] = np.ascontiguousarray(weight, dtype=np.float32)
        layer = {"type": "dense", "weight": wname, "bias": None, "weight_layout": layout}
        if bias is not None:
            self.tensors[bname] = np.ascontiguousarray(bias.reshape(1, -1), dtype=np.float32)
            layer["bias"] = bname
        self.layers.append(layer)

    def activation(self, fn: str) -> None:
        if fn not in ("relu", "tanh", "sigmoid"):
            raise UnsupportedModelError(f"hidden activation '{fn}' is not supported (relu, tanh, sigmoid)")
        self.layers.append({"type": "activation", "fn": fn})

    def write(
        self,
        out_dir: pathlib.Path,
        reference_inputs: np.ndarray,
        reference_outputs: np.ndarray,
        standardization: tuple[np.ndarray, np.ndarray] | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> pathlib.Path:
        if not any(l["type"] == "dense" for l in self.layers):
            raise UnsupportedModelError("model has no dense layers")
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format": "boundml-import",
            "version": 1,
            "source": self.source,
            "framework_version": self.framework_version,
            "name": self.name,
            "features": self.features,
            "label": self.label,
            "classes": self.classes,
            "layers": self.layers,
            "output_activation": self.output_activation,
            "reference": {
                "inputs": np.asarray(reference_inputs, dtype=np.float64).tolist(),
                "outputs": np.asarray(reference_outputs, dtype=np.float32).tolist(),
            },
            "atol": atol,
            "rtol": rtol,
        }
        if standardization is not None:
            mean, std = standardization
            bundle["preprocessing"] = {"mean": np.asarray(mean, float).tolist(), "std": np.asarray(std, float).tolist()}
        (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=1))
        save_file(self.tensors, str(out_dir / "weights.safetensors"))
        return out_dir
