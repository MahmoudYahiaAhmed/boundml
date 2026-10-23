"""Rebuild a BoundML-trained dense package as a PyTorch module (benchmark baselines).

Reads manifest.json and weights.safetensors of a package produced by `boundml pack`
(graph from DenseSpec: dense{i}.weight [in, out], dense{i}.bias [1, out], optional
dense{i}.act nodes, head 'output' = sigmoid/softmax or identity).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def load_manifest(pkg: pathlib.Path) -> dict:
    return json.loads((pkg / "manifest.json").read_text())


def standardizer(manifest: dict):
    pre = manifest["preprocessing"]
    if pre["normalize"] == "none":
        return lambda x: x
    mean = np.array([f["mean"] for f in pre["features"]])
    std = np.array([f["std"] for f in pre["features"]])
    return lambda x: (x - mean) / std


def torch_module(pkg: pathlib.Path):
    import torch
    from safetensors.numpy import load_file

    m = load_manifest(pkg)
    w = load_file(str(pkg / "weights.safetensors"))
    nodes = m["graph"]["nodes"]
    acts = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}
    layers = []
    i = 0
    while f"dense{i}.weight" in w:
        wt, b = w[f"dense{i}.weight"], w[f"dense{i}.bias"]
        lin = torch.nn.Linear(wt.shape[0], wt.shape[1])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(wt.T.copy()))
            lin.bias.copy_(torch.from_numpy(b.reshape(-1).copy()))
        layers.append(lin)
        act = next((n["op"] for n in nodes if n["name"] == f"dense{i}.act"), None)
        if act:
            layers.append(acts[act]())
        i += 1
    head = next(n["op"] for n in nodes if n["name"] == "output")
    if head == "sigmoid":
        layers.append(torch.nn.Sigmoid())
    elif head == "softmax":
        layers.append(torch.nn.Softmax(dim=-1))
    return torch.nn.Sequential(*layers).eval(), m


def read_features(csv_path: pathlib.Path, features: list[str]) -> np.ndarray:
    import csv

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[c]) for c in features] for r in rows])
