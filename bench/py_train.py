"""PyTorch training baseline equivalent to a BoundML config.

Same architecture, initialization family, output-bias init, AdamW settings (weight
decay on weights only), batch size, epochs, train-only standardization and a
per-epoch shuffle. Prints one JSON line with wall time and test metrics.

    python bench/py_train.py <config.toml> <test.csv> <threads|default> [<control.json>]

With a control file written by `train_core`, the standardized data, initial weights
and every epoch's batch order come from that file instead, so this run starts from
the same parameters and visits the same batches as BoundML's in-memory trainer.
"""

import json
import math
import pathlib
import sys
import time
import tomllib

import numpy as np
import torch


def main() -> None:
    cfg_path, test_path, threads = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    if threads != "default":
        torch.set_num_threads(int(threads))
    cfg = tomllib.loads(cfg_path.read_text())
    base = cfg_path.parent
    d, m, t = cfg["data"], cfg.get("model", {}), cfg["training"]
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from pymodel import read_features

    import csv

    def labels(p):
        with p.open() as f:
            rows = list(csv.DictReader(f))
        if d["task"] == "classification":
            return np.array([d["classes"].index(r[d["label"]]) for r in rows])
        return np.array([float(r[d["label"]]) for r in rows])

    xtr = read_features(base / d["train"], d["features"])
    ytr = labels(base / d["train"])
    xte = read_features(test_path, d["features"])
    yte = labels(test_path)
    mean, std = xtr.mean(0), xtr.std(0)
    std[std < 1e-12] = 1.0
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    control = json.loads(pathlib.Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
    if control:
        nf = control["n_features"]
        xtr = np.array(control["x_train"], dtype=np.float32).reshape(-1, nf)
        xte = np.array(control["x_test"], dtype=np.float32).reshape(-1, nf)
        ytr = np.array(control["y_train"])
        yte = np.array(control["y_test"])
        if d["task"] == "classification":
            ytr, yte = ytr.astype(np.int64), yte.astype(np.int64)

    torch.manual_seed(cfg.get("seed", 0))
    classes = len(d.get("classes", []))
    out = 1 if classes <= 2 else classes
    widths = [len(d["features"]), *m.get("hidden", []), out]
    act = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}[m.get("activation", "relu")]
    layers = []
    for i in range(len(widths) - 1):
        lin = torch.nn.Linear(widths[i], widths[i + 1])
        hidden_relu = act is torch.nn.ReLU and i < len(widths) - 2
        bound = math.sqrt(6 / widths[i]) if hidden_relu else math.sqrt(6 / (widths[i] + widths[i + 1]))
        torch.nn.init.uniform_(lin.weight, -bound, bound)
        torch.nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(widths) - 2:
            layers.append(act())
    model = torch.nn.Sequential(*layers)
    with torch.no_grad():
        last = layers[-1].bias
        if classes == 0:
            last.fill_(float(ytr.mean()))
        elif classes == 2:
            p = min(max(ytr.mean(), 1e-3), 1 - 1e-3)
            last.fill_(math.log(p / (1 - p)))
        else:
            pri = np.bincount(ytr, minlength=classes) / len(ytr)
            last.copy_(torch.tensor(np.log(np.maximum(pri, 1e-3))))
    if control:
        linears = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
        with torch.no_grad():
            for i, lin in enumerate(linears):
                w = control["init"][f"dense{i}.weight"]
                b = control["init"][f"dense{i}.bias"]
                lin.weight.copy_(torch.tensor(w["values"], dtype=torch.float32).reshape(w["rows"], w["cols"]).T)
                lin.bias.copy_(torch.tensor(b["values"], dtype=torch.float32))
    o = t["optimizer"]
    decay = [p for n, p in model.named_parameters() if n.endswith("weight")]
    nodecay = [p for n, p in model.named_parameters() if n.endswith("bias")]
    if o["kind"] == "adamw":
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": o.get("weight_decay", 0.01)}, {"params": nodecay, "weight_decay": 0.0}],
                                lr=o["lr"], betas=(o.get("beta1", 0.9), o.get("beta2", 0.999)), eps=o.get("eps", 1e-8))
    else:
        opt = torch.optim.SGD([{"params": decay, "weight_decay": o.get("weight_decay", 0.0)}, {"params": nodecay, "weight_decay": 0.0}],
                              lr=o["lr"], momentum=o.get("momentum", 0.0))
    xt = torch.tensor(xtr, dtype=torch.float32)
    if classes == 0 or classes == 2:
        yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        loss_fn = torch.nn.functional.mse_loss if classes == 0 else torch.nn.functional.binary_cross_entropy_with_logits
    else:
        yt = torch.tensor(ytr, dtype=torch.long)
        loss_fn = torch.nn.functional.cross_entropy
    bs = t["batch_size"]
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    start = time.perf_counter()
    perms = [torch.tensor(p, dtype=torch.long) for p in control["perms"]] if control else None
    for epoch in range(t["epochs"]):
        perm = perms[epoch] if perms else torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    secs = time.perf_counter() - start
    with torch.no_grad():
        z = model(torch.tensor(xte, dtype=torch.float32)).numpy()
    if classes == 0:
        metric = {"rmse": float(np.sqrt(((z[:, 0] - yte) ** 2).mean()))}
    elif classes == 2:
        metric = {"accuracy": float(((z[:, 0] >= 0) == (yte == 1)).mean())}
    else:
        metric = {"accuracy": float((z.argmax(1) == yte).mean())}
    print(json.dumps({"runtime": "pytorch", "threads": threads, "controlled": bool(control), "torch": torch.__version__, "train_seconds": secs, "test": metric}))


if __name__ == "__main__":
    main()
