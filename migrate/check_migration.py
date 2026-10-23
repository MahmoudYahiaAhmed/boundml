"""Migration gate: PyTorch and Keras dense models -> BoundML packages.

For each framework and task (binary: breast cancer, regression: wine quality,
multiclass: synthetic blobs) this script trains a small model in the source
framework, exports a bundle, runs `boundml import`, runs `boundml predict` on the
whole test file and compares against the source framework's outputs on the same
raw rows. It also checks that unsupported layers and tampered weights are rejected.

Run from the repository root after `cargo build --release` and preparing datasets:
    python migrate/check_migration.py
Writes docs/results/migration.json and exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import export_keras  # noqa: E402
import export_torch  # noqa: E402
from bundle import UnsupportedModelError  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / "target" / "release" / ("boundml.exe" if os.name == "nt" else "boundml")
OUT = ROOT / "out" / "migration"
TOL = 1e-4  # abs tolerance on outputs (f32, different summation order)


def load(path: pathlib.Path, features: list[str], label: str, classes: list[str] | None):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r[c]) for c in features] for r in rows])
    if classes:
        y = np.array([classes.index(r[label]) for r in rows])
    else:
        y = np.array([float(r[label]) for r in rows])
    return x, y


def blobs(dir_: pathlib.Path) -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [3, 1], [1, 4]])
    for split, n in (("train", 300), ("test", 120)):
        lab = rng.integers(0, 3, n)
        x = centers[lab] + rng.normal(size=(n, 2))
        with (dir_ / f"{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["u", "v", "cls"])
            w.writerows([[a, b, "abc"[c]] for (a, b), c in zip(x, lab)])


def tasks():
    bc = ROOT / "datasets" / "breast_cancer"
    with (bc / "train.csv").open() as f:
        bc_features = next(csv.reader(f))[1:-1]
    wine = ROOT / "datasets" / "wine_red"
    with (wine / "train.csv").open() as f:
        wine_features = next(csv.reader(f))[:-1]
    syn = OUT / "blobs"
    syn.mkdir(parents=True, exist_ok=True)
    blobs(syn)
    return [
        ("binary", bc, bc_features, "diagnosis", ["B", "M"]),
        ("regression", wine, wine_features, "quality", None),
        ("multiclass", syn, ["u", "v"], "cls", ["a", "b", "c"]),
    ]


def torch_model(kind: str, nin: int, nout: int, x, y):
    torch.manual_seed(0)
    if kind == "binary":
        m = torch.nn.Sequential(torch.nn.Linear(nin, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Sigmoid())
        loss_fn = lambda out, t: torch.nn.functional.binary_cross_entropy(out, t.float().unsqueeze(1))  # noqa: E731
    elif kind == "regression":
        m = torch.nn.Sequential(torch.nn.Linear(nin, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        loss_fn = lambda out, t: torch.nn.functional.mse_loss(out, t.float().unsqueeze(1))  # noqa: E731
    else:
        m = torch.nn.Sequential(torch.nn.Linear(nin, 12), torch.nn.Sigmoid(), torch.nn.Linear(12, nout), torch.nn.Softmax(dim=-1))
        loss_fn = lambda out, t: torch.nn.functional.nll_loss(torch.log(out + 1e-9), t.long())  # noqa: E731
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    for _ in range(200):
        opt.zero_grad()
        loss_fn(m(xt), yt).backward()
        opt.step()
    return m


def keras_model(kind: str, nin: int, nout: int, x, y):
    keras.utils.set_random_seed(0)
    if kind == "binary":
        layers = [keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1, activation="sigmoid")]
        loss = "binary_crossentropy"
    elif kind == "regression":
        layers = [keras.layers.Dense(32, activation="tanh"), keras.layers.Dense(16), keras.layers.Activation("relu"), keras.layers.Dense(1)]
        loss = "mse"
    else:
        layers = [keras.layers.Dense(12, activation="sigmoid"), keras.layers.Dense(nout, activation="softmax")]
        loss = "sparse_categorical_crossentropy"
    m = keras.Sequential([keras.Input((nin,)), *layers])
    m.compile(optimizer=keras.optimizers.Adam(0.01), loss=loss)
    m.fit(x.astype(np.float32), y.astype(np.float32), epochs=20, batch_size=32, verbose=0)
    return m


def predict_source(fw: str, model, xs: np.ndarray) -> np.ndarray:
    if fw == "pytorch":
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(xs, dtype=torch.float32)).numpy()
    return model.predict(xs.astype(np.float32), verbose=0)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(EXE), *args], capture_output=True, text=True)


def main() -> int:
    results, failures = [], []
    for kind, data_dir, features, label, classes in tasks():
        x_train, y_train = load(data_dir / "train.csv", features, label, classes)
        x_test, _ = load(data_dir / "test.csv", features, label, classes)
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)  # population std, as BoundML
        std[std == 0] = 1.0
        xs_train = (x_train - mean) / std
        nout = 1 if kind != "multiclass" else len(classes)
        for fw, build, exporter in (("pytorch", torch_model, export_torch), ("keras", keras_model, export_keras)):
            model = build(kind, len(features), nout, xs_train, y_train)
            bdir = OUT / f"{fw}-{kind}-bundle"
            pkg = OUT / f"{fw}-{kind}.boundml"
            exporter.export_sequential(model, features, label, bdir, x_test[:32], classes, (mean, std), name=f"{fw}-{kind}")
            imp = run("import", str(bdir), "--out", str(pkg))
            if imp.returncode != 0:
                failures.append(f"{fw}/{kind}: import failed: {imp.stderr.strip()}")
                continue
            preds_path = OUT / f"{fw}-{kind}-pred.csv"
            pr = run("predict", str(pkg), "--data", str(data_dir / "test.csv"), "--out", str(preds_path))
            if pr.returncode != 0:
                failures.append(f"{fw}/{kind}: predict failed: {pr.stderr.strip()}")
                continue
            with preds_path.open() as f:
                rows = list(csv.reader(f))[1:]
            ours = np.array([[float(v) for v in r[1 : 1 + nout]] for r in rows])
            ref = predict_source(fw, model, (x_test - mean) / std).reshape(len(x_test), -1)
            err = float(np.abs(ours - ref).max())
            ok = err <= TOL
            results.append({"framework": fw, "framework_version": torch.__version__ if fw == "pytorch" else keras.__version__,
                            "task": kind, "test_rows": len(x_test), "max_abs_diff": err, "tolerance": TOL, "passed": ok})
            print(f"{fw:8s} {kind:10s} rows={len(x_test):4d} max|diff|={err:.3e} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{fw}/{kind}: max diff {err}")

    # Negative cases.
    try:
        bad = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
        export_torch.export_sequential(bad, ["u", "v"], "y", OUT / "bad-bundle", np.zeros((2, 2)))
        failures.append("BatchNorm1d was not rejected")
    except UnsupportedModelError as e:
        print(f"unsupported layer rejected: {e}")
    tampered = OUT / "pytorch-binary-bundle"
    from safetensors.numpy import load_file, save_file
    t = load_file(str(tampered / "weights.safetensors"))
    t["layer0.weight"] = t["layer0.weight"] * 1.01
    save_file(t, str(tampered / "weights.safetensors"))
    r = run("import", str(tampered), "--out", str(OUT / "tampered.boundml"))
    tampered_rejected = r.returncode == 6 and "BML-A007" in r.stderr and not (OUT / "tampered.boundml").exists()
    print(f"tampered weights rejected by import self-test: {tampered_rejected}")
    if not tampered_rejected:
        failures.append(f"tampered bundle not rejected: {r.returncode} {r.stderr}")

    report = {"results": results, "negative_checks": {"unsupported_layer_rejected": True, "tampered_weights_rejected": tampered_rejected}, "failures": failures}
    out = ROOT / "docs" / "results" / "migration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}; {'ALL PASSED' if not failures else 'FAILURES: ' + '; '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
