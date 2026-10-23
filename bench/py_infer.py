"""Python inference baseline worker (same protocol as the Rust `infer` worker).

    python bench/py_infer.py torch <package> <csv>
    python bench/py_infer.py ort   <package> <csv> <model.onnx>

Timing includes NumPy standardization with the package's fitted statistics so the
work matches BoundML's raw-in / output-out path. Prints FIRST after the first
prediction, then a JSON line, then waits for stdin.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pymodel import load_manifest, read_features, standardizer  # noqa: E402


def main() -> None:
    runtime, pkg, csv_path = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    if runtime == "torch":
        import torch

        from pymodel import torch_module

        model, manifest = torch_module(pkg)

        def predict(x):
            with torch.inference_mode():
                return model(torch.from_numpy(x.astype(np.float32))).numpy()
    else:
        import onnxruntime as ort

        manifest = load_manifest(pkg)
        sess = ort.InferenceSession(sys.argv[4], providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def predict(x):
            return sess.run(None, {name: x.astype(np.float32)})[0]

    std = standardizer(manifest)
    rows = read_features(csv_path, manifest["schema"]["features"])
    predict(std(rows[:1]))
    print("FIRST", flush=True)

    for i in range(200):
        predict(std(rows[i % len(rows) : i % len(rows) + 1]))
    lat = []
    for i in range(5000):
        r = rows[i % len(rows) : i % len(rows) + 1]
        t = time.perf_counter_ns()
        predict(std(r))
        lat.append((time.perf_counter_ns() - t) / 1000)
    lat.sort()
    batch = rows[np.arange(256) % len(rows)]
    for _ in range(20):
        predict(std(batch))
    t, iters = time.perf_counter(), 0
    while time.perf_counter() - t < 2.0:
        out = predict(std(batch))
        iters += 1
    rps = iters * 256 / (time.perf_counter() - t)
    pct = lambda p: lat[round((len(lat) - 1) * p)]  # noqa: E731
    print(json.dumps({"runtime": runtime, "latency_us": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
                      "batch256_rows_per_s": rps, "checksum": float(out.astype(np.float64).sum())}), flush=True)
    sys.stdin.readline()


if __name__ == "__main__":
    main()
