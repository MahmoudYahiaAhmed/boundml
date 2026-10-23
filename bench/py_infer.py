"""Python inference baseline worker (same protocol and fields as the Rust workers).

    python bench/py_infer.py torch <package> <csv>
    python bench/py_infer.py ort   <package> <csv> <model.onnx>

Reports import time (framework import), model load time, NumPy standardization
alone, warm core inference on pre-standardized inputs, and end-to-end
(standardize + infer). Interpreter start-up is measured separately by the driver.
"""

import time

T0 = time.perf_counter()

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pymodel import load_manifest, read_features, standardizer  # noqa: E402


def quantiles(v: list[float]) -> dict:
    v = sorted(v)
    return {k: v[round((len(v) - 1) * p)] for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}


def timed(n: int, f) -> list[float]:
    out = []
    for i in range(n):
        t = time.perf_counter_ns()
        f(i)
        out.append((time.perf_counter_ns() - t) / 1000)
    return out


def rows_per_s(f) -> float:
    for _ in range(20):
        f()
    t, iters = time.perf_counter(), 0
    while time.perf_counter() - t < 2.0:
        f()
        iters += 1
    return iters * 256 / (time.perf_counter() - t)


def main() -> None:
    runtime, pkg, csv_path = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    if runtime == "torch":
        import torch

        from pymodel import torch_module

        import_s = time.perf_counter() - T0
        t = time.perf_counter()
        model, manifest = torch_module(pkg)
        threads = torch.get_num_threads()

        def predict(x):
            with torch.inference_mode():
                return model(torch.from_numpy(x.astype(np.float32))).numpy()
    else:
        import onnxruntime as ort

        import_s = time.perf_counter() - T0
        t = time.perf_counter()
        manifest = load_manifest(pkg)
        sess = ort.InferenceSession(sys.argv[4], providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        threads = "ort-default"

        def predict(x):
            return sess.run(None, {name: x.astype(np.float32)})[0]

    std = standardizer(manifest)
    load_s = time.perf_counter() - t
    rows = read_features(csv_path, manifest["schema"]["features"])
    predict(std(rows[:1]))
    print("FIRST", flush=True)

    n = len(rows)
    singles = [std(rows[i : i + 1]).astype(np.float32) for i in range(n)]
    prep = timed(5000, lambda i: std(rows[i % n : i % n + 1]))
    for i in range(200):
        predict(singles[i % n])
    lat_core = timed(5000, lambda i: predict(singles[i % n]))
    lat_e2e = timed(5000, lambda i: predict(std(rows[i % n : i % n + 1])))
    batch = rows[np.arange(256) % n]
    bx = std(batch).astype(np.float32)
    core = rows_per_s(lambda: predict(bx))
    e2e = rows_per_s(lambda: predict(std(batch)))
    out = predict(bx)
    print(json.dumps({"runtime": runtime, "threads": threads, "import_s": import_s, "load_s": load_s,
                      "preprocess_us_per_row": quantiles(prep), "latency_core_us": quantiles(lat_core),
                      "latency_e2e_us": quantiles(lat_e2e), "batch256_core_rows_per_s": core,
                      "batch256_e2e_rows_per_s": e2e, "checksum": float(out.astype(np.float64).sum())}), flush=True)
    sys.stdin.readline()


if __name__ == "__main__":
    main()
