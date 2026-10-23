"""Benchmark driver: BoundML vs PyTorch eager vs ONNX Runtime on the demo datasets.

Prerequisites (repository root):
    cargo build --release --workspace
    python -I datasets/prepare.py --download   # then boundml split, see docs/walkthroughs.md
    boundml train examples/<dataset>/mlp.toml --checkpoint out/<dataset>/mlp/ckpt
Run:
    python bench/run_bench.py [--runs 5]
Writes docs/results/bench.json. Methodology: docs/benchmarks.md.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import time

import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
REL = ROOT / "target" / "release"
OUT = ROOT / "out" / "bench"
PY = sys.executable
RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 5

DATASETS = {
    "breast_cancer": ("examples/breast_cancer/mlp.toml", "out/breast_cancer/mlp/ckpt", "datasets/breast_cancer/test.csv"),
    "wine_quality": ("examples/wine_quality/mlp.toml", "out/wine_quality/mlp/ckpt", "datasets/wine_red/test.csv"),
}


def peak_memory(p: psutil.Process) -> int:
    info = p.memory_info()
    return int(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0))


def run_worker(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
    ps = psutil.Process(proc.pid)
    first = proc.stdout.readline()
    cold = time.perf_counter() - t0
    if first.strip() != "FIRST":
        raise RuntimeError(f"{cmd}: unexpected output {first!r}")
    result = json.loads(proc.stdout.readline())
    result["cold_start_s"] = cold
    result["peak_working_set_bytes"] = peak_memory(ps)
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait(timeout=60)
    return result


def median_runs(cmd: list[str]) -> dict:
    runs = [run_worker(cmd) for _ in range(RUNS)]
    med = lambda f: statistics.median(f(r) for r in runs)  # noqa: E731
    return {
        "runtime": runs[0]["runtime"],
        "runs": RUNS,
        "cold_start_s": med(lambda r: r["cold_start_s"]),
        "peak_working_set_bytes": med(lambda r: r["peak_working_set_bytes"]),
        "latency_us": {k: med(lambda r, k=k: r["latency_us"][k]) for k in ("p50", "p95", "p99")},
        "batch256_rows_per_s": med(lambda r: r["batch256_rows_per_s"]),
        "checksum": runs[0]["checksum"],
    }


def export_onnx(pkg: pathlib.Path, path: pathlib.Path) -> None:
    import torch

    sys.path.insert(0, str(ROOT / "bench"))
    from pymodel import torch_module

    model, m = torch_module(pkg)
    x = torch.zeros(4, len(m["schema"]["features"]))
    torch.onnx.export(model, (x,), str(path), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, dynamo=False)


def dir_size(p: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def train_bench(cfg: str, test: str) -> dict:
    res: dict = {"boundml": [], "pytorch_default": [], "pytorch_1_thread": []}
    for i in range(RUNS):
        ck = OUT / f"train-ck-{i}"
        shutil.rmtree(ck, ignore_errors=True)
        r = subprocess.run([str(REL / f"boundml{EXE}"), "--json", "train", cfg, "--checkpoint", str(ck), "--quiet"], capture_output=True, text=True, cwd=ROOT, check=True)
        secs = json.loads(r.stdout)["result"]["seconds"]
        ev = subprocess.run([str(REL / f"boundml{EXE}"), "--json", "evaluate", str(ck), "--data", test], capture_output=True, text=True, cwd=ROOT, check=True)
        metrics = json.loads(ev.stdout)["result"]["model"]
        res["boundml"].append({"train_seconds": secs, "test": {k: metrics[k] for k in ("rmse", "accuracy") if k in metrics}})
        for key, threads in (("pytorch_default", "default"), ("pytorch_1_thread", "1")):
            r = subprocess.run([PY, "bench/py_train.py", cfg, test, threads], capture_output=True, text=True, cwd=ROOT, check=True)
            res[key].append(json.loads(r.stdout.strip().splitlines()[-1]))
        shutil.rmtree(ck, ignore_errors=True)
        shutil.rmtree(pathlib.Path(str(ck) + ".prev"), ignore_errors=True)
    summary = {}
    for k, v in res.items():
        metric = next(iter(v[0]["test"]))
        summary[k] = {"median_train_seconds": statistics.median(x["train_seconds"] for x in v), "metric": metric,
                      "median_test_metric": statistics.median(x["test"][metric] for x in v),
                      "all_test_metric": [x["test"][metric] for x in v]}
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import onnxruntime
    import torch

    env = {
        "date": time.strftime("%Y-%m-%d"),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
        "boundml_build": "cargo --release (lto=thin, codegen-units=1), default target CPU",
        "torch_threads_default": torch.get_num_threads(),
        "runs_per_measurement": RUNS,
    }
    results: dict = {"environment": env, "datasets": {}}
    for name, (cfg, ck, test) in DATASETS.items():
        pkg = OUT / f"{name}.boundml"
        subprocess.run([str(REL / f"boundml{EXE}"), "pack", ck, "--out", str(pkg)], check=True, cwd=ROOT, capture_output=True)
        onnx = OUT / f"{name}.onnx"
        export_onnx(pkg, onnx)
        inf = {
            "boundml": median_runs([str(REL / f"infer{EXE}"), str(pkg), test]),
            "pytorch_eager": median_runs([PY, "bench/py_infer.py", "torch", str(pkg), test]),
            "onnxruntime": median_runs([PY, "bench/py_infer.py", "ort", str(pkg), test, str(onnx)]),
        }
        sizes = {
            "boundml_cli_binary_bytes": (REL / f"boundml{EXE}").stat().st_size,
            "boundml_package_bytes": dir_size(pkg),
            "onnx_model_bytes": onnx.stat().st_size,
            "torch_install_bytes": dir_size(pathlib.Path(torch.__file__).parent),
            "onnxruntime_install_bytes": dir_size(pathlib.Path(onnxruntime.__file__).parent),
        }
        results["datasets"][name] = {"inference": inf, "sizes": sizes, "training": train_bench(cfg, test)}
        print(json.dumps(results["datasets"][name], indent=1))
    path = ROOT / "docs" / "results" / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
