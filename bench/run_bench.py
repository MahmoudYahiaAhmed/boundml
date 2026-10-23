"""Benchmark driver: BoundML vs PyTorch eager vs ONNX Runtime on the demo datasets.

Prerequisites (repository root):
    cargo build --release --workspace
    cargo build --release --manifest-path bench/baselines/Cargo.toml
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
BASE = ROOT / "bench" / "baselines" / "target" / "release"
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


def median_of(values: list):
    """Median of numbers; recurse into dicts; keep the first value of anything else."""
    first = values[0]
    if isinstance(first, dict):
        return {k: median_of([v[k] for v in values]) for k in first}
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return statistics.median(values)
    return first


def median_runs(cmd: list[str]) -> dict:
    runs = [run_worker(cmd) for _ in range(RUNS)]
    out = median_of(runs)
    out["runs"] = RUNS
    return out


def python_startup() -> float:
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        subprocess.run([PY, "-I", "-c", "pass"], check=True)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


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


def run_measured(cmd: list[str]) -> tuple[str, int]:
    """Run to completion; return stdout and the peak working set seen while polling.

    stdout is drained by a thread so a large output cannot fill the pipe and block
    the child while we poll it.
    """
    import threading

    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
    chunks: list[str] = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stdout.read()), daemon=True)
    reader.start()
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, peak_memory(proc))
        except psutil.Error:
            break
        time.sleep(0.005)
    proc.wait()
    reader.join()
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} exited {proc.returncode}")
    return "".join(chunks), peak


def train_bench(cfg: str, test: str) -> dict:
    keys = ("boundml_cli", "boundml_core_in_memory", "burn_ndarray", "pytorch_default", "pytorch_1_thread")
    res: dict = {k: [] for k in keys}
    for i in range(RUNS):
        ck = OUT / f"train-ck-{i}"
        shutil.rmtree(ck, ignore_errors=True)
        shutil.rmtree(pathlib.Path(str(ck) + ".prev"), ignore_errors=True)
        out, peak = run_measured([str(REL / f"boundml{EXE}"), "--json", "train", cfg, "--checkpoint", str(ck), "--quiet"])
        secs = json.loads(out)["result"]["seconds"]
        ev = subprocess.run([str(REL / f"boundml{EXE}"), "--json", "evaluate", str(ck), "--data", test], capture_output=True, text=True, cwd=ROOT, check=True)
        metrics = json.loads(ev.stdout)["result"]["model"]
        res["boundml_cli"].append({"train_seconds": secs, "peak_working_set_bytes": peak,
                                   "test": {k: metrics[k] for k in ("rmse", "accuracy") if k in metrics}})
        for key, cmd in (
            ("boundml_core_in_memory", [str(REL / f"train_core{EXE}"), cfg, test]),
            ("burn_ndarray", [str(BASE / f"train_burn{EXE}"), cfg, test]),
            ("pytorch_default", [PY, "bench/py_train.py", cfg, test, "default"]),
            ("pytorch_1_thread", [PY, "bench/py_train.py", cfg, test, "1"]),
        ):
            out, peak = run_measured(cmd)
            rec = json.loads(out.strip().splitlines()[-1])
            rec["peak_working_set_bytes"] = peak
            res[key].append(rec)
        shutil.rmtree(ck, ignore_errors=True)
        shutil.rmtree(pathlib.Path(str(ck) + ".prev"), ignore_errors=True)
    summary = {}
    for k, v in res.items():
        metric = next(iter(v[0]["test"]))
        summary[k] = {"median_train_seconds": statistics.median(x["train_seconds"] for x in v), "metric": metric,
                      "median_test_metric": statistics.median(x["test"][metric] for x in v),
                      "all_test_metric": [x["test"][metric] for x in v],
                      "median_peak_working_set_bytes": statistics.median(x["peak_working_set_bytes"] for x in v)}
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
        "tract_onnx": "0.23.9 (bench/baselines/Cargo.lock), concrete-shape optimized plans, single thread",
        "burn": "0.21.0 ndarray backend (bench/baselines/Cargo.lock)",
        "python_interpreter_startup_s": python_startup(),
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
            "tract": median_runs([str(BASE / f"infer_tract{EXE}"), str(onnx), str(pkg), test]),
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
