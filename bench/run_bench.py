"""Benchmark driver: BoundML vs PyTorch eager and ONNX Runtime on the demo datasets.

Prerequisites (repository root):
    cargo build --release --workspace
    python -I datasets/prepare.py --download   # then boundml split, see docs/walkthroughs.md
    boundml train examples/<dataset>/mlp.toml --checkpoint out/<dataset>/mlp/ckpt
Run:
    python bench/run_bench.py [--runs 5]
Writes docs/results/bench.json. Methodology: docs/benchmarks.md.

Every inference worker first writes its outputs for every test row at batch sizes 1,
7 and 64 (outside any timed region). The driver checks them elementwise against
BoundML's batch-1 outputs (finite, |a - b| <= ATOL + RTOL * |b|) and aborts the whole
benchmark, without writing results, on any mismatch. Worker exit codes are checked and
every worker step has a timeout, after which the worker is killed and the run aborts.
Every run's record (command, exit status, timings, memory, parity error) is kept.

Each runtime gets one untimed warm-up launch per dataset (file-system cache, first
DLL load) before the measured runs. Peak memory is the OS high-water mark: the peak
working set on Windows, `ru_maxrss` from `wait4` on Linux and macOS.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time

import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
REL = ROOT / "target" / "release"
OUT = ROOT / "out" / "bench"
PY = sys.executable
RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 5
ATOL, RTOL = 1e-5, 1e-4
STEP_TIMEOUT_S = 300
TRAIN_TIMEOUT_S = 900
WINDOWS = os.name == "nt"
MEMORY_METRIC = ("peak_wset: OS-reported peak working set (high-water mark) of the worker process"
                 if WINDOWS else "ru_maxrss: OS-reported peak resident set size (high-water mark) from wait4")
INFER_MEMORY = TRAIN_MEMORY = MEMORY_METRIC

DATASETS = {
    "breast_cancer": ("examples/breast_cancer/mlp.toml", "out/breast_cancer/mlp/ckpt", "datasets/breast_cancer/test.csv"),
    "wine_quality": ("examples/wine_quality/mlp.toml", "out/wine_quality/mlp/ckpt", "datasets/wine_red/test.csv"),
}


class BenchError(RuntimeError):
    pass


def rel(arg: str) -> str:
    """Command argument as recorded: repository-relative, interpreter as `python`."""
    if arg == PY:
        return "python"
    try:
        return pathlib.Path(arg).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return arg


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def peak_memory(p: psutil.Process) -> int:
    """Windows only: the peak working set is a high-water mark, readable while the process lives."""
    return int(p.memory_info().peak_wset)


def wait_with_peak(proc: subprocess.Popen, timeout: float, ps: psutil.Process | None = None) -> tuple[int | None, int]:
    """Wait for exit; return (exit code or None on timeout, peak memory in bytes).

    Windows: poll the peak working set until exit. POSIX: `wait4` returns the child's
    `ru_maxrss` (KiB on Linux, bytes on macOS), a true high-water mark.
    """
    deadline = time.perf_counter() + timeout
    peak = 0
    while time.perf_counter() < deadline:
        if WINDOWS:
            try:
                peak = max(peak, peak_memory(ps or psutil.Process(proc.pid)))
            except psutil.Error:
                pass
            if proc.poll() is not None:
                return proc.returncode, peak
        else:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(status)
                scale = 1 if sys.platform == "darwin" else 1024
                return proc.returncode, int(usage.ru_maxrss) * scale
        time.sleep(0.005)
    return None, peak


class Worker:
    """A benchmark worker whose stdout lines are read with a timeout."""

    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, cwd=ROOT)
        self.ps = psutil.Process(self.proc.pid) if WINDOWS else None
        self.peak = 0
        self.lines: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=lambda: self.stderr.append(self.proc.stderr.read()), daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def fail(self, why: str) -> BenchError:
        self.proc.kill()
        self.proc.wait()
        err = "".join(self.stderr).strip()[-2000:]
        return BenchError(f"{' '.join(map(rel, self.cmd))}: {why} (exit {self.proc.returncode}); stderr: {err}")

    def expect(self, what: str) -> str:
        try:
            line = self.lines.get(timeout=STEP_TIMEOUT_S)
        except queue.Empty:
            raise self.fail(f"timed out after {STEP_TIMEOUT_S} s waiting for {what}") from None
        if line is None:
            raise self.fail(f"exited before {what}")
        return line.strip()

    def finish(self) -> int:
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()
        code, self.peak = wait_with_peak(self.proc, STEP_TIMEOUT_S, self.ps)
        if code is None:
            raise self.fail("did not exit after the results")
        if code != 0:
            raise self.fail("non-zero exit status")
        return code


def load_parity(path: pathlib.Path) -> dict[str, list[float]]:
    doc = json.loads(path.read_text())
    return {k: [float(v) for v in vals] for k, vals in doc["batches"].items()}


def parity_error(got: dict[str, list[float]], ref: list[float], who: str) -> float:
    """Largest |a - b| over every row and batch size; raise on any mismatch."""
    worst = 0.0
    for bs, vals in got.items():
        if len(vals) != len(ref):
            raise BenchError(f"parity {who} batch {bs}: {len(vals)} outputs, expected {len(ref)}")
        for i, (a, b) in enumerate(zip(vals, ref)):
            if not (a == a and abs(a) != float("inf")):
                raise BenchError(f"parity {who} batch {bs}: non-finite output {a} at index {i}")
            err = abs(a - b)
            if not err <= ATOL + RTOL * abs(b):
                raise BenchError(f"parity {who} batch {bs}: output {i} = {a}, reference {b} (|diff| {err:.3g})")
            worst = max(worst, err)
    return worst


def run_inference(cmd: list[str], parity_path: pathlib.Path, ref: list[float] | None, who: str) -> tuple[dict, list[float]]:
    parity_path.unlink(missing_ok=True)
    w = Worker(cmd)
    if w.expect("FIRST") != "FIRST":
        raise w.fail("unexpected first line")
    cold = time.perf_counter() - w.t0
    if w.expect("PARITY") != "PARITY":
        raise w.fail("unexpected parity line")
    got = load_parity(parity_path)
    ref = ref if ref is not None else got["1"]
    try:
        max_err = parity_error(got, ref, who)
    except BenchError:
        w.proc.kill()
        raise
    result = json.loads(w.expect("results"))
    result["cold_start_s"] = cold
    code = w.finish()
    result["peak_memory_bytes"] = w.peak
    record = {"command": [rel(c) for c in cmd], "exit_status": code, "wall_s": time.perf_counter() - w.t0,
              "parity_max_abs_error": max_err, "parity_outputs_checked": len(ref) * len(got), **result}
    return record, ref


def median_of(values: list):
    """Median of numbers; recurse into dicts; keep the first value of anything else."""
    first = values[0]
    if isinstance(first, dict):
        return {k: median_of([v[k] for v in values]) for k in first}
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return statistics.median(values)
    return first


def spread_of(values: list):
    """[min, max] of numbers, recursing into dicts; None for anything else."""
    first = values[0]
    if isinstance(first, dict):
        return {k: spread_of([v[k] for v in values]) for k in first}
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return [min(values), max(values)]
    return None


def summarize(records: list[dict]) -> dict:
    keys = [k for k in records[0] if k not in ("command", "exit_status")]
    trimmed = [{k: r[k] for k in keys} for r in records]
    out = median_of(trimmed)
    out["spread"] = spread_of(trimmed)
    out["command"] = records[0]["command"]
    out["runs"] = records
    return out


def python_startup() -> list[float]:
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        subprocess.run([PY, "-I", "-c", "pass"], check=True)
        times.append(time.perf_counter() - t)
    return times


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


def run_measured(cmd: list[str]) -> dict:
    """Run to completion with a timeout; return a record with stdout, exit status and peak memory.

    stdout is drained by a thread so a large output cannot fill the pipe and block
    the child while we poll it.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    chunks: list[str] = []
    errs: list[str] = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stdout.read()), daemon=True)
    ereader = threading.Thread(target=lambda: errs.append(proc.stderr.read()), daemon=True)
    reader.start()
    ereader.start()
    code, peak = wait_with_peak(proc, TRAIN_TIMEOUT_S)
    if code is None:
        proc.kill()
        proc.wait()
        raise BenchError(f"{' '.join(map(rel, cmd))}: timed out after {TRAIN_TIMEOUT_S} s")
    reader.join()
    ereader.join()
    if proc.returncode != 0:
        raise BenchError(f"{' '.join(map(rel, cmd))}: exit {proc.returncode}; stderr: {''.join(errs).strip()[-2000:]}")
    return {"command": [rel(c) for c in cmd], "exit_status": proc.returncode, "wall_s": time.perf_counter() - t0,
            "stdout": "".join(chunks), "peak_memory_bytes": peak}


def train_bench(cfg: str, test: str) -> dict:
    keys = ("boundml_cli", "boundml_core_in_memory", "pytorch_default", "pytorch_1_thread")
    res: dict = {k: [] for k in keys}
    control_hashes = set()
    for i in range(RUNS):
        ck = OUT / f"train-ck-{i}"
        for p in (ck, pathlib.Path(str(ck) + ".prev")):
            shutil.rmtree(p, ignore_errors=True)
        rec = run_measured([str(REL / f"boundml{EXE}"), "--json", "train", cfg, "--checkpoint", str(ck), "--quiet"])
        secs = json.loads(rec.pop("stdout"))["result"]["seconds"]
        ev = subprocess.run([str(REL / f"boundml{EXE}"), "--json", "evaluate", str(ck), "--data", test],
                            capture_output=True, text=True, cwd=ROOT, timeout=STEP_TIMEOUT_S)
        if ev.returncode != 0:
            raise BenchError(f"boundml evaluate exited {ev.returncode}: {ev.stderr.strip()[-2000:]}")
        metrics = json.loads(ev.stdout)["result"]["model"]
        res["boundml_cli"].append({**rec, "train_seconds": secs,
                                   "test": {k: metrics[k] for k in ("rmse", "accuracy") if k in metrics}})
        control = OUT / f"control-{i}.json"
        for key, cmd in (
            ("boundml_core_in_memory", [str(REL / f"train_core{EXE}"), cfg, test, str(control)]),
            ("pytorch_default", [PY, "bench/py_train.py", cfg, test, "default", str(control)]),
            ("pytorch_1_thread", [PY, "bench/py_train.py", cfg, test, "1", str(control)]),
        ):
            rec = run_measured(cmd)
            out = json.loads(rec.pop("stdout").strip().splitlines()[-1])
            res[key].append({**rec, **out})
        control_hashes.add(sha256(control))
        for p in (ck, pathlib.Path(str(ck) + ".prev")):
            shutil.rmtree(p, ignore_errors=True)
    summary: dict = {"control_file_sha256": sorted(control_hashes)}
    for k, v in res.items():
        metric = next(iter(v[0]["test"]))
        secs = [x["train_seconds"] for x in v]
        mem = [x["peak_memory_bytes"] for x in v]
        summary[k] = {"median_train_seconds": statistics.median(secs), "train_seconds_range": [min(secs), max(secs)],
                      "metric": metric, "median_test_metric": statistics.median(x["test"][metric] for x in v),
                      "all_test_metric": [x["test"][metric] for x in v],
                      "median_peak_memory_bytes": statistics.median(mem), "peak_memory_range": [min(mem), max(mem)],
                      "memory_metric": TRAIN_MEMORY, "command": v[0]["command"], "runs": v}
    return summary


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import onnxruntime
    import torch

    startup = python_startup()
    env = {
        "date": time.strftime("%Y-%m-%d"),
        "command": f"python bench/run_bench.py --runs {RUNS}",
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
        "boundml_build": "cargo --release (lto=thin, codegen-units=1), default target CPU",
        "torch_threads_default": torch.get_num_threads(),
        "python_interpreter_startup_s": statistics.median(startup),
        "python_interpreter_startup_runs_s": startup,
        "runs_per_measurement": RUNS,
        "warm_up_launches_per_runtime": 1,
        "parity": {"batch_sizes": [1, 7, 64], "atol": ATOL, "rtol": RTOL, "reference": "BoundML, batch 1",
                   "rule": "every output finite and |a - b| <= atol + rtol * |b|"},
        "inference_memory_metric": INFER_MEMORY,
        "training_memory_metric": TRAIN_MEMORY,
    }
    results: dict = {"environment": env, "datasets": {}}
    for name, (cfg, ck, test) in DATASETS.items():
        pkg = OUT / f"{name}.boundml"
        shutil.rmtree(pkg, ignore_errors=True)
        subprocess.run([str(REL / f"boundml{EXE}"), "pack", ck, "--out", str(pkg)], check=True, cwd=ROOT,
                       capture_output=True, timeout=STEP_TIMEOUT_S)
        onnx = OUT / f"{name}.onnx"
        export_onnx(pkg, onnx)
        identity = {"package_manifest_sha256": sha256(pkg / "manifest.json"),
                    "package_weights_sha256": sha256(pkg / "weights.safetensors"),
                    "onnx_sha256": sha256(onnx), "test_csv_sha256": sha256(ROOT / test), "config": cfg}
        parity = OUT / f"{name}.parity.json"
        workers = {
            "boundml": [str(REL / f"infer{EXE}"), str(pkg), test, str(parity)],
            "pytorch_eager": [PY, "bench/py_infer.py", "torch", str(pkg), test, str(parity)],
            "onnxruntime": [PY, "bench/py_infer.py", "ort", str(pkg), test, str(parity), str(onnx)],
        }
        ref = None
        for key, cmd in workers.items():  # untimed warm-up launch, parity still enforced
            _, ref = run_inference(cmd, parity, ref, f"{name}/{key} (warm-up)")
        records: dict = {k: [] for k in workers}
        for _ in range(RUNS):
            for key, cmd in workers.items():
                rec, ref = run_inference(cmd, parity, ref, f"{name}/{key}")
                records[key].append(rec)
        inf = {k: {**summarize(v), "memory_metric": INFER_MEMORY} for k, v in records.items()}
        sizes = {
            "boundml_cli_binary_bytes": (REL / f"boundml{EXE}").stat().st_size,
            "boundml_package_bytes": dir_size(pkg),
            "onnx_model_bytes": onnx.stat().st_size,
            "torch_install_bytes": dir_size(pathlib.Path(torch.__file__).parent),
            "onnxruntime_install_bytes": dir_size(pathlib.Path(onnxruntime.__file__).parent),
        }
        results["datasets"][name] = {"model_identity": identity, "inference": inf, "sizes": sizes,
                                     "training": train_bench(cfg, test)}
        print(f"{name}: done", flush=True)
    path = ROOT / "docs" / "results" / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except BenchError as e:
        print(f"benchmark aborted, no results written: {e}", file=sys.stderr)
        sys.exit(1)
