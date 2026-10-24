"""Benchmark driver: BoundML vs PyTorch eager and ONNX Runtime on the demo datasets.

Prerequisites (repository root):
    cargo build --release --workspace
    python -I datasets/prepare.py --download   # then boundml split, see docs/walkthroughs.md
    boundml train examples/<dataset>/mlp.toml --checkpoint out/<dataset>/mlp/ckpt
Run:
    python bench/run_bench.py [--runs 5]
    python bench/run_bench.py --training-only [--runs 5]   # rerun the training rows only
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


def monitor(pid: int) -> psutil.Process | None:
    """A psutil handle for `pid`, or None when the process is already gone (it may exit
    before monitoring starts); its exit status is still collected by the caller."""
    try:
        return psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None


def wait_with_peak(proc: subprocess.Popen, timeout: float, ps: psutil.Process | None = None) -> tuple[int | None, int]:
    """Wait for exit; return (exit code or None on timeout, peak memory in bytes).

    Windows: poll the peak working set until exit. POSIX: `wait4` returns the child's
    `ru_maxrss` (KiB on Linux, bytes on macOS), a true high-water mark. A process that
    exits before it can be sampled reports peak 0 (recorded, never invented).
    """
    deadline = time.perf_counter() + timeout
    peak = 0
    if WINDOWS and ps is None:
        ps = monitor(proc.pid)
    while time.perf_counter() < deadline:
        if WINDOWS:
            if ps is not None:
                try:
                    peak = max(peak, peak_memory(ps))
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    ps = None  # exited between polls; the exit code below still counts
                except psutil.AccessDenied as e:
                    raise BenchError(f"cannot read memory of pid {proc.pid}: {e}") from e
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


def settle(proc: subprocess.Popen, threads: list[threading.Thread]) -> None:
    """Make sure the child has exited and every pipe and reader thread is closed."""
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    for f in (proc.stdin, proc.stdout, proc.stderr):
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
    for t in threads:
        t.join(timeout=5)


class Worker:
    """A benchmark worker whose stdout lines are read with a timeout.

    Use as a context manager: leaving the block always settles the child process and
    closes its pipes, whether the run succeeded, failed validation or timed out.
    """

    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, cwd=ROOT)
        self.ps = monitor(self.proc.pid) if WINDOWS else None
        self.peak = 0
        self.lines: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        self.threads = [threading.Thread(target=self._pump, daemon=True),
                        threading.Thread(target=self._pump_stderr, daemon=True)]
        for t in self.threads:
            t.start()

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    def _pump_stderr(self) -> None:
        try:
            self.stderr.append(self.proc.stderr.read())
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        settle(self.proc, self.threads)

    def fail(self, why: str) -> BenchError:
        self.close()
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
        try:
            self.proc.stdin.write("\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise self.fail(f"could not signal the end of measurements ({e})") from None
        code, self.peak = wait_with_peak(self.proc, STEP_TIMEOUT_S, self.ps)
        if code is None:
            raise self.fail("did not exit after the results")
        if code != 0:
            raise self.fail("non-zero exit status")
        self.close()
        return code


# --- B1: parity validation -------------------------------------------------------------
PARITY_BATCHES = ("1", "7", "64")


def count_rows(csv_path: pathlib.Path) -> int:
    """Data rows in a CSV file (header excluded), the number of rows every runtime predicts."""
    import csv

    with csv_path.open(newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def output_cols(manifest: dict) -> int:
    """Output columns per row from the package's task: `classes` for multiclass, else 1."""
    task = manifest["task"]
    return int(task["classes"]) if task.get("kind") == "multiclass" else 1


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def load_parity(path: pathlib.Path, rows: int, cols: int, who: str) -> dict[str, list[float]]:
    """Read and validate a worker's parity file.

    Requires exactly the batch sizes 1, 7 and 64, `outputs_per_row == cols`, and for
    every batch exactly `rows * cols` finite numbers. Anything else is a BenchError.
    """
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise BenchError(f"parity {who}: unreadable parity file ({e})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("batches"), dict):
        raise BenchError(f"parity {who}: malformed parity file (no 'batches' object)")
    if doc.get("outputs_per_row") != cols:
        raise BenchError(f"parity {who}: outputs_per_row {doc.get('outputs_per_row')!r}, model has {cols}")
    batches = doc["batches"]
    if sorted(batches, key=str) != sorted(PARITY_BATCHES):
        raise BenchError(f"parity {who}: batch sizes {sorted(batches)}, expected {list(PARITY_BATCHES)}")
    expected = rows * cols
    if expected <= 0:
        raise BenchError(f"parity {who}: nothing to compare ({rows} rows x {cols} outputs)")
    out: dict[str, list[float]] = {}
    for bs in PARITY_BATCHES:
        vals = batches[bs]
        if not isinstance(vals, list) or len(vals) != expected:
            n = len(vals) if isinstance(vals, list) else type(vals).__name__
            raise BenchError(f"parity {who} batch {bs}: {n} outputs, expected {expected} ({rows} rows x {cols})")
        try:
            floats = [float(v) for v in vals]
        except (TypeError, ValueError):
            raise BenchError(f"parity {who} batch {bs}: non-numeric output") from None
        for i, v in enumerate(floats):
            if isinstance(vals[i], bool) or not _finite(v):
                raise BenchError(f"parity {who} batch {bs}: non-finite or invalid output {vals[i]!r} at index {i}")
        out[bs] = floats
    return out


def parity_error(got: dict[str, list[float]], ref: list[float], who: str) -> float:
    """Largest |a - b| over every row and batch size; raise on any mismatch.

    Every required batch size must be present, the reference must be non-empty and
    finite, and every candidate value must be finite and within the tolerance.
    """
    if not ref:
        raise BenchError(f"parity {who}: empty reference")
    for i, b in enumerate(ref):
        if not _finite(b):
            raise BenchError(f"parity {who}: non-finite reference {b} at index {i}")
    missing = [bs for bs in PARITY_BATCHES if bs not in got]
    if missing:
        raise BenchError(f"parity {who}: missing batch sizes {missing}")
    worst = 0.0
    for bs in PARITY_BATCHES:
        vals = got[bs]
        if len(vals) != len(ref):
            raise BenchError(f"parity {who} batch {bs}: {len(vals)} outputs, expected {len(ref)}")
        for i, (a, b) in enumerate(zip(vals, ref)):
            if not _finite(a):
                raise BenchError(f"parity {who} batch {bs}: non-finite output {a} at index {i}")
            err = abs(a - b)
            if not err <= ATOL + RTOL * abs(b):
                raise BenchError(f"parity {who} batch {bs}: output {i} = {a}, reference {b} (|diff| {err:.3g})")
            worst = max(worst, err)
    return worst


def run_inference(cmd: list[str], parity_path: pathlib.Path, ref: list[float] | None, who: str,
                  rows: int, cols: int) -> tuple[dict, list[float]]:
    """One worker run. Its record is returned only if its parity outputs validate; any
    failure raises BenchError after the worker has been settled."""
    parity_path.unlink(missing_ok=True)
    with Worker(cmd) as w:
        if w.expect("FIRST") != "FIRST":
            raise w.fail("unexpected first line")
        cold = time.perf_counter() - w.t0
        if w.expect("PARITY") != "PARITY":
            raise w.fail("unexpected parity line")
        try:
            got = load_parity(parity_path, rows, cols, who)
            ref = ref if ref is not None else got["1"]
            max_err = parity_error(got, ref, who)
        except BenchError as e:
            raise w.fail(str(e)) from None
        line = w.expect("results")
        try:
            result = json.loads(line)
        except ValueError:
            raise w.fail(f"malformed results line {line[:200]!r}") from None
        if not isinstance(result, dict):
            raise w.fail("results line is not a JSON object")
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
    the child while we poll it. The child and its pipes are settled on every path.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    chunks: list[str] = []
    errs: list[str] = []
    threads = [threading.Thread(target=lambda: chunks.append(proc.stdout.read()), daemon=True),
               threading.Thread(target=lambda: errs.append(proc.stderr.read()), daemon=True)]
    for t in threads:
        t.start()
    try:
        code, peak = wait_with_peak(proc, TRAIN_TIMEOUT_S)
        if code is None:
            raise BenchError(f"{' '.join(map(rel, cmd))}: timed out after {TRAIN_TIMEOUT_S} s")
        for t in threads:
            t.join(timeout=30)
    finally:
        settle(proc, threads)
    if code != 0:
        raise BenchError(f"{' '.join(map(rel, cmd))}: exit {code}; stderr: {''.join(errs).strip()[-2000:]}")
    return {"command": [rel(c) for c in cmd], "exit_status": code, "wall_s": time.perf_counter() - t0,
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


def training_only() -> None:
    """Rerun only the training comparison and replace it in the existing bench.json.

    Inference results, sizes and model identity are kept exactly as recorded; the
    rerun's own commit, date, command and uncommitted files are added under
    `environment.training_rerun`. Nothing is written unless every training run passes.
    """
    path = ROOT / "docs" / "results" / "bench.json"
    results = json.loads(path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    trained = {name: train_bench(cfg, test) for name, (cfg, _ck, test) in DATASETS.items()}
    for name, training in trained.items():
        results["datasets"][name]["training"] = training
    results["environment"]["training_rerun"] = {
        "date": time.strftime("%Y-%m-%d"),
        "command": f"python bench/run_bench.py --training-only --runs {RUNS}",
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "runs_per_measurement": RUNS,
        "reason": "PyTorch permutation tensors are now built before its training timer (as in train_core)",
    }
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path} (training only)")


def main() -> None:
    if "--training-only" in sys.argv:
        training_only()
        return
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
        rows = count_rows(ROOT / test)
        cols = output_cols(json.loads((pkg / "manifest.json").read_text()))
        workers = {
            "boundml": [str(REL / f"infer{EXE}"), str(pkg), test, str(parity)],
            "pytorch_eager": [PY, "bench/py_infer.py", "torch", str(pkg), test, str(parity)],
            "onnxruntime": [PY, "bench/py_infer.py", "ort", str(pkg), test, str(parity), str(onnx)],
        }
        ref = None
        for key, cmd in workers.items():  # untimed warm-up launch, parity still enforced
            _, ref = run_inference(cmd, parity, ref, f"{name}/{key} (warm-up)", rows, cols)
        records: dict = {k: [] for k in workers}
        for _ in range(RUNS):
            for key, cmd in workers.items():
                rec, ref = run_inference(cmd, parity, ref, f"{name}/{key}", rows, cols)
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
