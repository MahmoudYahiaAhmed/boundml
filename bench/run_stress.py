"""Sustained stress driver: runs the Rust `stress` worker for N seconds and samples
its working set / private bytes every second. Fails if memory keeps growing.

    python bench/run_stress.py [seconds]
Writes docs/results/stress.json.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def main() -> int:
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    work = ROOT / "out" / "stress"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    pkg = work / "model.boundml"
    boundml = ROOT / "target" / "release" / f"boundml{EXE}"
    # bench/stress.toml plans 4 sessions for the 4 concurrent prediction workers.
    ck = work / "ckpt"
    subprocess.run([str(boundml), "train", "bench/stress.toml", "--checkpoint", str(ck), "--quiet"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([str(boundml), "pack", str(ck), "--out", str(pkg)], cwd=ROOT, check=True, capture_output=True)
    cmd = [str(ROOT / "target" / "release" / f"stress{EXE}"), str(pkg), "datasets/breast_cancer/test.csv", "bench/stress.toml", str(work), str(secs)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=ROOT)
    ps = psutil.Process(proc.pid)
    samples, progress = [], []
    done = threading.Event()

    def sample() -> None:
        t0 = time.time()
        while not done.is_set():
            try:
                mi = ps.memory_info()
                samples.append({"t": round(time.time() - t0, 1), "working_set": mi.rss, "private": getattr(mi, "private", mi.vms), "threads": ps.num_threads()})
            except psutil.Error:
                break
            time.sleep(1.0)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    final = None
    for line in proc.stdout:
        rec = json.loads(line)
        progress.append(rec)
        if rec.get("done"):
            final = rec
            break
    peak = ps.memory_info()
    done.set()
    th.join()
    proc.stdin.write("\n")
    proc.stdin.flush()
    code = proc.wait(timeout=60)
    # Growth check: compare the median working set of the second and last quarters.
    q = max(1, len(samples) // 4)
    early = sorted(s["working_set"] for s in samples[q : 2 * q])
    late = sorted(s["working_set"] for s in samples[-q:])
    growth = late[len(late) // 2] / early[len(early) // 2] - 1 if early and late else 0.0
    threads = {s["threads"] for s in samples}
    ok = code == 0 and final is not None and growth < 0.10
    report = {
        "seconds": secs,
        "exit_code": code,
        "final": final,
        "peak_working_set_bytes": int(getattr(peak, "peak_wset", 0) or peak.rss),
        "working_set_growth_q2_to_q4": growth,
        "thread_counts_seen": sorted(threads),
        "samples": samples,
        "progress": progress,
        "passed": ok,
    }
    out = ROOT / "docs" / "results" / "stress.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "progress")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
