"""Build, package, checksum and smoke-test a BoundML release archive.

    python scripts/release.py [--target TRIPLE] [--skip-build] [--out DIR]

Steps:
1. `cargo build --release --locked -p boundml-cli [--target TRIPLE]`
2. Stage `boundml-<version>-<target>/` with the executable, README, licenses,
   CHANGELOG, third-party notices, the quickstart doc and the breast-cancer
   real-data example (CC BY 4.0, attribution included).
3. Write `.zip` (Windows) or `.tar.gz` (others) with sorted entries and fixed
   timestamps, plus `<archive>.sha256` and an updated `SHA256SUMS`.
4. Extract the *archive* into a fresh directory whose path contains spaces, and run
   the documented first-use steps with the extracted executable. Each step is timed,
   and the report is written to `<out>/smoke-<target>.json`. Any failure exits non-zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXED_TIME = (2026, 1, 1, 0, 0, 0)

INCLUDE = [
    ("README.md", "README.md"),
    ("LICENSE-MIT", "LICENSE-MIT"),
    ("LICENSE-APACHE", "LICENSE-APACHE"),
    ("CHANGELOG.md", "CHANGELOG.md"),
    ("THIRD_PARTY.md", "THIRD_PARTY.md"),
    ("docs/quickstart.md", "docs/quickstart.md"),
    ("docs/install.md", "docs/install.md"),
    ("docs/errors.md", "docs/errors.md"),
    ("docs/troubleshooting.md", "docs/troubleshooting.md"),
    ("examples/breast_cancer/linear.toml", "examples/breast_cancer/linear.toml"),
    ("examples/breast_cancer/mlp.toml", "examples/breast_cancer/mlp.toml"),
    ("datasets/README.md", "datasets/README.md"),
    ("datasets/breast_cancer/train.csv", "datasets/breast_cancer/train.csv"),
    ("datasets/breast_cancer/valid.csv", "datasets/breast_cancer/valid.csv"),
    ("datasets/breast_cancer/test.csv", "datasets/breast_cancer/test.csv"),
]


def version() -> str:
    meta = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT))
    return next(p["version"] for p in meta["packages"] if p["name"] == "boundml-cli")


def host_target() -> str:
    out = subprocess.check_output(["rustc", "-vV"], text=True)
    return next(l.split(": ")[1] for l in out.splitlines() if l.startswith("host: "))


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(target: str, host: str) -> pathlib.Path:
    cmd = ["cargo", "build", "--release", "--locked", "-p", "boundml-cli"]
    if target != host:
        cmd += ["--target", target]
    subprocess.run(cmd, cwd=ROOT, check=True)
    exe = "boundml.exe" if "windows" in target else "boundml"
    base = ROOT / "target" / (target if target != host else "") / "release"
    return base / exe


def archive(stage: pathlib.Path, out: pathlib.Path, windows: bool) -> pathlib.Path:
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    if windows:
        path = out / f"{stage.name}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                info = zipfile.ZipInfo(f"{stage.name}/{f.relative_to(stage).as_posix()}", FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if f.name.startswith("boundml") and f.suffix in ("", ".exe") else 0o644) << 16
                z.writestr(info, f.read_bytes())
    else:
        path = out / f"{stage.name}.tar.gz"
        with tarfile.open(path, "w:gz") as t:
            for f in files:
                info = t.gettarinfo(str(f), f"{stage.name}/{f.relative_to(stage).as_posix()}")
                info.mtime, info.uid, info.gid, info.uname, info.gname = 1767225600, 0, 0, "", ""
                info.mode = 0o755 if f.name == "boundml" else 0o644
                with f.open("rb") as fh:
                    t.addfile(info, fh)
    return path


def extract(path: pathlib.Path, dest: pathlib.Path) -> None:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    else:
        with tarfile.open(path) as t:
            t.extractall(dest, filter="data")


def smoke(path: pathlib.Path, stage_name: str, ver: str) -> dict:
    steps, ok = [], True
    with tempfile.TemporaryDirectory(prefix="boundml release smoke ") as d:
        root = pathlib.Path(d) / "My Apps"
        t0 = time.perf_counter()
        extract(path, root)
        steps.append({"step": "extract archive", "seconds": time.perf_counter() - t0, "exit": 0})
        home = root / stage_name
        exe = home / ("boundml.exe" if path.suffix == ".zip" else "boundml")
        work = pathlib.Path(d) / "first project"
        work.mkdir()

        def run(name: str, args: list[str], expect: int = 0, contains: str = "") -> None:
            nonlocal ok
            t = time.perf_counter()
            r = subprocess.run([str(exe), *args], cwd=work, capture_output=True, text=True)
            secs = time.perf_counter() - t
            out = r.stdout + r.stderr
            good = r.returncode == expect and contains in out
            ok &= good
            steps.append({"step": name, "command": "boundml " + " ".join(args), "seconds": secs,
                          "exit": r.returncode, "expected_exit": expect, "passed": good, "output": out.strip()[-1200:]})

        run("version", ["--version"], contains=f"boundml {ver}")
        run("init", ["init", "demo"])
        run("validate", ["validate", "demo/model.toml"])
        run("train", ["train", "demo/model.toml", "--checkpoint", "demo/ckpt", "--quiet"])
        run("evaluate", ["evaluate", "demo/ckpt", "--data", "demo/data/test.csv"], contains="constant baseline")
        run("pack", ["pack", "demo/ckpt", "--out", "demo/model.boundml"])
        run("verify", ["verify", "demo/model.boundml"], contains="PASS")
        run("predict", ["predict", "demo/model.boundml", "--data", "demo/data/test.csv", "--out", "demo/pred.csv", "--preview", "3"], contains="wrote 400 predictions")
        run("budget failure", ["plan", "demo/model.toml", "--mode", "infer", "--tensor-budget", "64KiB"], expect=4, contains="BML-C005")
        bc = home / "examples" / "breast_cancer" / "linear.toml"
        run("real data: train", ["train", str(bc), "--checkpoint", "bc/ckpt", "--quiet"])
        run("real data: evaluate", ["evaluate", "bc/ckpt", "--data", str(home / "datasets" / "breast_cancer" / "test.csv")], contains="accuracy")
    first_prediction = sum(s["seconds"] for s in steps if s["step"] in ("init", "train", "pack", "predict"))
    return {"archive": path.name, "passed": ok, "steps": steps,
            "seconds_init_to_first_prediction": first_prediction,
            "seconds_extract": steps[0]["seconds"]}


def main() -> None:
    args = sys.argv[1:]
    host = host_target()
    target = args[args.index("--target") + 1] if "--target" in args else host
    out = pathlib.Path(args[args.index("--out") + 1]) if "--out" in args else ROOT / "dist"
    out.mkdir(parents=True, exist_ok=True)
    ver = version()
    exe = ROOT / "target" / "release" / ("boundml.exe" if "windows" in target else "boundml")
    if "--skip-build" not in args:
        exe = build(target, host)
    stage_name = f"boundml-{ver}-{target}"
    with tempfile.TemporaryDirectory() as tmp:
        stage = pathlib.Path(tmp) / stage_name
        stage.mkdir()
        shutil.copy2(exe, stage / exe.name)
        for src, dst in INCLUDE:
            d = stage / dst
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, d)
        path = archive(stage, out, "windows" in target)
    digest = sha256(path)
    (out / f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n")
    sums = out / "SHA256SUMS"
    lines = [l for l in (sums.read_text().splitlines() if sums.exists() else []) if not l.endswith(path.name)]
    sums.write_text("\n".join(sorted(lines + [f"{digest}  {path.name}"])) + "\n")
    report = smoke(path, stage_name, ver)
    report.update({"sha256": digest, "bytes": path.stat().st_size, "target": target,
                   "built_on": platform.platform(), "version": ver})
    (out / f"smoke-{target}.json").write_text(json.dumps(report, indent=1))
    for s in report["steps"]:
        flag = "ok " if s.get("passed", True) else "FAIL"
        print(f"{flag} {s['seconds']:7.3f}s  {s['step']}")
    print(f"{path.name}  {path.stat().st_size} bytes  sha256 {digest}")
    print(f"init -> first prediction: {report['seconds_init_to_first_prediction']:.2f}s; smoke {'PASSED' if report['passed'] else 'FAILED'}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
