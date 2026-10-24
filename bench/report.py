"""Render docs/results/bench.json into docs/benchmarks.md and the README evidence table.

Replaces the text between these marker pairs:
  docs/benchmarks.md  <!-- BENCH:BEGIN/END -->, <!-- TARGETS:BEGIN/END -->
  README.md           <!-- README-BENCH:BEGIN/END -->

    python bench/report.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs" / "results" / "bench.json").read_text(encoding="utf-8"))
DOC = ROOT / "docs" / "benchmarks.md"
NAMES = {"boundml": "BoundML", "pytorch_eager": "PyTorch eager (Python)",
         "onnxruntime": "ONNX Runtime (Python API)"}
TRAIN = {"boundml_cli": "`boundml train` CLI (own init and streamed batch order; CSV re-read every epoch + validation)",
         "boundml_core_in_memory": "BoundML core, in-memory (writes the control file)",
         "pytorch_default": "PyTorch, default threads, controlled",
         "pytorch_1_thread": "PyTorch, 1 thread, controlled"}
TITLES = {"breast_cancer": "Breast cancer (MLP 30-16-1, sigmoid output)", "wine_quality": "Wine quality (MLP 11-32-16-1, regression)"}


def mib(b: float) -> str:
    return f"{b / 2**20:.1f} MiB"


def ms(s: float) -> str:
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


def us(q: dict) -> str:
    return f"{q['p50']:.1f} / {q['p95']:.1f} / {q['p99']:.1f}"


def mrows(v: float) -> str:
    return f"{v / 1e6:.2f} M"


# Commit IDs recorded before the 2026-10-24 history rewrite (which removed local paths
# from old files and changed no code) and their IDs in the current history.
REWRITTEN = {"79ef2ab344a4dfb2096af9c57699ac648e25e67e": "31f5e25af649a54035930c0e35deec778fdd932b"}


def commit_ref(sha: str) -> str:
    if sha in REWRITTEN:
        return f"`{REWRITTEN[sha][:7]}` (recorded as `{sha[:7]}`, its ID before the 2026-10-24 history rewrite)"
    return f"`{sha[:7]}`"


def uncommitted(env: dict) -> str:
    files = [f.split(maxsplit=1)[-1] for f in env.get("git_uncommitted_files", [])]
    return f" (uncommitted changes to: {', '.join(f'`{f}`' for f in files)})" if files else " (clean tree)"


def rng(r: dict, key: str, fmt) -> str:
    """Median with the min–max over runs."""
    lo, hi = r["spread"][key]
    return f"{fmt(r[key])} ({fmt(lo)}–{fmt(hi)})"


def render() -> str:
    env = DATA["environment"]
    out = [f"Measured {env['date']} on {env['cpu']} ({env['logical_cpus']} logical CPUs), {env['os']}; "
           f"{env['runs_per_measurement']} runs per measurement; medians shown, with the min–max over runs in "
           f"parentheses where given; one untimed warm-up launch per runtime precedes them. Commit {commit_ref(env['git_commit'])}"
           f"{uncommitted(env)}. "
           f"Python interpreter start-up (`python -I -c pass`): **{ms(env['python_interpreter_startup_s'])}**.", "",
           f"Every reported run's outputs matched BoundML's for every test row at batch sizes "
           f"{', '.join(map(str, env['parity']['batch_sizes']))} ({env['parity']['rule']}, atol {env['parity']['atol']:g}, "
           f"rtol {env['parity']['rtol']:g}), validated before its timings were accepted. Memory is `{env['inference_memory_metric'].split(':')[0]}`: "
           f"{env['inference_memory_metric'].split(': ', 1)[1]}.", ""]
    for name, d in DATA["datasets"].items():
        inf = d["inference"]
        out += [f"### {TITLES.get(name, name)}", "", "**Start-up and memory**", "",
                "| Runtime | Process start → first prediction | Framework import | Model load | Peak process memory | Planned tensor bytes | Max parity error |",
                "|---|---|---|---|---|---|---|"]
        for k, r in inf.items():
            imp = ms(r["import_s"]) if "import_s" in r else "—"
            planned = f"{r['planned_tensor_bytes']:,} B" if "planned_tensor_bytes" in r else "not reported"
            out.append(f"| {NAMES[k]} | {rng(r, 'cold_start_s', ms)} | {imp} | {ms(r['load_s'])} | {rng(r, 'peak_memory_bytes', mib)} | "
                       f"{planned} | {r['spread']['parity_max_abs_error'][1]:.1e} |")
        out += ["", "**Warm inference** (µs per call, p50 / p95 / p99; throughput in rows/s for one 256-row batch)", "",
                "| Runtime | Threads | Preprocessing alone (1 row) | Core inference, 1 row | Preprocess + infer, 1 row | Core, batch 256 | Preprocess + infer, batch 256 |",
                "|---|---|---|---|---|---|---|"]
        for k, r in inf.items():
            out.append(f"| {NAMES[k]} | {r['threads']} | {us(r['preprocess_us_per_row'])} | {us(r['latency_core_us'])} | {us(r['latency_e2e_us'])} | "
                       f"{rng(r, 'batch256_core_rows_per_s', mrows)} | {rng(r, 'batch256_e2e_rows_per_s', mrows)} |")
        copies = "; ".join(f"{NAMES[k]}: {r['unavoidable_copies']}" for k, r in inf.items() if "unavoidable_copies" in r)
        out += ["", f"Input copies inside the timed calls: {copies}."]
        tr = d["training"]
        rerun = env.get("training_rerun")
        if rerun:
            out += ["", f"Training rows re-measured on {rerun['date']} at commit `{rerun['git_commit'][:7]}` "
                    f"({rerun['runs_per_measurement']} runs, `{rerun['command']}`); inference above is from the run described at the top."]
        out += ["", "**Training** (test split). The controlled rows load one control file written by BoundML's in-memory "
                "trainer: the same standardized data, initial weights and per-epoch batch order. The CLI row is a separate "
                "end-to-end measurement with its own streamed batch order.", "",
                "| Trainer | Train time (min–max) | Test metric (all runs) | Peak process memory |", "|---|---|---|---|"]
        for k, r in tr.items():
            if k not in TRAIN:
                continue
            lo, hi = r["train_seconds_range"]
            metrics = ", ".join(f"{v:.4f}" for v in sorted(set(r["all_test_metric"])))
            out.append(f"| {TRAIN[k]} | {ms(r['median_train_seconds'])} ({ms(lo)}–{ms(hi)}) | {r['metric']} {metrics} | "
                       f"{mib(r['median_peak_memory_bytes'])} |")
        s = d["sizes"]
        out += ["", f"Sizes: `boundml` binary {s['boundml_cli_binary_bytes'] / 1e6:.1f} MB; package {s['boundml_package_bytes'] / 1e3:.1f} KB "
                f"(includes schema, preprocessing, self-tests); ONNX file {s['onnx_model_bytes'] / 1e3:.1f} KB; "
                f"installed PyTorch {s['torch_install_bytes'] / 1e6:.0f} MB, ONNX Runtime {s['onnxruntime_install_bytes'] / 1e6:.0f} MB.", ""]
    return "\n".join(out)


def readme_table() -> str:
    inf = DATA["datasets"]["breast_cancer"]["inference"]
    order = ["boundml", "onnxruntime", "pytorch_eager"]
    rows = [
        ("Process start → first prediction", lambda r: r["cold_start_s"], ms, True),
        ("Peak process memory (OS high-water mark)", lambda r: r["peak_memory_bytes"], mib, True),
        ("1-row latency, p50, incl. preprocessing", lambda r: r["latency_e2e_us"]["p50"], lambda v: f"{v:.1f} µs", True),
        ("Batch-256 throughput, model only", lambda r: r["batch256_core_rows_per_s"], lambda v: f"{v / 1e6:.1f} M rows/s", False),
        ("Batch-256 throughput, incl. preprocessing", lambda r: r["batch256_e2e_rows_per_s"], lambda v: f"{v / 1e6:.1f} M rows/s", False),
    ]
    out = ["| 30-feature MLP, breast-cancer data | BoundML | ONNX Runtime (Python) | PyTorch (Python) |",
           "|---|---|---|---|"]
    for label, f, fmt, low in rows:
        best = (min if low else max)(order, key=lambda k: f(inf[k]))
        cells = [f"**{fmt(f(inf[k]))}**" if k == best else fmt(f(inf[k])) for k in order]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def targets() -> str:
    out = ["| ID | Dataset | Threshold | Measured | Verdict |", "|---|---|---|---|---|"]

    def verdict(ok: bool) -> str:
        return "**met**" if ok else "**not met**"

    for name, d in DATA["datasets"].items():
        i, t = d["inference"], d["training"]
        b, p, o = i["boundml"], i["pytorch_eager"], i["onnxruntime"]
        r1 = b["peak_memory_bytes"] / p["peak_memory_bytes"]
        r2 = b["cold_start_s"] / p["cold_start_s"]
        r3 = b["batch256_e2e_rows_per_s"] / o["batch256_e2e_rows_per_s"]
        r3c = b["batch256_core_rows_per_s"] / o["batch256_core_rows_per_s"]
        r4 = t["boundml_cli"]["median_train_seconds"] / t["pytorch_1_thread"]["median_train_seconds"]
        m = t["boundml_cli"]["metric"]
        qb, qp = t["boundml_cli"]["median_test_metric"], t["pytorch_default"]["median_test_metric"]
        if m == "accuracy":
            q, ok5, thr5 = f"{(qb - qp) * 100:+.1f} pp accuracy", abs(qb - qp) <= 0.02, "within 2 pp"
        else:
            q, ok5, thr5 = f"RMSE {qb:.4f} vs {qp:.4f} ({(qb / qp - 1) * 100:+.1f}%)", abs(qb / qp - 1) <= 0.10, "within 10% RMSE"
        ds = TITLES.get(name, name).split(" (")[0]
        both = (f"incl. preprocessing {r3 * 100:.0f}% ({verdict(r3 >= 0.50)}); "
                f"model only {r3c * 100:.0f}% ({verdict(r3c >= 0.50)})")
        out += [
            f"| T1 memory | {ds} | ≤ 20% of PyTorch peak | {r1 * 100:.1f}% | {verdict(r1 <= 0.20)} |",
            f"| T2 cold start | {ds} | ≤ 10% of PyTorch | {r2 * 100:.1f}% | {verdict(r2 <= 0.10)} |",
            f"| T3 throughput | {ds} | ≥ 50% of ORT batch-256 rows/s (the pre-declared text does not say whether "
            f"preprocessing is included, so both readings are shown) | {both} | "
            f"{'**met under both readings**' if min(r3, r3c) >= 0.50 else '**met** only if preprocessing is included' if r3 >= 0.50 else '**not met**'} |",
            f"| T4 training | {ds} | `boundml train` ≤ 2× PyTorch 1-thread | {r4:.2f}× | {verdict(r4 <= 2.0)} |",
            f"| T5 quality | {ds} | {thr5} of PyTorch | {q} | {verdict(ok5)} |",
        ]
    return "\n".join(out)


def splice(path: pathlib.Path, a: str, b: str, body: str) -> None:
    doc = path.read_text(encoding="utf-8")
    head, rest = doc.split(a, 1)
    _, tail = rest.split(b, 1)
    path.write_text(f"{head}{a}\n{body}\n{b}{tail}", encoding="utf-8")
    print(f"updated {path.relative_to(ROOT)} {a}")


def main() -> None:
    splice(DOC, "<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->", render())
    splice(DOC, "<!-- TARGETS:BEGIN -->", "<!-- TARGETS:END -->", targets())
    splice(ROOT / "README.md", "<!-- README-BENCH:BEGIN -->", "<!-- README-BENCH:END -->", readme_table())


if __name__ == "__main__":
    main()
