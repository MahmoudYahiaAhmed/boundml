"""Record the README demo from the real `boundml` binary.

Runs each command below in a fresh temporary directory, captures its real stdout/
stderr, exit code and wall time, and writes:

* assets/demo.cast - asciinema v2 recording (real output, real relative timing)
* assets/demo.svg  - self-contained animated SVG of the same recording

Presentation choices (stated in the README caption): commands appear at a fixed
typing speed, and pauses are clamped to 0.15-1.2 s so the loop stays short. Output
text is never edited, except that long lines wrap at the terminal width.

    python scripts/record_demo.py [path/to/boundml]
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "target" / "release" / ("boundml.exe" if os.name == "nt" else "boundml")
COMMANDS = [
    "boundml --version",
    "boundml init demo",
    "boundml train demo/model.toml --checkpoint demo/ckpt --quiet",
    "boundml evaluate demo/ckpt --data demo/data/test.csv",
    "boundml pack demo/ckpt --out demo/model.boundml",
    "boundml predict demo/model.boundml --data demo/data/test.csv --out demo/pred.csv --preview 3",
    "boundml plan demo/model.toml --mode infer --tensor-budget 64KiB",
]
COLS, ROWS = 92, 24
TYPE_S, LH, CW, PAD = 0.028, 19, 8.43, 18


def run_all() -> list[dict]:
    events = []
    with tempfile.TemporaryDirectory(prefix="boundml demo ") as d:
        for cmd in COMMANDS:
            args = cmd.split()[1:]
            t = time.perf_counter()
            r = subprocess.run([str(EXE), *args], cwd=d, capture_output=True, text=True)
            secs = time.perf_counter() - t
            out = (r.stdout + r.stderr).rstrip("\n")
            events.append({"cmd": cmd, "out": out, "code": r.returncode, "secs": secs})
    return events


def lines_for(events: list[dict]) -> list[tuple[float, str, str]]:
    """(time, kind, text) per terminal line; kind is prompt/out/err/note."""
    t, out = 0.4, []
    for e in events:
        out.append((t, "prompt", e["cmd"]))
        t += len(e["cmd"]) * TYPE_S + 0.25
        t += min(max(e["secs"], 0.15), 1.2)
        kind = "err" if e["code"] != 0 else "out"
        for raw in e["out"].splitlines() or [""]:
            for piece in textwrap.wrap(raw, COLS, drop_whitespace=False, replace_whitespace=False) or [""]:
                out.append((t, kind, piece))
                t += 0.03
        if e["code"] != 0:
            out.append((t, "note", f"[exit code {e['code']}]"))
        t += 0.55
    return out


def write_cast(events: list[dict], path: pathlib.Path) -> None:
    rec = [json.dumps({"version": 2, "width": COLS, "height": ROWS, "title": "BoundML demo (real commands)"})]
    t = 0.4
    for e in events:
        for ch in "$ " + e["cmd"]:
            rec.append(json.dumps([round(t, 3), "o", ch]))
            t += TYPE_S
        rec.append(json.dumps([round(t, 3), "o", "\r\n"]))
        t += min(max(e["secs"], 0.15), 1.2)
        body = e["out"].replace("\n", "\r\n") + "\r\n" if e["out"] else ""
        rec.append(json.dumps([round(t, 3), "o", body]))
        t += 0.55
    path.write_text("\n".join(rec) + "\n", encoding="utf-8")


def write_svg(lines: list[tuple[float, str, str]], path: pathlib.Path) -> None:
    total = lines[-1][0] + 4.0
    width = int(COLS * CW + 2 * PAD)
    height = ROWS * LH + 2 * PAD + 34
    colors = {"prompt": "#E6EDF3", "out": "#C9D1D9", "err": "#FFB4A8", "note": "#9AA4AE"}
    css, body = [], []

    def pct(x: float) -> str:
        return f"{100 * x / total:.3f}%"

    for i, (t, kind, text) in enumerate(lines):
        y = PAD + 34 + (i + 1) * LH - 5
        if kind == "prompt":
            content = f'<tspan fill="#3DD9B3">$ </tspan>{html.escape(text)}'
        else:
            content = html.escape(text)
        css.append(f"@keyframes l{i}{{0%,{pct(t)}{{opacity:0}}{pct(t + 0.001)},99.9%{{opacity:1}}100%{{opacity:0}}}}.l{i}{{animation:l{i} {total:.2f}s infinite step-end}}")
        body.append(f'<text class="l{i}" x="{PAD}" y="{y}" fill="{colors[kind]}" xml:space="preserve">{content}</text>')
    # Scroll like a terminal once more than ROWS lines are visible.
    frames = ["0%{transform:translateY(0px)}"]
    for i, (t, _, _) in enumerate(lines):
        off = max(0, i + 1 - ROWS) * LH
        if off:
            frames.append(f"{pct(t)}{{transform:translateY(-{off}px)}}")
    frames.append("100%{transform:translateY(0px)}")
    css.append("@keyframes scroll{" + "".join(frames) + "}" + f".scroll{{animation:scroll {total:.2f}s infinite step-end}}")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">BoundML command-line demo</title>
<desc id="desc">Recorded from the real boundml binary: version, init, train, evaluate, pack, predict with a preview, and a rejected memory plan with its explanation.</desc>
<style>text{{font-family:"Cascadia Mono",Consolas,"SFMono-Regular",Menlo,monospace;font-size:14px;opacity:0}}{''.join(css)}</style>
<rect width="{width}" height="{height}" rx="12" fill="#16191D"/>
<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" fill="none" stroke="#2D333B"/>
<circle cx="22" cy="20" r="6" fill="#3a3f46"/><circle cx="42" cy="20" r="6" fill="#3a3f46"/><circle cx="62" cy="20" r="6" fill="#3a3f46"/>
<text x="{width / 2}" y="25" fill="#9AA4AE" text-anchor="middle" style="opacity:1;font-size:12px">boundml demo - real output</text>
<clipPath id="c"><rect x="0" y="34" width="{width}" height="{height - 34}"/></clipPath>
<g clip-path="url(#c)"><g class="scroll">{"".join(body)}</g></g>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    if not EXE.exists():
        sys.exit(f"{EXE} not found; run cargo build --release first")
    events = run_all()
    (ROOT / "assets").mkdir(exist_ok=True)
    write_cast(events, ROOT / "assets" / "demo.cast")
    write_svg(lines_for(events), ROOT / "assets" / "demo.svg")
    (ROOT / "assets" / "demo-transcript.txt").write_text(
        "\n".join(f"$ {e['cmd']}\n{e['out']}\n[exit {e['code']}, {e['secs']:.2f}s]" for e in events) + "\n", encoding="utf-8")
    for e in events:
        print(f"exit {e['code']}  {e['secs']:.2f}s  {e['cmd']}")


if __name__ == "__main__":
    main()
