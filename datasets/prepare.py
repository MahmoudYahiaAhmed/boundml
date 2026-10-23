"""Download (optional) and convert the two public demo datasets to headered CSV.

Sources (UCI Machine Learning Repository, CC BY 4.0):
  * Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17
  * Wine Quality (red):                   https://archive.ics.uci.edu/dataset/186

Usage:  python -I datasets/prepare.py [--download]
Raw files live in datasets/raw/ (git-ignored); outputs in datasets/<name>/all.csv.
Downloads are verified against pinned SHA-256 hashes before use.
"""

import csv
import hashlib
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "raw"

SOURCES = {
    "wdbc.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a",
    ),
    "winequality-red.csv": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
    ),
}

WDBC_BASE = [
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
]
WDBC_FEATURES = [f"{b}_{s}" for s in ("mean", "se", "worst") for b in WDBC_BASE]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(download):
    RAW.mkdir(exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        path = RAW / name
        if not path.exists():
            if not download:
                sys.exit(f"missing {path}; rerun with --download")
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        if got != digest:
            sys.exit(f"{name}: sha256 {got} != pinned {digest}")


def write(rows, header, out):
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def main():
    fetch("--download" in sys.argv)
    with (RAW / "wdbc.data").open() as f:
        rows = [[r[0], *r[2:], r[1]] for r in csv.reader(f) if r]
    write(rows, ["id", *WDBC_FEATURES, "diagnosis"], ROOT / "breast_cancer" / "all.csv")

    with (RAW / "winequality-red.csv").open() as f:
        reader = csv.reader(f, delimiter=";")
        header = [h.strip().replace(" ", "_") for h in next(reader)]
        rows = [r for r in reader if r]
    write(rows, header, ROOT / "wine_red" / "all.csv")


if __name__ == "__main__":
    main()
