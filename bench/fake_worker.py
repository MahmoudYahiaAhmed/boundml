"""Scripted stand-in for an inference worker, used only by bench/test_run_bench.py.

    python bench/fake_worker.py <mode> <parity.json> [<rows>] [<cols>]

Follows the worker protocol (FIRST, parity file + PARITY, results line, wait for
stdin) and misbehaves in the way `mode` names.
"""

import json
import sys


def main() -> None:
    mode, parity = sys.argv[1], sys.argv[2]
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cols = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    vals = [0.5] * (rows * cols)
    batches = {"1": vals, "7": vals, "64": vals}
    doc = {"outputs_per_row": cols, "batches": batches}
    if mode == "missing-batch":
        del batches["64"]
    elif mode == "empty":
        doc["batches"] = {"1": [], "7": [], "64": []}
    elif mode == "short":
        batches["7"] = vals[:-1]
    elif mode == "wrong-width":
        doc["outputs_per_row"] = cols + 1
    elif mode == "nan":
        batches["7"] = vals[:-1] + [float("nan")]
    print("FIRST", flush=True)
    if mode == "exit-before-parity":
        sys.exit(4)
    with open(parity, "w") as f:
        if mode == "malformed-parity":
            f.write("{not json")
        else:
            json.dump(doc, f)  # json.dump writes NaN as NaN, which json.loads accepts
    print("PARITY", flush=True)
    if mode == "malformed-results":
        print("not json", flush=True)
    else:
        print(json.dumps({"runtime": "fake"}), flush=True)
    sys.stdin.readline()
    sys.exit(3 if mode == "nonzero-exit" else 0)


if __name__ == "__main__":
    main()
