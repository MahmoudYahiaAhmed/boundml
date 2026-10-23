# 60-second demo script (draft)

Screen: a terminal in an empty folder, with `boundml` on PATH. Every command is real;
expected output is in `assets/demo-transcript.txt`.

| Time | Say | Do |
|---|---|---|
| 0–8 s | "You have a small numeric model to ship inside a Rust app. Usually that means Python for training, an export step, and preprocessing rewritten by hand. BoundML keeps all of it in one checked package." | Show the README banner. |
| 8–18 s | "One command makes a runnable example: a config and a CSV." | `boundml init demo` |
| 18–28 s | "Train from the CSV. Preprocessing is fitted on the training rows only." | `boundml train demo/model.toml --checkpoint demo/ckpt --quiet`, then `boundml evaluate demo/ckpt --data demo/data/test.csv` (point at the constant baseline column). |
| 28–38 s | "Package it: schema, fitted preprocessing, weights, self-tests and a memory plan, all checked on load." | `boundml pack demo/ckpt --out demo/model.boundml`, then `boundml verify demo/model.boundml` |
| 38–48 s | "Predict from CSV. The Rust API does exactly the same thing in-process." | `boundml predict demo/model.boundml --data demo/data/test.csv --out demo/pred.csv --preview 3`; flash `examples/embed/src/main.rs`. |
| 48–56 s | "If the model can't fit your memory budget, you find out before anything is allocated, and you get told why." | `boundml plan demo/model.toml --mode infer --tensor-budget 64KiB` (exit code 4, contributors listed). |
| 56–60 s | "CPU only, early release. Benchmarks, including where other runtimes win, are linked in the README." | Show the benchmark section. |
