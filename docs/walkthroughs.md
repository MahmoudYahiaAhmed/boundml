# Walkthroughs

All outputs below were produced by the commands shown, on 2026-10-23, Windows 11 x64,
`target/release/boundml`. The commands are written as `boundml`: install it with
`cargo install --path crates/boundml-cli --locked` ([install.md](install.md)), or
run `target/release/boundml` (PowerShell: `.	argeteleaseoundml`) after the build
below. Sections 1–3 need a checkout: the release archive ships only the
breast-cancer splits. Training is deterministic, so you should get the same
metrics on the same platform. Wall-clock times will differ.

## 0. Quickstart (synthetic data, no download)

```bash
cargo build --release
```

```bash
boundml init demo
```

(`init` refuses to overwrite an existing `demo` directory with `BML-T001`; pick another
name if you already ran the README quickstart.)

```bash
boundml validate demo/model.toml
```

```bash
boundml plan demo/model.toml --mode train
```

```bash
boundml train demo/model.toml --checkpoint demo/out/ckpt
```

```bash
boundml evaluate demo/out/ckpt --data demo/data/test.csv
```

```bash
boundml pack demo/out/ckpt --out demo/out/model.boundml
```

```bash
boundml verify demo/out/model.boundml
```

```bash
boundml predict demo/out/model.boundml --data demo/data/test.csv --out demo/out/pred.csv
```

```bash
boundml serve demo/out/model.boundml --bind 127.0.0.1:8080
```

```bash
curl -s -X POST localhost:8080/v1/predict -H "content-type: application/json" -d '{"rows":[{"temperature":20,"humidity":50,"load":0.5}]}'
```

Embed the same package in your own program: see `examples/embed/src/main.rs`.

```bash
cargo run -p boundml-embed-example -- demo/out/model.boundml temperature=20 humidity= load=0.5
```

(`humidity=` is a missing value. The example config uses `impute_mean`, so the
training mean is used.)

Memory budgets in action: the example sets `deploy.tensor_budget = "1MiB"` for 2
sessions.

```bash
boundml plan demo/model.toml --mode infer --tensor-budget 64KiB
```

prints `BML-C005 tensor budget exceeded: plan needs 68228 bytes but the budget is
65536 bytes; largest contributors: arena x 2 session(s) (65536 bytes), ...` and exits
with code 4. No arena is allocated.

## 1. Breast Cancer Wisconsin (binary classification)

Data: UCI, CC BY 4.0, 569 rows, 30 numeric features, label `diagnosis` ∈ {B, M}.
Split with seed 2026 into 367 train / 78 validation / 124 test rows (hashes in
`datasets/README.md`). The `id` column is in the files and is ignored by the
`extra_columns = "ignore"` policy. Hyperparameters were fixed before looking at the
test set. Models were selected on validation only, and each final config was
evaluated on test once.

Optional, to re-download the raw UCI files; the checkout already ships `all.csv`
and the splits below:

```bash
python -I datasets/prepare.py --download
```

```bash
boundml split datasets/breast_cancer/all.csv --out datasets/breast_cancer --valid 0.15 --test 0.2 --seed 2026
```

```bash
boundml train examples/breast_cancer/linear.toml --checkpoint out/wt/breast_cancer-linear/ckpt --quiet
```

```bash
boundml evaluate out/wt/breast_cancer-linear/ckpt --data datasets/breast_cancer/test.csv
```

| Model (test, 124 rows) | Accuracy | Log loss | Precision (M) | Recall (M) | F1 |
|---|---|---|---|---|---|
| Constant (majority class) | 0.5806 | 0.6870 | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression (`linear.toml`) | **0.9597** | **0.1379** | 1.0000 | 0.9038 | 0.9495 |
| MLP 30-16-1 ReLU (`mlp.toml`) | **0.9597** | 0.2298 | 0.9796 | 0.9231 | 0.9505 |

The MLP does **not** beat logistic regression here: accuracy is identical and its log
loss is worse (less well-calibrated probabilities). For this dataset, the linear
model is the better choice.

## 2. Wine Quality, red (regression)

Data: UCI, CC BY 4.0, 1599 rows, 11 numeric features, target `quality` (integer
sensory score, treated as continuous). Split seed 2026: 1046 / 225 / 328. All columns
other than features and label are rejected (`extra_columns = "reject"`).

```bash
boundml split datasets/wine_red/all.csv --out datasets/wine_red --valid 0.15 --test 0.2 --seed 2026
```

```bash
boundml train examples/wine_quality/mlp.toml --checkpoint out/wt/wine_quality-mlp/ckpt --quiet
boundml train examples/wine_quality/linear.toml --checkpoint out/wt/wine_quality-linear/ckpt --quiet
```

```bash
boundml evaluate out/wt/wine_quality-mlp/ckpt --data datasets/wine_red/test.csv
```

| Model (test, 328 rows) | RMSE | MAE | R² |
|---|---|---|---|
| Constant (training mean) | 0.8284 | 0.6901 | −0.0004 |
| Linear regression (`linear.toml`, 150 epochs) | **0.6242** | **0.4697** | **0.4320** |
| MLP 11-32-16-1 ReLU (`mlp.toml`, 45 epochs) | 0.6635 | 0.5039 | 0.3582 |

How the MLP was tuned: the first configuration (64-32, 150 epochs, wd 1e-4) overfit
(validation RMSE 0.82). Five alternatives were compared on the **validation** split
only. The best of these (32-16, lr 1e-3, wd 0.05) reached its lowest validation RMSE,
0.658, around epoch 42 of a 60-epoch run, so the final config uses 45 epochs. Re-run
end to end, the final config scores 0.6652 on validation against 0.6647 for the linear
model (the `compare` output in section 3), which makes the two indistinguishable. On the
**test** split the linear model is better (0.624 vs 0.664). This result is reported as observed. With ~1000 training
rows, a linear model is a strong baseline, and BoundML's packaging and execution
features do not make an MLP statistically better.

The output layer's bias starts at the training label mean (regression) or log-prior
(classification), as described in `docs/architecture.md`. Without this, the
regression MLPs spent many epochs learning the offset of the ~5.6-mean target.

## 3. Updating a deployed model

```bash
boundml pack out/wt/wine_quality-linear/ckpt --out out/wt/current.boundml
```

```bash
boundml pack out/wt/wine_quality-mlp/ckpt --out out/wt/candidate.boundml
```

```bash
boundml compare out/wt/current.boundml out/wt/candidate.boundml --data datasets/wine_red/valid.csv
```

The report lists graph/parameter changes, refitted preprocessing (when the statistics differ), planned memory
before and after, max/mean output difference and the validation metric for both
packages. `--max-abs-diff` turns large numeric changes into a breaking verdict
(exit code 6) for CI gating.

Feature engineering note: both datasets are used exactly as published. No derived
features, encodings or windows are applied, so nothing outside the package is needed
to reproduce the predictions.
