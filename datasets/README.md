# Demo datasets

Both datasets come from the UCI Machine Learning Repository and are licensed
under Creative Commons Attribution 4.0 (CC BY 4.0). The CSV files in this
directory are format conversions (header added, separator normalized) and
deterministic splits of the originals; values are unchanged.

| Directory | Source | Citation | Raw SHA-256 |
|---|---|---|---|
| `breast_cancer/` | [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17) — `wdbc.data` | Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). DOI 10.24432/C5DW2B | `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a` |
| `wine_red/` | [Wine Quality](https://archive.ics.uci.edu/dataset/186) — `winequality-red.csv` | Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). DOI 10.24432/C56S3T | `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e` |

Reproduce:

```bash
python -I datasets/prepare.py --download
boundml split datasets/breast_cancer/all.csv --out datasets/breast_cancer --valid 0.15 --test 0.2 --seed 2026
boundml split datasets/wine_red/all.csv --out datasets/wine_red --valid 0.15 --test 0.2 --seed 2026
```

Split SHA-256 (seed 2026):

| File | Rows | SHA-256 |
|---|---|---|
| breast_cancer/train.csv | 367 | `cb3de7419f72f2817f18ba205e6be35ec32f4bcc67713d598939641aa5074172` |
| breast_cancer/valid.csv | 78 | `cea4a66187de9e02fb713026985e382470048de879e8d088b3c9ae0bd7853ab3` |
| breast_cancer/test.csv | 124 | `35b698623144bdb08669f6f084544b921536a6f5055a36fe10b7eb59cc66896b` |
| wine_red/train.csv | 1046 | `0921d808bcc442542bf0ed089a7206d42d63dbfdff12b535d65ecaa04a2b0aaa` |
| wine_red/valid.csv | 225 | `3f199fb79754147550624310e36751a09a96ded392849fc61b09b236ba61710a` |
| wine_red/test.csv | 328 | `7c38008b72cb802e0a74551edc48f861f3196241a2b7a094924227e54ea4374d` |

The test splits were used once per final configuration; hyperparameters were
chosen on the validation splits only.
