# Morning handoff — BoundML private beta

## Verdict

**PRIVATE BETA CANDIDATE READY FOR OWNER REVIEW.**

It works as documented on Windows, Linux and macOS. CI is green on the candidate
commit, and the CLI, Rust and Python install paths were run from built artifacts in
fresh directories. It is **not** ready to *distribute*. These need you:

- The PyPI name `boundml` belongs to an unrelated project.
- No conduct contact has been supplied.
- Nothing has been published (as instructed).

## Start here (five minutes)

1. **Revisions.**
   - Branch `readiness/private-beta`; candidate commit `8ccdd91ccbfd3723805f8d1c5f297dacdd09858d`.
   - The commit that adds this file only changes `docs/MORNING-HANDOFF.md`, which is not shipped in archives.
   - `main` is unchanged at `fe9156e83a7a9f406bb4e9d2c78afdc60b7c60f2`. Merging is your decision.
2. **CI:** [run 40140798481](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40140798481) on `8ccdd91` passed all 10 jobs: Rust, archive smoke and Python on Linux, Windows and macOS, plus migration.
3. **Fixes tonight:**
   - Benchmark-harness defects (B1–B3).
   - A Python bug: `predict(df)` crashed on a real CSV with a text label column.
   - Checksum files written on Windows failed `sha256sum -c` on Linux and macOS.
   - Docs that promised a Releases page that doesn't exist.
4. **Try it:** the commands are in [section 6](#6-try-it-yourself).
5. **Decide:** the owner decisions are in [section 7](#7-limitations-failed-checks-and-owner-decisions).

## 1. Revisions

| | |
|---|---|
| Starting revision | `fe9156e83a7a9f406bb4e9d2c78afdc60b7c60f2` (`main` = `origin/main`, clean tree, repository PRIVATE), verified on 2026-10-24 |
| Final branch | `readiness/private-beta` (pushed to the private repository, not merged) |
| Candidate commit (tested, artifacts built) | `8ccdd91ccbfd3723805f8d1c5f297dacdd09858d` |
| Final branch head | the commit that adds this file; it changes only `docs/MORNING-HANDOFF.md` (excluded from release archives) |
| Versions | Rust crates 0.1.0, Python package 0.1.0 (`python/pyproject.toml`, `python/Cargo.toml`), toolchain Rust 1.94.0 |

Earlier claims that I checked rather than assumed:
- All 10 CI jobs passed at `fe9156e`: confirmed, [run 40034572176](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40034572176).
- Nothing was tagged or published: confirmed, 0 tags and no releases.
- **Correction:** run 40031771890 (`c8297a0`) is recorded by GitHub as *cancelled*, not failed as reported earlier.

## 2. What changed and why

| Commit | Change | Why |
|---|---|---|
| `be955d2` | **B1** Parity files must contain exactly batch sizes 1, 7 and 64. Each needs `rows × outputs` finite values: rows counted from the test CSV, outputs taken from the package task. The reference must be non-empty and finite. Malformed files and results lines are errors. **B2** A worker exiting before psutil attaches no longer crashes the driver. Every failure path kills the child, closes its pipes and joins its reader threads. **B3** PyTorch builds its controlled batch order before its training timer. A new 20-test harness suite runs in CI on all three OSes. | Before: a worker reporting only some batch sizes passed, as did an empty reference. Error paths leaked processes and pipes. PyTorch's timer included tensor conversion. 21 of the new test cases fail or error against the old driver (`out/b123-old-driver.log`). |
| `42abd73` | Test: a server at capacity stops reading an endlessly trickled rejected body within the single drain deadline. The troubleshooting guide states the limits. | The Windows 503 fix had no test of its time ceiling, and its limits were undocumented. |
| `b35e9ff` | Training rows re-measured (5 runs) with `--training-only`; inference results unchanged; previous record kept in `docs/results/history/2026-10-24-pre-b3/`. | B3 changed the PyTorch timer boundary. |
| `3a0841a` | Python `predict` on a DataFrame or dict rows converts only feature columns. Unknown columns follow the schema policy; non-numeric features raise `BML-D005`. | Found by running the README example: `predict(pd.read_csv(test.csv))` crashed on the `diagnosis` text column. 3 new tests fail on the old code. |
| `a0a397d` | README and docs rewritten around the executed examples. Conduct and security contacts are marked pending. | The docs promised a Releases page and a PyPI install, and had a Rust dependency path that couldn't resolve. The conduct channel was the vulnerability form. |
| `4cb8b6c` | CI runs on `readiness/**` branches and on manual dispatch. | To test this branch without touching `main` or opening a PR. |
| `741efd7` | Checksum files are written with LF on every OS. Private review docs are excluded from archives. Publication plan and release-notes draft added. | The Windows `.sha256` failed `sha256sum -c` on Unix (found by checking the CI artifact). |
| `8ccdd91` | A shipped page no longer links to an archive-excluded document. | The archive link check failed on `741efd7` (CI run 40140675419, below). |

## 3. Server fix review (Windows overload 503)

I reviewed the implementation and its tests:
- `discard_body` has one overall 250 ms deadline, not one per chunk, and a 64 KiB byte limit (it can overshoot by at most one frame).
- Each frame is dropped as soon as it is counted, so the body is never accumulated.
- A client disconnect or body error ends the drain.
- The delayed-body test reproduces the original Windows reset (the old code fails with error 10053).
- Both server tests release the held slot before asserting, so a failure cannot deadlock shutdown.

**Limit:** this improves the reproduced case. It does not guarantee a readable 503 for larger or slower bodies; those can still end in a connection reset. Draining requests hold their connection for up to 250 ms without an admission slot. All of this is documented in `docs/troubleshooting.md`. No redesign was needed.

## 4. Verification

| Check | Platform | Tested revision | Result | Evidence |
|---|---|---|---|---|
| fmt, clippy `-D warnings`, rustdoc `-D warnings` | Windows 11 | `a0a397d` | pass | `out/verify/final-validation-windows.log` |
| Rust tests (incl. all earlier review regression tests) | Windows 11 | `a0a397d` | 103 passed, 0 failed | `out/verify/rust-tests-windows.log` |
| Benchmark-harness tests (ResourceWarning as error) | Windows 11 | `a0a397d` | 20 passed | same log |
| Full CI: Rust, archive smoke, Python (10 + 20 harness tests), migration | ubuntu-latest, windows-latest, macos-latest | `8ccdd91` | **success, 10/10 jobs** | [run 40140798481](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40140798481) |
| Full CI | all three | `4cb8b6c` | success, 10/10 jobs | [run 40139782934](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40139782934) |
| Full CI | all three | `741efd7` | **package-smoke failed** on ubuntu and macos (archive link check); the rest was cancelled by the newer push | [run 40140675419](https://github.com/MahmoudYahiaAhmed/boundml/actions/runs/40140675419), fixed in `8ccdd91` |
| CLI archive path A (24 checks): version, init → predict, budget failure exit 4, schema error exit 3 without an output file, input protection, real data | Windows 11, local archive | `8ccdd91` | pass, reports `revision 8ccdd91ccb` | `out/verify/final-cli-windows.log` |
| CLI archive path A, CI-built archive | WSL2 Ubuntu 24.04 | `8ccdd91` | pass, reports `revision 8ccdd91ccb` | `out/verify/linux-ci-artifacts.log` |
| CLI archive smoke (13 steps), CI-built | macOS arm64 (CI only) | `8ccdd91` | pass | CI artifact `archive-macos-latest/smoke-*.json` |
| Rust embedding path B: README snippet in a new project (path and private-git dependency), session limit (BML-A010), `examples/embed` | Windows 11 | `b35e9ff` (Rust code unchanged since) | pass, `power = 4.821724` | `out/verify/rust-embedding.log` |
| Python wheel path C (14 checks): metadata and tags, README examples, bitwise CLI parity for dicts and DataFrames | Windows 11, Python 3.13 | `8ccdd91` | pass | `out/verify/final-python-windows.log` |
| Path C, same wheel | Windows 11, Python 3.11, 3.12, 3.14 | `b35e9ff` | pass | `out/verify/python-versions-windows.log` |
| Path C, CI-built Linux wheel | WSL2 Ubuntu, Python 3.12 | `8ccdd91` | pass | `out/verify/linux-ci-artifacts.log` |
| Markdown links and anchors (22 files) | — | `8ccdd91` | 0 problems | `out/verify/check_links.py` |
| Training benchmark rerun (5 runs, clean tree) | Windows 11 | `42abd73` | metrics identical; times disclosed | `docs/benchmarks.md`, `docs/results/bench.json` |

`out/` holds local logs and scripts. It is ignored by Git and not committed; the old pre-redaction Git bundle stays there, untouched and never uploaded.

## 5. Artifacts

Built from `8ccdd91ccbfd3723805f8d1c5f297dacdd09858d`. They are not bit-for-bit reproducible: a rebuild gives a different hash.

| File | Platform | Built by | Bytes | SHA-256 |
|---|---|---|---|---|
| `dist/boundml-0.1.0-x86_64-pc-windows-msvc.zip` | Windows x86_64 | local, Windows 11 | 2164741 | `bd6006fd3331754e9acfd1ed0bdb94cd51c616d342baf1a85009586566b44ae9` |
| `dist/python/boundml-0.1.0-cp39-abi3-win_amd64.whl` | Windows x86_64 | local | 896506 | `100ef4d36f1615bd3ba3375720bf016d7647ae1c1ff62ebfcbf065a19b3845a5` |
| `boundml-0.1.0-x86_64-pc-windows-msvc.zip` | Windows x86_64 | CI run 40140798481 | 2164334 | `f6fa0b02fbe1374c5d96ef3150e349f8d7764fe1eaa671c783f1e767ec65acf0` |
| `boundml-0.1.0-x86_64-unknown-linux-gnu.tar.gz` | Linux x86_64 | CI | 2312155 | `c35acaed15800b2963c9e82d5f3aeffaad015dd87f3484bcd5543b1c7bc3064f` |
| `boundml-0.1.0-aarch64-apple-darwin.tar.gz` | macOS arm64 | CI | 2158188 | `daec09507d8730ff197770eb1442fc1da3ec8da10b028a4c4687d52af9e01ce2` |
| `boundml-0.1.0-cp39-abi3-win_amd64.whl` | Windows x86_64 | CI | 895490 | `d743096a822814420c459ed440652fd1361e1ebdcdbe0a69a88f27b26b4d59b3` |
| `boundml-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl` | Linux x86_64, glibc ≥ 2.34 | CI | 1036260 | `b16fda3f6e767d442d90c504987b47026a6a637922cb1fd1f6355cbcb20b9f96` |
| `boundml-0.1.0-cp39-abi3-macosx_11_0_arm64.whl` | macOS arm64 | CI | 924523 | `cc5ccc83890392f48cfcc4d344d94488d9a15b2bd879037cb7dfcaf7b2613edd` |

- **Local artifacts** are in `dist/`, which is ignored by Git.
- **CI artifacts** are downloaded in `dist/ci-40140798481/` and also available from the run page, while GitHub retains them.
- **Revisions:** every archive's `boundml --version` reports `revision 8ccdd91ccb`. The binaries were built at that commit, not renamed or copied from another build.

## 6. Try it yourself

From the repository root, on the `readiness/private-beta` branch.

**CLI (Windows PowerShell, local archive):**
```powershell
$a = "dist\boundml-0.1.0-x86_64-pc-windows-msvc.zip"
(Get-FileHash $a -Algorithm SHA256).Hash -eq (Get-Content "$a.sha256").Split(" ")[0].ToUpper()
Expand-Archive $a -DestinationPath "$env:TEMP\boundml try"
cd "$env:TEMP\boundml try\boundml-0.1.0-x86_64-pc-windows-msvc"
.\boundml --version
.\boundml init demo
.\boundml train demo\model.toml --checkpoint demo\ckpt --quiet
.\boundml evaluate demo\ckpt --data demo\data\test.csv
.\boundml pack demo\ckpt --out demo\model.boundml
.\boundml predict demo\model.boundml --data demo\data\test.csv --out demo\pred.csv --preview 3
.\boundml plan demo\model.toml --mode infer --tensor-budget 64KiB   # expected: BML-C005, exit 4
```

**Rust (from the repository root, using the package made above):**
```powershell
cargo run --release -p boundml-embed-example -- "$env:TEMP\boundml try\boundml-0.1.0-x86_64-pc-windows-msvc\demo\model.boundml" temperature=20 humidity= load=0.5
```
Expected: `power = 4.821724` and `2 concurrent sessions produced identical results`.

**Python (fresh environment, local wheel):**
```powershell
python -m venv "$env:TEMP\boundml venv"
& "$env:TEMP\boundml venv\Scripts\python" -m pip install dist\python\boundml-0.1.0-cp39-abi3-win_amd64.whl pandas
cd "$env:TEMP\boundml try\boundml-0.1.0-x86_64-pc-windows-msvc"
& "$env:TEMP\boundml venv\Scripts\python" -c "import boundml; m = boundml.load('demo/model.boundml'); print(m.features, m.predict({'temperature': 20, 'humidity': None, 'load': 0.5}))"
```
Expected: `['temperature', 'humidity', 'load'] [4.821724]`.

Do **not** run `pip install boundml` without a file path: that installs an unrelated PyPI project.

## 7. Limitations, failed checks and owner decisions

**Owner decisions:**
1. **PyPI name.** `boundml` on PyPI belongs to an unrelated branch-and-bound solver toolbox (v1.5.2). Choose another distribution name; `boundml-rs` and `boundml-py` were free on 2026-10-25. Also decide whether the import name stays `boundml`. All five crates.io names were free.
2. **Contacts.** A private Code of Conduct contact is needed: `CODE_OF_CONDUCT.md` has a marked placeholder. Security reporting through GitHub needs the repository to be public, with the feature enabled.
3. **Merge, visibility, tag and publish**, in the order given in `docs/publication-plan.md`. `release.yml` (tag-triggered) has never run.
4. **Linux wheels** currently need glibc 2.34+. Building with manylinux2014 would widen support; not done.

**Limitations still true:**
- One benchmark machine and two small models.
- **ONNX Runtime is about 2–2.4× faster on 256-row batches.** T3 model-only reaches 42% and 51% against a 50% threshold.
- The tested **MLPs did not beat the linear baselines**.
- The training rerun ran slower across all rows, including the untouched CLI: T4 on breast cancer is 1.78× against its 2× limit (previously 1.02×). I attribute this to machine state; it is disclosed in `docs/benchmarks.md`.
- Python 3.9 and 3.10 were not run.
- macOS was tested only in CI.
- The reviewer's Linux object-file error is still unreproduced.
- No fuzzing or `cargo audit`.
- No outside users. Agent rehearsals are not evidence of demand: see the trial plan in `docs/publication-plan.md`.

**Failed or cancelled checks tonight:**
- CI run 40140675419 (`741efd7`) failed `package-smoke` on Linux and macOS. The cause was a link from a shipped page to an excluded document; fixed in `8ccdd91`, which passed.
- I stopped one training rerun midway because I compiled code during it. It was restarted on an idle machine; only the clean run is recorded.

## 8. Confirmations

- The repository **remains PRIVATE** (`gh repo view`: `PRIVATE`).
- **No registry packages** were published (crates.io: none; PyPI: none).
- **No tags** (`git ls-remote --tags origin`: 0) and **no GitHub releases** were created.
- **`main` was not changed or merged into**; work is on `readiness/private-beta` only. No history was rewritten, nothing was force-pushed, and no branch was deleted.
- **No announcements or external messages** were sent and no one was contacted. The only network reads beyond GitHub were two package-name lookups on pypi.org and crates.io.
- No credentials were added or exposed, and no paid services were enabled.
