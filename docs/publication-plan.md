# Publication plan (private draft)

Nothing in this file has been done. It lists what publishing would take, in order,
and what only the owner can decide. Status as of 2026-10-25.

## Four separate questions

| Question | Status | Evidence / what is missing |
|---|---|---|
| 1. **Implementation readiness**: does it work as documented? | **Yes, for the scoped beta** | CI green on Linux, Windows and macOS; install paths verified in fresh environments; see [MORNING-HANDOFF.md](MORNING-HANDOFF.md) |
| 2. **Distribution readiness**: can people install it without a checkout? | **No** | No GitHub release; nothing on crates.io; the PyPI name `boundml` is taken by an unrelated project; Linux wheels need glibc 2.34+ |
| 3. **Public-release authorization** | **Not given** | Owner decisions below |
| 4. **Evidence of real user demand** | **None** | Only automated checks and agent rehearsals so far; see the trial plan |

## Owner decisions and setup still required

1. **Python distribution name.** `boundml` on PyPI belongs to an unrelated project
   (a branch-and-bound solver toolbox, v1.5.2, `github.com/sirenard/BoundML`).
   Choose a free distribution name (`boundml-rs` and `boundml-py` were free on
   2026-10-25) and decide whether the import name stays `boundml`; it would clash
   if both packages were installed in one environment. A rename means editing
   `python/pyproject.toml` (`[project] name`) and the install docs.
2. **Contacts.** A private Code of Conduct contact (an email address or form only
   maintainers read) to replace the placeholder in `CODE_OF_CONDUCT.md`, and a
   security route. GitHub private vulnerability reporting works only once the
   repository is public and the feature is enabled in Settings → Security.
3. **Visibility.** Make the repository public (Settings → General → Danger zone),
   then enable private vulnerability reporting.
4. **Registry accounts and tokens.** A crates.io API token stored as the
   `CARGO_REGISTRY_TOKEN` secret of a protected `crates-io` environment (the
   `publish-crates` workflow expects it); a PyPI project and, preferably, trusted
   publishing for the chosen name.
5. **Linux wheel compatibility.** Wheels built on current runners need glibc 2.34+
   (Ubuntu 22.04+). For older distributions, build with `maturin` in a
   manylinux2014 container (for example `PyO3/maturin-action` with `manylinux: 2014`).
   Not done.

## Order of publication

1. Merge `readiness/private-beta` into `main` after review (owner).
2. Make the repository public, enable vulnerability reporting, add the contacts.
3. Tag `v0.1.0` on `main`. `release.yml` builds the three archives, runs the same
   smoke test as CI, and attaches them with `SHA256SUMS` and
   `.github/release-notes/v0.1.0.md`. It has never run; check the first run.
4. crates.io, in dependency order (each must be live before the next can resolve
   it; `publish-crates` runs `--dry-run` first by default):
   `boundml-core` → `boundml-data` → `boundml-artifact` → `boundml-train` →
   `boundml-cli`. `boundml-bench`, the examples and `boundml-py` are
   `publish = false` or outside the workspace.
5. PyPI, under the chosen name: wheels for Windows x86_64, Linux x86_64
   (manylinux2014 if decided) and macOS arm64, plus an sdist.

## Intended artifacts for 0.1.0

| File | Platform | Built by |
|---|---|---|
| `boundml-0.1.0-x86_64-pc-windows-msvc.zip` + `.sha256` | Windows x86_64 | `scripts/release.py` (locally, CI `package-smoke`, `release.yml`) |
| `boundml-0.1.0-x86_64-unknown-linux-gnu.tar.gz` + `.sha256` | Linux x86_64 | same |
| `boundml-0.1.0-aarch64-apple-darwin.tar.gz` + `.sha256` | macOS arm64 | same |
| `SHA256SUMS` | all | same |
| `boundml-0.1.0-cp39-abi3-win_amd64.whl` | Windows x86_64, Python ≥ 3.9 | `maturin build` (CI `python` job) |
| `boundml-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl` | Linux x86_64, glibc ≥ 2.34 | same |
| `boundml-0.1.0-cp39-abi3-macosx_11_0_arm64.whl` | macOS arm64 | same |
| crates `boundml-{core,data,artifact,train,cli}` 0.1.0 | source | `cargo publish` |

The concrete files and checksums built for this review are listed in
[MORNING-HANDOFF.md](MORNING-HANDOFF.md). Builds are not bit-for-bit reproducible.

## Installation commands at each stage

| Stage | CLI | Rust library | Python |
|---|---|---|---|
| Now (private) | build: `python scripts/release.py --out dist`, or the CI `archive-<os>` artifact; or `cargo install --path crates/boundml-cli --locked` | `boundml-artifact = { path = "<checkout>/crates/boundml-artifact" }` or `{ git = "https://github.com/MahmoudYahiaAhmed/boundml" }` with access | `pip install ./python` or `pip install <wheel file>` |
| After a GitHub release | download the archive from the release, check `SHA256SUMS` | unchanged | `pip install <wheel attached to the release>` (if attached) |
| After crates.io | `cargo install boundml-cli --locked` | `boundml-artifact = "0.1"` | unchanged |
| After PyPI | unchanged | unchanged | `pip install <chosen name>` (never `pip install boundml`) |

## Trial with 3–5 outside developers (plan only; nobody has been contacted)

**Who:** Rust developers who already ship, or want to ship, a small numeric model in a
CLI tool, desktop app, gateway or local service. Aim for at least one team that
retrains regularly and at least one Python-first user. Exclude people who would only
benchmark it.

**What each person does, with their own data and without help:**
1. Install from the documented path (archive or wheel) and reach a first prediction
   on the bundled example.
2. Train on their own CSV (or import their own small PyTorch/Keras model), pack it,
   and check predictions against what they use today.
3. Embed the package in their program, or call it from Python.
4. Optionally, run `boundml compare` on a retrained model.

**Measure:** time from download to first prediction on their own data; every point
where they got stuck (error code, doc page); agreement with their current model;
memory and start-up in their environment; whether they would keep using it, and
what is missing. Collect this with the *Workload report* issue template.

**Decide afterwards:** continue as is, narrow the scope (for example to Rust embedding
only), or stop. A good result is at least two of the five completing step 3 unaided,
and at least one wanting to keep it.
