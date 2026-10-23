# Launch readiness report — BoundML 0.1.0

Prepared 2026-10-23. Nothing has been pushed, published or posted.

## Ready locally (verified on Windows 11 x64)

| Item | Result |
|---|---|
| Audit findings (2 high, 5 medium, 8 low) | all fixed with regression tests; audit repros re-run: oversized/batch-mixing/malformed packages exit 5, dropped-feature update is Breaking (exit 6) |
| Tests | `cargo test --workspace --locked`: 71 passed, 0 failed |
| Lint / docs | `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `cargo doc -D warnings`: clean |
| Release archive | `dist/boundml-0.1.0-x86_64-pc-windows-msvc.zip` (2,114,545 bytes), sha256 `5e31a81369bed4122efb7a2397ccba1d651efd302992cea333812957d3b9b5c8` |
| Archive smoke test | extracted into a path with spaces; version, init, validate, train, evaluate, pack, verify, predict, budget failure (exit 4), real-data train/evaluate all pass |
| First use from a prebuilt archive | extract 0.04 s + init→first prediction 0.23 s of execution (download and reading time excluded) |
| Newcomer rehearsal (fresh clone + archive, independent agent) | every documented command ran and every metric, prediction, error code and exit code matched; 11 usability gaps found (PowerShell `.oundml`, PowerShell `curl` alias, missing `renamed.csv` command, real-data working directory, `install.md` Releases link, archive missing linked files, …) — all fixed and re-verified; extraction → first prediction 0.98 s with `Expand-Archive`; `cargo install` 32 s (crates cached) |
| Native comparison | tract (Rust) and Burn added; tract 2.5–3.3× faster at batch 256, BoundML faster at single-row, start-up, memory |
| Migration gate | 6 PyTorch/Keras models, max diff 1.2e-6; unsupported layer and tampered weights rejected |
| Stress (post-fix binary) | 300 s, 10,469 reloads + self-tests, 10,469 train/checkpoint/resume cycles, 5.59 billion rows over 4 sessions with no drift; working-set growth 0.7%, peak 8.8 MiB |
| Hygiene | no personal paths, credentials or private data in tracked files; build outputs, `out/`, `dist/`, raw downloads ignored |
| crates.io | names `boundml`, `boundml-core`, `-data`, `-artifact`, `-train`, `-cli` unclaimed as of 2026-10-23; `boundml-core` passes `cargo publish --dry-run`; the other four fail dry-run only because their dependencies are not yet on crates.io |

## Depends on things outside this machine

| Item | Needed |
|---|---|
| Public repository URL / owner | not chosen; no URL is written anywhere. Add `repository` to `[workspace.package]` (and inherit it) once known; crate READMEs use repo-relative links that will only resolve on crates.io after that |
| CI status | push and observe `ci.yml`; no badge until a run has passed |
| Linux / macOS archives | first `release.yml` run on a `v0.1.0` tag (creates a draft release only) |
| Private vulnerability reporting | enable in repository settings (`SECURITY.md` and `CODE_OF_CONDUCT.md` point there) |
| Maintainer contact | the maintainer may want a public contact email in `SECURITY.md`; none was added without explicit consent |
| crates.io publication | `publish-crates.yml` (manual, `crates-io` environment + token), order core → data → artifact → train → cli |
| Social preview, description, topics | `launch/repository-settings.md` |
| Demand validation | 3–5 external workload evaluations (`docs/design-partner-plan.md`, `launch/feedback-form.md`) |
| Not run | fuzzing, `cargo audit`/`cargo deny` |

## Launch materials (drafts)

`launch/release-notes-v0.1.0.md`, `launch/announcement-github.md`,
`launch/linkedin-post.md`, `launch/rust-community-intro.md`,
`launch/demo-script-60s.md`, `launch/feedback-form.md`, `launch/starter-issues.md`,
`launch/repository-settings.md`.

## Publication checklist (when you ask for it)

1. Create the repository, set `repository` metadata, push `main`.
2. Enable private vulnerability reporting; apply description/topics/social preview.
3. Wait for `ci.yml` to pass on all three platforms; fix anything platform-specific.
4. Tag `v0.1.0`; review the draft release created by `release.yml` (archives,
   `SHA256SUMS`, smoke reports); publish it.
5. Optionally run `publish-crates.yml` (dry run first) and then update README/install
   docs to mention `cargo install boundml-cli`.
6. Post the drafts.
