## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checks

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings`
- [ ] `cargo test --workspace --locked`
- [ ] Docs updated (README, `docs/`, `CHANGELOG.md` under *Unreleased*) if behaviour or output changed
- [ ] New error paths have a `BML-` code documented in `docs/errors.md`
- [ ] Package format unchanged, or `format_version` bumped and `docs/package-format.md` updated
- [ ] Kernel/autodiff changes: `python scripts/torch_reference.py` re-run and fixture committed

## Evidence

<!-- Test names, commands run, benchmark numbers with the command that produced them. -->
