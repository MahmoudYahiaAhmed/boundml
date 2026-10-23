# Repository settings to apply at publication (drafts — nothing has been configured)

**Description** (GitHub limit 350 characters):

> Small models. Simple deployment. Native Rust. Train dense models from CSV, package preprocessing and weights together with a checked schema and memory plan, and run predictions inside your Rust application. CPU, early release.

**Website**: none yet (do not set a docs.rs link until the crates are published).

**Topics**: `rust`, `machine-learning`, `tabular-data`, `inference`, `model-deployment`,
`neural-network`, `cli`, `embedded-ml`, `mlops`, `csv`

**Social preview**: upload `assets/social-preview.png` (2560×1280, 2:1).

**Settings to enable**
- Security → Private vulnerability reporting: **on**. `SECURITY.md` and
  `CODE_OF_CONDUCT.md` point reporters there.
- Issues: on. Discussions: optional.
- Branch protection on `main`: require the `ci` workflow checks once they have been
  observed passing.
- Environment `crates-io` with required reviewer and the `CARGO_REGISTRY_TOKEN`
  secret, used only by `publish-crates.yml`.
- Actions → General → Workflow permissions: **read repository contents** (the
  release workflow raises its own permission only for the draft-release job).

**Labels used by templates**: `bug`, `enhancement`, `feedback`, `good first issue`.
