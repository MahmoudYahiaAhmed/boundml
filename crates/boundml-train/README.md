# boundml-train

Training for [BoundML](../../README.md): TOML configuration, the training loop
(SGD/AdamW, validation), metrics against constant baselines, atomic and exactly
resumable checkpoints, and packaging into inference packages.

Example: `cargo run --release -p boundml-train --example train_and_embed`.
