# boundml-data

Numeric CSV input for [BoundML](../../README.md):
- features mapped by column name, with explicit missing-value and extra-column policies;
- standardization fitted on training data only;
- bounded, deterministic, exactly resumable minibatch streaming;
- deterministic train/valid/test splits.

Policies: [docs/data.md](../../docs/data.md).
