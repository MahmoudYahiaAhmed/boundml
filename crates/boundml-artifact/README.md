# boundml-artifact

Load, verify and run [BoundML](../../README.md) model packages inside a Rust
application. A package bundles the feature schema, fitted preprocessing, graph,
weights, bounds, provenance and self-test vectors.

```rust,no_run
use boundml_artifact::Package;
use std::sync::Arc;

let package = Arc::new(Package::load("model.boundml".as_ref())?);
let mut p = package.predictor(64)?;                 // one per worker thread
let mut raw = vec![None; p.n_features()];
p.fill_named([("temperature", Some(20.0)), ("load", Some(0.5))], &mut raw, 0)?;
let mut out = vec![0.0f32; p.output_cols()];
p.predict_raw(&raw, 1, &mut out)?;                  // allocation-free
# Ok::<(), boundml_artifact::ArtifactError>(())
```

Format and trust model: [docs/package-format.md](../../docs/package-format.md).
