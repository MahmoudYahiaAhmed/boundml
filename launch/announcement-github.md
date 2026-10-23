# BoundML 0.1.0: small models, simple deployment, native Rust (early release)

If you've shipped a small prediction model inside a Rust application, you know the
extra work:
- a Python training script;
- an export step;
- preprocessing re-implemented by hand in Rust, which must stay in sync;
- feature order that nobody checks;
- no warning when an update quietly changes what the model expects.

BoundML is my attempt to make that workflow one tool for the narrow case of **small
dense models on numeric features**:

```bash
boundml init demo
boundml train demo/model.toml --checkpoint demo/ckpt
boundml pack demo/ckpt --out demo/model.boundml
boundml predict demo/model.boundml --data demo/data/test.csv --out demo/pred.csv --preview 3
```

The package carries the feature schema, the preprocessing fitted on the training data,
the weights, self-test vectors and a planned tensor-memory budget, and all of it is
checked when you load it. In your app:

```rust
let package = Arc::new(Package::load("model.boundml".as_ref())?);
let mut p = package.predictor(1)?;
p.fill_named([("temperature", Some(20.0)), ("load", Some(0.5))], &mut raw, 0)?;
p.predict_raw(&raw, 1, &mut out)?;
```

What it is **not**: a PyTorch replacement, a GPU framework, or the fastest batch
runtime. tract is 2.5–3.3× faster than BoundML on 256-row batches in my
benchmarks, and the benchmark page says so. BoundML is ahead on start-up, memory and
single-row calls, and it removes the glue code.

Everything is reproducible: [benchmarks](docs/benchmarks.md), [what's verified
and what isn't](docs/release-status.md). It has only been built and tested on
Windows so far.

I'm looking for people with real workloads to try it and tell me what blocks them.
There is a *Workload report* issue template for exactly that.
