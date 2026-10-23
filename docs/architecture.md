# Architecture

## Crates and dependency boundary

| Crate | Purpose | Notable dependencies |
|---|---|---|
| `boundml-core` | shapes, graph IR, kernels, reverse-mode AD, memory planner, arena executor, SGD/AdamW, RNG | `serde`, `thiserror` only |
| `boundml-data` | schema, CSV reader, fitted preprocessing, bounded shuffle streaming, split | `csv` |
| `boundml-artifact` | package format, `Predictor` (embedding API), `compare`, checksums, atomic writes | `safetensors`, `sha2`, `serde_json` |
| `boundml-train` | TOML config, training loop, metrics, checkpoints, `pack`, `evaluate` | `toml` |
| `boundml-cli` | `boundml` binary, local HTTP server, PyTorch/Keras bundle import | `clap`, `axum`, `tokio`, `tower-http` |
| `examples/embed` | embedding example; depends only on `boundml-artifact` | — |
| `bench` | benchmark and stress workers | — |

Applications embed `boundml-artifact` (49 crates in its dependency tree, including no
training, HTTP or async code). The native computation is all BoundML code: no ML
framework, BLAS or ONNX runtime is linked.

## Tensors

- Every tensor is a dense, row-major **f32** matrix `[rows, cols]`. A scalar is
  `[1, 1]`; a bias is `[1, n]`.
- A dimension is either fixed or the symbolic batch `B`. `B` is bound per execution,
  in `1..=max_batch`. Fixed dims must be `1..=2^24`, and elements per tensor
  `<= 2^28`. All shape, element, byte and arena-offset arithmetic is checked
  (`Shape::elements`, `Shape::bytes`, planner `checked_add`) and fails with
  `BML-C001`/`BML-C002`.
- **Broadcasting** (`Add`, `Mul` only): the right operand `b` may equal `a`'s shape,
  or have size 1 in a dimension, which then broadcasts over `a`. Gradients for `b`
  are summed over the broadcast dimensions.
- f64 is used only for loss accumulation, normalization statistics and test
  references. Production tensors are f32.

## Operators

| Op | Forward | Backward (accumulated `+=`) | Values retained for backward |
|---|---|---|---|
| `matmul{ta,tb}` | `op(a) @ op(b)` | `dA' = G op(B)^T`, `dB' = op(A)^T G`, transposed back when `ta`/`tb` | both operands |
| `add` | `a + b` (broadcast) | `da += g`; `db += reduce(g)` | none |
| `mul` | `a * b` (broadcast) | `da += g*b`; `db += reduce(g*a)` | both operands |
| `relu` | `max(x,0)` | `g` where `y > 0`; **derivative at 0 is 0** | output |
| `sigmoid` | stable two-branch form | `g*y*(1-y)` | output |
| `tanh` | `tanh` | `g*(1-y^2)` | output |
| `softmax` | row-wise, max-shifted | `y*(g - sum(g*y))` | output |
| `log_softmax` | `x - max - log(sum(exp(x-max)))` | `g - exp(y)*sum(g)` | output |
| `transpose` | `x^T` | `g^T` | none |
| `sum_all`, `mean_all` | → `[1,1]` | broadcast `g` (÷ n for mean) | none |
| `mse_loss` | mean of `(p-t)^2` over all elements | `2(p-t)/n` | both |
| `bce_with_logits_loss` | mean of `max(z,0) - z*y + log1p(exp(-|z|))` | `(sigmoid(z)-y)/n` | both |
| `cross_entropy_loss` | mean over rows of `lse(z) - z[t]`; `t` = class index stored as f32, must be an integer in `0..K` (else `BML-C008`) | `(softmax(z) - onehot)/rows` | both |

All kernels are allocation-free and scratch-free (they compute row statistics on
the fly), so `kernel_scratch_bytes` is 0 by construction. Unsupported operators
cannot be represented: `GraphDef` deserialization rejects unknown `op` tags and
unknown fields.

## Graph validation (`Graph::new`)

Topological order (operands must precede uses), arity and shape rules, a single
`input` and at most one `target`, a scalar loss that depends on parameters, a target
that is never differentiated and not used by the inference output, no unused nodes or
parameters, unique names `[A-Za-z0-9_.-]{1,128}`, and at most 4096 nodes and 1024
parameters. One graph carries both an inference `output` and a training `loss`.
Execution touches only the ancestors of the root required by the mode.

**Row independence.** The inference path must compute each output row only from the
same input row, so a prediction never depends on which other rows share its batch.
Validation therefore rejects, on the inference path: tensors with the batch dimension
as columns; transposes or reductions of batch tensors; and matmuls that contract over
or transpose the batch dimension. The output must have the batch dimension as rows.
Training losses may still reduce over the batch.

## Differentiation semantics

- Reverse mode over the static graph. Forward runs in topological order. Backward
  runs in reverse over nodes that require gradients.
- **Fan-out**: every contribution is *accumulated* into the operand's gradient
  buffer. A gradient buffer is zeroed when it first becomes live (planner
  `zero_list`), and parameter gradients are zeroed at the start of every backward
  pass. `x*x`, shared weights and `h^T h` are covered by
  `tests/gradcheck.rs::fan_out_and_all_ops`.
- **Lifetime / repeated calls**: `Executor::forward_backward` recomputes the
  activations and overwrites the gradients on each call. Gradients are never
  accumulated across calls, and activations live only within one call. Parameter
  gradients stay readable via `param_grads()` until the next call.
- Legitimate zero gradients (for example, dead ReLUs) are not treated as errors.
- Non-finite inputs, targets, losses and gradients are rejected (`BML-C007`).

## Memory planning and the enforced boundary

`Plan::new(graph, mode, max_batch, planner)` runs **before** any tensor buffer is
allocated:

1. Schedule: active nodes run forward at steps `0..A`. In training, the backward
   for the node at forward step `t` runs at step `2A-1-t`.
2. Each value buffer lives from its definition (inputs from step 0) to its last
   forward use or last backward read. A gradient buffer lives from the first consumer
   backward step that writes it to its own backward step. Parameter gradients live
   for the whole step.
3. Intervals are **inclusive**, so an operator's output never shares memory with its
   own operands.
4. `Reuse` places buffers largest-first at the lowest offset that does not collide
   with any time-overlapping buffer. `Conservative` gives every buffer its own
   region. Offsets are aligned to 64 bytes.
5. `verify_no_conflicts` re-checks every plan. A property test
   (`plan::tests::reuse_planner_never_overlaps_live_buffers`) checks random interval
   sets for no overlap and for capacity at or above peak live bytes.

**Enforcement scope.** For packages, the declared `bounds.sessions` is enforced:
every `Predictor` of a loaded `Package` (including the one `verify` uses internally)
holds a reservation until it is dropped, and the session past the allowance fails
with `BML-A010` *before* its arena is allocated. The allowance belongs to one loaded
`Package` instance (and its cheap clones); loading the same directory twice gives two
independent allowances. Lower-level `boundml_core` APIs (`InferenceSession`,
`Executor`) are not accounted: callers using them directly manage their own sessions.
`Predictor` preprocesses raw values directly into the arena's planned input buffer, so
it owns no tensor storage outside the plan.

`MemoryReport` lists, for a configuration (`max_batch`, `sessions`, optimizer):
shared weights, inputs, outputs, activations, values retained for backward, activation
gradients, parameter gradients, optimizer state, kernel scratch, alignment padding,
the sum of buffers without reuse, and arena capacity per session after reuse.
`total_planned_bytes = weights + sessions × arena + optimizer state`, plus, for
training, the validation executor's arena and the training loop's batch/evaluation
staging buffers. Data-loader buffers (the training shuffle buffer) are reported as
`data_loader_bytes` but not budgeted.
`check_budget` rejects a plan over budget with `BML-C005`, giving requested bytes,
available bytes and the largest contributors. This happens in `Executor::from_plan`'s
callers *before* `vec![0.0; arena]` runs.

**What is bounded**: planned tensor storage for the stated configuration.
**What is not**: package/checkpoint loading buffers, graph and manifest metadata, CSV
parsing and the training shuffle buffer (`shuffle_buffer × features × 4` bytes, set
in config), thread stacks, allocator overhead, HTTP buffers and process overhead.
It is not a bound on RSS, and it does not guarantee that the OS can satisfy an
allocation.

| Claim | How it is established |
|---|---|
| Plan never overlaps live buffers | analysis + runtime assertion (`arena::split`) + property test |
| Over-budget plans fail before allocation | API order + tests (`budget_rejected_before_allocation`, `over_budget_training_is_rejected_before_allocation`, CLI exit code 4) |
| No heap allocation in `predict_into`, `forward_backward`, `Optimizer::step` after construction | counting global allocator test (`tests/alloc_free.rs`) |
| `Predictor::predict_raw` (preprocessing + execution) does not allocate | counting allocator test (`boundml-artifact/tests/alloc_free.rs`) |
| Process memory stays flat under sustained use | measured (`docs/results/stress.json`); not a guarantee |

## Unsafe code

The only `unsafe` in production code is `boundml-core/src/arena.rs::split`. It
returns one mutable slice and up to three shared slices of the same arena. Before
creating them it asserts, in release builds too, that every range is in bounds and
that the write range is disjoint from every arena read range. Unit tests cover the
success, aliasing and out-of-bounds cases.

## Ownership and concurrency

`Model`/`Package` hold the validated graph and weights. They are immutable and shared
via `Arc`. `InferenceSession`/`Predictor`/`Executor` own their arena, are `Send`,
and are not shared. Each worker thread creates its own, so mutable scratch and
gradients are never shared implicitly. The server keeps a fixed pool of `workers`
predictors, and that count may not exceed the package's planned `sessions`, which is
what its budget covers.

## Determinism

The RNG is xoshiro256** with serializable state. Initialization, shuffling (seeded
per epoch from `seed` and the epoch number) and the order of floating-point operations
are fixed. Interrupted+resumed and uninterrupted runs produce bitwise-identical
parameters, optimizer state and history on the same binary and CPU (test
`interrupted_and_resumed_run_matches_uninterrupted`). Bitwise reproducibility across
CPUs, compilers or future optimized kernels is **not** promised.

## Artifact trust model

SHA-256 values in `MANIFEST.sha256` and `manifest.files` detect corruption and
truncation. They do **not** authenticate the author, because anyone who can modify a
package can recompute them. Packages contain no code. Loading treats every field as
untrusted: sizes are checked from metadata before reading, symlinks are rejected,
only fixed file names are opened (no paths come from the manifest), tensors must be
finite f32 rank-2 with exactly the declared names and shapes, and the graph is
re-validated. The memory plan is recomputed by the installed runtime and never read
from the file. Test vectors detect drift on their own inputs only. Signing is out of
scope for this release. If it is added, its key management needs its own
specification.
