# BoundML desk research: prior art and gap check

- Date: 2026-10-23. All "accessed" dates below are 2026-10-23.
- Method: primary sources only (official repo README, official docs, crates.io / PyPI JSON APIs, GitHub release pages). Pages were read through an automated fetch-and-summarise tool, so treat specific numbers as "as reported by the page on that date".
- Conventions:
  - **documented**: stated on a page checked.
  - **not documented in pages checked**: the pages I read say nothing about it. This does *not* mean the feature is absent.
  - **unverified**: I could not load or confirm the source.
- Version dates come from crates.io `created_at` or PyPI `upload_time` unless noted otherwise. GitHub README pages often showed an empty "Releases" box, so they were not used for versions.

## 1. Project-by-project notes

### Rust frameworks and runtimes

**Burn**: https://github.com/tracel-ai/burn, https://crates.io/api/v1/crates/burn
- Latest: `burn` 0.22.1, published 2026-10-20 (crates.io).
- Built for training and inference: autodiff, a training dashboard, and gradient checkpointing (`gradient_checkpointing()`). It has two CPU backends. `cpu` is CubeCL/LLVM JIT. `flex` is pure-Rust eager and runs on Wasm and `no_std`; per the README it is "the only backend usable in no_std".
- Imports: ONNX goes through `burn-onnx`, which generates Rust code and supports "a limited set of ONNX operators". PyTorch and safetensors weights load via `burn-store`. `burn-dataset` provides loading and transforms, e.g. a custom CSV dataset.
- Native records use the "burnpack" format as of 0.22. Legacy `.mpk`, `.bin` and JSON records cannot be read directly, so migration goes through `burn-store`. That is a real record-format break.
- Not documented in pages checked: bundling fitted preprocessing with the model, feature-schema validation, static memory budgets for inference, bundled test vectors, package diff/compatibility check.

**Candle**: https://github.com/huggingface/candle, https://crates.io/api/v1/crates/candle-core
- Latest: `candle-core` 0.11.0, published 2026-06-26.
- Supports training ("Model training" is listed, with a PyTorch-like API; optimizers come from the external `optimisers` crate). Has an optimized CPU backend (optional MKL or Accelerate) and Wasm browser demos.
- Loads safetensors, npz, ggml/GGUF and PyTorch files. `candle-onnx` evaluates ONNX. Its stated goal is serverless and Python-free inference with lightweight binaries.
- Not documented in pages checked: preprocessing bundling, schema, memory planning.

**CubeCL**: https://github.com/tracel-ai/cubecl, https://crates.io/api/v1/crates/cubecl
- Latest: 0.11.3, published 2026-10-22.
- A kernel language and JIT for writing `#[cube]` Rust functions. Targets CUDA, HIP, Metal, Vulkan, WebGPU, and CPU through LLVM.
- The README says "currently in alpha", expects breaking changes between minor versions, and recommends pinning a version. Burn uses it in production.

**Luminal**: https://github.com/luminal-ai/luminal, https://crates.io/api/v1/crates/luminal
- Latest on crates.io: 0.2.0, published 2024-03-01. The repo is clearly more active than that, so the crates.io release lags the repo.
- Described as "a high-performance general-purpose inference compiler". It compiles static graphs ahead of time with symbolic dimensions and searches compiler decisions. CUDA is the primary backend; Metal is available behind a feature.
- The README says autograd is "modeled ahead of time". It has a search memory cap (`SEARCH_MEMORY_MIB`), but a memory-planning mechanism is not documented in pages checked.

**dfdx**: https://github.com/coreylowman/dfdx, https://crates.io/api/v1/crates/dfdx
- Latest: 0.13.0, published 2023-07-27. There has been no release in about 3.25 years. The README still says "Still in pre-alpha state", and the repo is not marked archived.
- Tensor shapes are checked at compile time. Has training (Sgd, Adam, AdamW, RMSprop) and `.npz` checkpoints, on CPU with optional CUDA.
- Treat it as dormant for dependency purposes, but it remains the clearest Rust prior art for "typed shapes".

**tract (Sonos)**: https://github.com/sonos/tract, https://crates.io/api/v1/crates/tract-onnx
- Latest: 0.23.9 (`tract-core` created 2026-10-15). The 0.21.x line still gets backports (0.21.19 on 2026-10-15).
- An inference-only engine; training is not documented in pages checked. Formats: ONNX (load only), NNEF/tract-OPL (load and save), TFLite (load and save), and TF1 frozen graphs.
- Runs on CPU (x86, ARMv6/7/8, SVE), Wasm, and Metal/CUDA. CI covers "embedded targets". `prepare()` optimises and compiles for the runtime, and the README suggests translating to NNEF at build time. Memory planning details are not documented in pages checked.
- tract-OPL promises compatibility within a minor version (0.x.y models run on 0.x.z with z >= y). This is a documented artifact-compatibility rule.

**RTen**: https://github.com/robertknight/rten, https://crates.io/api/v1/crates/rten
- Latest: 0.27.0, published 2026-10-02.
- Runs ONNX models on CPU only (AVX2, AVX-512, NEON, Wasm SIMD), with a JS/Wasm API. Supports f32 and int8 weights.
- Ships supporting pre- and post-processing crates (`rten-imageproc`, `rten-text`, `rten-imageio`). These are domain libraries, not fitted preprocessing bundled into the model.
- Not documented in pages checked: training, memory planning, schema.

**Aprender (paiml)**: https://github.com/paiml/aprender, https://crates.io/api/v1/crates/aprender, https://paiml.github.io/aprender/architecture/provable-contracts.html, https://docs.rs/crate/aprender/0.63.0/source/book/src/lib/format.md
- Latest: 0.71.4, published 2026-10-14. The README text says v0.69.1, so the README lags crates.io.
- A "complete ML framework in pure Rust" with one `apr` CLI (113 commands per the README).
  - Classical ML: linear and logistic regression, trees, RF, GBM, NB, KNN, SVM, k-means, PCA and others.
  - LLM features: run, chat, serve, finetune, distill and quantize.
  - MLP as a classical-model API is not documented in pages checked.
- Has its own `.apr` container (v1 "APRN", v2 "APR\0").
  - The `format` module covers conversion to and from GGUF, safetensors and ONNX, plus `ModelCard` metadata, a `signing` submodule (algorithm not stated), "golden-output references", and diff/lint tooling.
  - The CLI includes `apr validate`, `apr diff` ("tensor-by-tensor"), `apr inspect` and `apr stamp`.
- "Provable contracts" are YAML files of equations, invariants and `FALSIFY-*` tests for **kernels and CLI commands**. They are enforced by a `#[contract]` proc macro, by tests, and by `pv lint` in CI.
  - The pages disagree on the count: 405 on the book page, 1837 or 2001 in the README.
  - Whether contracts cover model files or feature schemas is not documented in pages checked.
- Not documented in pages checked: fitted preprocessing stored in `.apr`, feature-schema validation at inference, a static memory budget, a package-level compatibility verdict. `apr serve plan` ("capacity plan before run") targets LLM serving.

**irithyll**: https://docs.rs/irithyll, https://crates.io/api/v1/crates/irithyll
- Latest: 10.0.1. crates.io gives `created_at` 2026-05-11, while the docs.rs page showed 2026-09-25 (unverified which date is the release date).
- A streaming ML library ("There is no training set") built around one `StreamingLearner` trait with `train_one`/`predict`. Models include SGBT (streaming boosted trees), RLS, SGD linear, and many neural and sequence models. Includes drift detectors.
- Has streaming preprocessing (normalizer, min-max, one-hot, target encoder, hashing) chained in pipelines, and a `FeatureType` enum (continuous or categorical).
- Saves JSON (default) or bincode checkpoints, with ONNX export behind a feature. `irithyll-core` is `#![no_std]` and exports trees as 12-byte packed nodes, tested on `thumbv6m-none-eabi`. Schema validation beyond `FeatureType` is not documented in pages checked.

**Linfa**: https://github.com/rust-ml/linfa, https://crates.io/api/v1/crates/linfa
- Latest: 0.8.1, published 2025-12-23, about 10 months ago.
- Classical ML: OLS/GLM, logistic regression (with partial fit), elastic net, LARS, SVM, trees, clustering and PCA. `linfa-preprocessing` covers normalization, whitening, count vectorizer and tf-idf. Has a `wasm-bindgen` feature and optional BLAS.
- Neural networks, serde/serialization, and a packaged model artifact are not documented in pages checked.

**SmartCore**: https://github.com/smartcorelib/smartcore, https://crates.io/api/v1/crates/smartcore
- Latest: 0.6.16, published 2026-10-12. Active; an ndarray-only trait refactor is in progress.
- Classical ML: linear models, logistic regression, RF, SVM, KNN, NB and clustering, plus encoders and model selection. Serde is optional. Wasm/WASI is targeted by default.
- Neural networks are not documented in pages checked. No packaged artifact beyond serde is documented.

**MicroFlow**: https://github.com/matteocarnelos/microflow-rs, https://crates.io/api/v1/crates/microflow
- Latest: 0.1.3, published 2024-05-31. No release in about 17 months.
- A TinyML inference engine. The `#[model("x.tflite")]` proc macro generates Rust code and embeds weights at compile time, and the runtime is `no_std`. Supported ops are a small quantized set: FC, Conv2D, DepthwiseConv2D, AvgPool, ReLU/ReLU6/Softmax.
- Training is not documented in pages checked. This is prior art for compile-time model embedding in Rust.

**Enzyme Rust / `std::autodiff`**: https://enzyme.mit.edu/rust/ (**unverified**: TLS certificate rejected by the fetcher), https://doc.rust-lang.org/nightly/std/autodiff/index.html
- Nightly-only and experimental (tracking issue #124509). Provides the `#[autodiff_forward]` and `#[autodiff_reverse]` macros.
- Requires `lto="fat"`, debug builds may fail, and `dyn Trait` arguments are not supported. It is not usable on stable Rust for a product dependency.

### Non-Rust reference points

**tinygrad**: https://github.com/tinygrad/tinygrad, https://pypi.org/pypi/tinygrad/json
- Latest: 0.14.0 on PyPI (2026-08-24). A Python stack with autograd, nn/optim, lazy execution that fuses kernels, and many backends (CPU, CUDA, Metal, WebGPU, QCOM and others).

**MLX**: https://github.com/ml-explore/mlx, https://pypi.org/pypi/mlx/json
- Latest: 0.32.4 on PyPI (date not captured).
- An Apple array framework with Python, C++, C and Swift APIs, autodiff, lazy evaluation and unified memory. Besides Apple silicon it runs on Linux CUDA and Linux CPU. Training is supported (including LoRA examples).

**PyTorch AOTInductor**: https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html
- Docs version 2.14. Compiles `torch.export` models into a `.pt2` package (PT2 Archive Spec) that can contain a shared library. C++ loads it with `AOTIModelPackageLoader`, so Python is needed only at compile time.
- Supports CPU, CUDA and Intel GPU, and dynamic dims through `torch.export.Dim`. With `AOTI_RUNTIME_CHECK_INPUTS=1` it checks inputs against the export-time size, dtype and strides.
- Training is not documented in pages checked.

**ExecuTorch**: https://github.com/pytorch/executorch/releases, https://docs.pytorch.org/executorch/stable/compiler-memory-planning.html, https://docs.pytorch.org/executorch/stable/bundled-io.html
- Latest: v1.5.1, released "23 Sep" (the page shows no year; most likely 2026, unverified).
- **Memory planning** runs ahead of time, as the last pass before emission.
  - `SymShapeEvalPass` resolves dynamic shapes to upper bounds, then `MemoryPlanningPass` places every mutable tensor in fixed-size arenas.
  - Algorithms are naive and greedy best-fit, and custom passes are possible. `mem_id` maps tensors to separate arenas (e.g. SRAM vs DRAM), and per-device buffers are supported.
  - Planning graph inputs and outputs is optional; if they are not planned, the runtime must supply those buffers.
- **Bundled Program** (`.bpte`, FlatBuffer) wraps the program with per-method test inputs and optional expected outputs.
  - Construction checks method names and input dtypes and shapes.
  - At runtime, C++ `load_bundled_input` and `verify_method_outputs` run the checks, and the example runner has `--output_verification`.
  - Tolerance settings are not documented on that page.
- Authoring and export happen in Python. Training in Rust and fitted tabular preprocessing are out of scope (not documented in pages checked).

**ONNX Runtime on-device training**: https://onnxruntime.ai/docs/get-started/training-on-device.html, https://github.com/microsoft/onnxruntime/releases
- Latest ORT: v1.31.0, released "09 Oct" (year not shown).
- Training artifacts are generated offline in Python and then trained on the device. An inference ONNX model can be exported on the device.
- The pages checked carry no deprecation notice for training (I did not check further). Supported ops and losses are not documented on that page.

**LiteRT on-device training / LiteRT (TFLite) Micro**: https://developers.google.com/edge/litert/models/ondevice_training, https://github.com/tensorflow/tflite-micro, https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md
- Training: a `tf.Module` exposes `train`, `infer`, `save` and `restore` as signatures, converted with `SELECT_TF_OPS` and resource variables. Weights persist as TF1 checkpoints. Requires TF >= 2.7 on the authoring side; on Android it runs through the Java or C++ API.
- TFLM memory:
  - One caller-provided tensor arena, split into head (shared buffers placed by `GreedyMemoryPlanner`), temporary, and tail (persistent).
  - An optional *offline* plan stored in model metadata ("OfflineMemoryAllocation") fixes tensor offsets, and `RecordingMicroInterpreter` reports actual usage.
  - Training is not documented in pages checked.
- Latest release: not shown (no Releases section on the repo page).

**IREE**: https://iree.dev/guides/deployment-configurations/, https://github.com/iree-org/iree/releases
- Latest stable: v3.13.0 (2026-10-19). An MLIR compiler and runtime.
- Deployment targets: CPU, CPU bare-metal ("minimal platform dependencies"), Vulkan, ROCm, CUDA and Metal. Memory-planning details are not documented on the page checked.

**DeepSpeed ZeRO-Offload**: https://www.deepspeed.ai/tutorials/zero-offload/
- Moves optimizer memory and compute to the host CPU (DeepSpeedCPUAdam). It claims models of up to 13B parameters on one GPU. Configured with `offload_optimizer.device: "cpu"` and ZeRO stage 2. This is GPU-scale training and is not relevant to small dense models.

**TensorFlow Transform**: https://www.tensorflow.org/tfx/transform/get_started
- Full-pass analyzers (`tft.mean`, vocabularies) compute constants that are frozen into `transform_fn`, a SavedModel graph that can be embedded in serving. This is the canonical pattern for "fitted preprocessing shipped with the model".
- Input layout comes from a feature spec / `DatasetMetadata`. Requires Python, TF and Beam.

**Edge Impulse EON Tuner**: https://docs.edgeimpulse.com/docs/edge-impulse-studio/eon-tuner
- A hosted Bayesian search over DSP (preprocessing) and model hyperparameters together, constrained by RAM, flash and latency on the configured target device. This is prior art for "preprocessing + model chosen under a memory budget", but it is a SaaS studio, not a library.

**River**: https://github.com/online-ml/river, https://pypi.org/pypi/river/0.26.1/json
- Latest: 0.26.1 on PyPI (2026-08-21). Python online ML (`learn_one`/`predict_one`) with `compose.Pipeline` for preprocessing plus model. Persistence is not documented in pages checked.

**Vowpal Wabbit**: https://github.com/VowpalWabbit/vowpal_wabbit, https://pypi.org/pypi/vowpalwabbit/9.11.9/json
- Latest: 9.11.9 on PyPI (2026-09-27). Online learning in C++ with feature hashing, contextual bandits and RL. The repo is not marked archived.

**llama.cpp**: https://github.com/ggml-org/llama.cpp, https://github.com/ggml-org/llama.cpp/releases
- Latest tag b11829 ("22 October", 2026 by footer). LLM and VLM inference in C/C++ with no dependencies, many backends, and 1.5- to 8-bit quantization.
- Training and fine-tuning are not documented in the README text checked. Not a tabular competitor.

## 2. Route survey

| Route | User | Painful task | Existing alternatives (what they do well) | Possible remaining gap | Migration burden | Small-team feasibility |
|---|---|---|---|---|---|---|
| General tensor/autograd framework | Rust ML engineers | Train and run nets without Python | Burn (training, many backends, no_std Flex), Candle (PyTorch-like, HF models), tinygrad, MLX | Little; crowded and well funded | High (new API, record formats) | Low: competing head-on is infeasible |
| Portable GPU kernels/compiler | Framework authors | One kernel source for many GPUs | CubeCL (alpha), IREE (stable 3.x), Luminal, tinygrad | Narrow, expert niche | High | Low |
| CPU inference without Python | Rust app devs | Ship a trained model in a binary | tract (ONNX/NNEF, embedded CI), RTen (ONNX, Wasm), Candle, AOTInductor (C++), ExecuTorch | Tabular-specific contract (schema + preprocessing) around inference | Low-medium (export ONNX) | Medium: wrap rather than replace |
| Memory-constrained training/offload | LLM trainers | Fit large models in VRAM | ZeRO-Offload, Burn gradient checkpointing | Not relevant for small dense models | n/a | Low relevance |
| Typed shapes & numerical correctness | Safety-minded Rust devs | Catch shape/dtype bugs before runtime | dfdx (compile-time shapes, dormant since 2023-07), Aprender YAML kernel contracts, AOTI input checks | Maintained, typed I/O contract at the *feature* level | Medium | Medium |
| On-device/continual training | Mobile/edge app devs | Personalise models locally | ORT on-device training, LiteRT train signatures, irithyll (streaming, no_std core) | Rust-native small-model retrain + re-verify loop | Medium-high (Python artifact generation for ORT/LiteRT) | Medium |
| Streaming/tabular ML in apps | Rust app devs | Classical models in-process | Linfa, SmartCore, irithyll, River (Py), VW (C++) | Packaged artifact with schema + fitted preprocessing + self-test | Low (same algorithms) | High: narrow scope |
| Browser/wasm/embedded/mobile | Edge/web devs | Tiny footprint, bounded memory | TFLM (arena + offline plan), MicroFlow (compile-time embed), ExecuTorch, tract, RTen Wasm, Burn Flex no_std | Declared memory budget checked at package time for *Rust-trained* models | Medium | Medium |
| Preprocessing consistency + packaging + update compatibility | Teams shipping model updates | Train/serve skew; silent breaking updates | TF Transform (fitted transform in SavedModel), ExecuTorch Bundled Program (self-test), tract-OPL compat rule, Aprender `apr diff`/`validate` | One artifact combining schema + fitted preprocessing + self-tests + compat verdict, with no Python | Low-medium | High: most plausible BoundML wedge |
| Scientific differentiable programming | Researchers | Gradients of arbitrary Rust code | Enzyme / `std::autodiff` (nightly, fat LTO), JAX | Stable-Rust AD | n/a | Low (compiler work) |
| Local LLM inference/fine-tuning | Desktop AI devs | Run and quantize LLMs locally | llama.cpp, Candle, MLX, Aprender `apr` | Saturated | n/a | Low |

## 3. Closest competitors to BoundML

Legend: D = documented, ND = not documented in pages checked, U = unverified.

| Capability | Burn | Aprender | ExecuTorch | tract / RTen | Linfa / SmartCore |
|---|---|---|---|---|---|
| Train in Rust | D (core purpose) | D (`aprender-train`, `apr train`; classical ML + LLM finetune) | ND (Python authoring) | ND (inference only) | D (classical fit; no NN) |
| Embed in Rust app | D (incl. no_std via Flex) | D (crates + CLI; CPU default) | ND for Rust (C++ runtime) | D (both are Rust runtimes; RTen also Wasm) | D (library calls) |
| Small dense models (linear/logistic/MLP) | D for MLP; linear/logistic via nn layers | D linear/logistic; MLP API ND | D (any exported PyTorch) | D (any ONNX) | D linear/logistic; MLP ND |
| Fitted preprocessing bundled in artifact | ND (`burn-dataset` transforms are not in records) | ND | ND | ND (RTen preprocessing crates are separate libs) | ND (preprocessing exists as separate estimators) |
| Feature-schema validation at load/predict | ND | ND (`apr validate` scope not stated) | Partial: Bundled Program checks test-input dtype/shape at build time | ND | ND |
| Memory budget / static planning | ND (gradient checkpointing only) | ND (`apr serve plan` = LLM capacity plan) | **D** (AOT arena planning, mem_ids, greedy/naive) | ND | ND |
| Bundled self-tests (test vectors) | ND | Partial: "golden-output references" in `format` module, details ND | **D** (Bundled Program + `verify_method_outputs`) | ND | ND |
| Package diff / compatibility check | ND (record break at 0.22; migration via `burn-store`) | Partial: `apr diff` tensor-by-tensor; compat verdict ND | ND | Partial: tract-OPL minor-version compat rule (runtime vs file), not model-to-model diff | ND |
| Versioned single-file package | D (burnpack) | D (`.apr` v1/v2, `ModelCard`, signing module) | D (`.pte` / `.bpte`) | D (NNEF/OPL; ONNX/`.rten`) | ND (serde optional) |
| Maintenance (latest release) | 0.22.1, 2026-10-20 | 0.71.4, 2026-10-14 | v1.5.1, 23 Sep (year U) | tract 0.23.9 2026-10-15; RTen 0.27.0 2026-10-02 | Linfa 0.8.1 2025-12-23; SmartCore 0.6.16 2026-10-12 |

### Reading of the evidence

- **Nobody in the pages checked documents BoundML's whole workflow** (Rust training, fitted preprocessing and feature schema in the artifact, a planned memory budget, bundled test vectors, and a compatibility diff, all in one versioned package). Each piece exists somewhere else:
  - Fitted preprocessing in the artifact: TF Transform (Python/TF).
  - AOT memory planning and bundled test I/O: ExecuTorch (Python export, C++ runtime).
  - Arena and offline memory plan: TFLM.
  - Artifact-compat rule: tract-OPL.
  - Tensor diff, golden outputs and contracts: Aprender.
- **Aprender is the nearest Rust overlap**. It has pure-Rust training of classical models, its own versioned container, `validate`/`diff`/`lint`, "golden-output references", signing, and contract-driven CI. It is very broad and fast-moving (113 CLI commands, LLM-heavy, and its docs disagree on counts). Fitted preprocessing, feature schema and memory budget are not documented in pages checked. Before committing, read the `aprender-core` `format.rs` source directly to rule out schema or preprocessing storage.
- **ExecuTorch is the nearest conceptual match** for "planned memory + bundled self-test". However, it is PyTorch/Python-authored with a C++ runtime, and nothing in the pages checked covers tabular preprocessing or schema.
- **Burn** could host BoundML-style features as a layer on top: it has training, no_std and a versioned record format. Its 0.22 record-format break shows the compatibility pain is real.
- **Dormant or lagging**: dfdx (last release 2023-07-27), MicroFlow (2024-05-31), and Luminal on crates.io (2024-03-01; the repo is active). Linfa's last release was 10 months ago.
