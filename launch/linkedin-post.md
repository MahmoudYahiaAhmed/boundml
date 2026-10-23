# LinkedIn post (draft, first person, not posted)

Shipping a small ML model inside an application is often harder than training it.

The model itself might be a 30-feature MLP. But to ship it you need a Python training
pipeline, an export step, preprocessing re-implemented by hand, and a promise that
the feature order still matches. Every update risks breaking something silently.

I built BoundML to try a simpler path for Rust applications. It is an open-source CLI
and library that:
→ trains small dense models directly from CSV, in Rust
→ packages the feature schema, fitted preprocessing, weights and self-tests together
→ plans tensor memory and rejects configurations over budget before allocating
→ runs predictions inside your Rust app, with no Python and no framework
→ tells CI whether a model update is compatible or breaking

What I measured, on one machine and the same weights:
• 9 ms from process start to first prediction (Python + PyTorch: 1.49 s)
• 6.3 MiB peak process memory (PyTorch: 202 MiB)
• tract, a mature Rust runtime, is 2.5–3.3× faster on large batches. That's in
  the benchmarks too.

It's an early release. CPU only, Windows-tested so far, small models only, and no
users yet. That's why I'm sharing it: if you ship small models inside software, I'd
value 20 minutes of your feedback on a real workload.

Repo, benchmarks and the "what's verified" page: [link]

#rustlang #machinelearning #mlops
