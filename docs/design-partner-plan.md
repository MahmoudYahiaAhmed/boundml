# Design-partner evaluation plan

Status: **not started**. No external user has evaluated BoundML. Nothing in this
repository is evidence of demand.

## Goal

Find out whether 3–5 teams that ship small tabular models inside Rust applications
would use BoundML for a real workload, and what stops them.

## Who to recruit

Teams that (a) maintain a Rust desktop app, local service or Linux gateway, (b) ship
at least one numeric prediction model today, and (c) update that model at least
quarterly. Recruiting and contact require the project owner's authorization; none
has been made.

## Session outline (60–90 min, the partner's own data)

1. **Current workflow** (15 min): walk through the last model update end to end.
   Where are preprocessing and feature order defined? How is the deployed model
   checked? What broke last time?
2. **Guided task** (30–45 min): with their CSV, run `init` → edit config →
   `validate` → `train` → `pack` → `verify` → embed with the `Predictor` API or
   `serve`. Then retrain and run `compare`. Record time per step, errors met and
   documentation lookups.
3. **Debrief** (15 min), questions below.

## Questions

- Which part of today's workflow causes the most incidents or effort?
- Which of the package checks (schema, preprocessing, self-test, memory bound,
  compare) would have caught a past incident? Which are noise?
- What is missing that would block real use? (Expected: categorical features,
  gradient-boosted trees, early stopping, GPU, wasm, signing.)
- Would you train in BoundML, or only import from PyTorch/Keras?
- Unacceptable limitations? Acceptable throughput and memory?
- Would you use it on a real workload within 3 months? What would have to change?

## Success criteria (decided before sessions)

- At least 3 of 5 partners complete the guided task without maintainer help in under
  60 minutes.
- At least 2 of 5 name a past incident one of the checks would have prevented.
- At least 2 of 5 say they would trial it on a real workload, naming the blocking gap
  if any.

If these are not met, revise the positioning. One possible revision: offer the
contract layer (`Schema`, `Preprocessor`, test vectors, `compare`) as a sidecar for
tract/RTen/Burn packages (candidate B in `docs/decision.md`).
