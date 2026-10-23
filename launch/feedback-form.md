# Workload feedback form (draft)

Use as a GitHub issue form (`.github/ISSUE_TEMPLATE/workload_report.yml` implements
it) or as a short interview guide. Ask about the data's shape, never the data itself.

1. **Your application**: what does it do, and where does the prediction run?
   (desktop app / local service / gateway / server / other)
2. **The model today**: task, rows, features, how often it is retrained, and what
   trains it (scikit-learn, PyTorch, XGBoost, hand-written, ...).
3. **Shipping and updating**: how do the model, its preprocessing and its feature
   order reach the application? What went wrong the last time you updated it?
4. **What you tried with BoundML**: which commands or APIs, and how long it took to
   get a first prediction.
5. **Did it work?** Yes / partly / no. If not, which step failed, and with which
   `BML-` code?
6. **Blockers**: which missing capability would stop you from using it? (Categorical
   features, trees, GPU, wasm, signing, Python bindings, platform, licence, other.)
7. **Value**: which part helped, if any? (bundled preprocessing, schema checks, memory
   plan, `compare`, embedding API, single binary, none)
8. **Alternatives**: what would you use instead, and why?
9. **Likelihood**: would you use it on a real workload in the next 3 months?
   (no / maybe / yes). What would have to change?
10. May we follow up? (optional contact, only if you want to be contacted)
