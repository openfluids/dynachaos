# Claims checklist

| Claim location | Claim | Status |
|---|---|---|
| README lead; JOSS Summary | dynachaos is a reusable tool for simulated or measured dynamical-systems time signals | supported-by: package API surface and test suite |
| README/JOSS Rust wording | Rust kernels accelerate selected heavy diagnostics | supported-by: test suite parity plus scale_envelope artifact |
| README Benchmarks | CI-mode Grassberger-Procaccia logistic N=1000 Rust speedup is 42.95x | supported-by: `benchmarks/results/scale_envelope.md` |
| README Benchmarks | Dense recurrence/RQA has an 8*N^2 byte analytical matrix envelope | supported-by: `benchmarks/results/scale_envelope.md` |
| README fallback policy; JOSS Software design | Pure-Python paths are parity/portability paths, not large-run performance claims | supported-by: test suite; unsupported performance implication removed |
| README reproduction gallery | The reproduction gallery is the flagship application and stress test, not the package boundary | supported-by: reproduction pipeline; scope claim reframed |
| JOSS reproducibility text | Fixed seeds and selected cache checks guard the fast figure pipeline | supported-by: test suite; universal figure-output claim softened |
| JOSS package scope | External trajectories can be analysed by the diagnostics | supported-by: diagnostics accepting array-like trajectory/time-series inputs |
| Removed/softened claims | Hype or unqualified superlatives about performance/scope | unsupported-removed |
