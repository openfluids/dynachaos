# Claims checklist

| Claim location | Claim | Status |
|---|---|---|
| README lead; JOSS Summary | dynachaos is a reusable tool for simulated or measured dynamical-systems time signals | supported-by: package API surface and test suite |
| README/JOSS Rust wording | Rust kernels accelerate selected heavy diagnostics | supported-by: test suite parity plus scale_envelope artifact |
| README Benchmarks | CI-mode Grassberger-Procaccia logistic N=1000 Rust speedup is 42.95x | supported-by: `benchmarks/results/scale_envelope.md` |
| README Benchmarks | Dense recurrence/RQA has an 8*N^2 byte analytical matrix cost | supported-by: `benchmarks/results/scale_envelope.md` |
| README fallback policy; JOSS Software design | Pure-Python paths are parity/portability paths, not large-run performance claims | supported-by: test suite; unsupported performance implication removed |
| README reproduction gallery | The reproduction gallery is the flagship application and stress test, not the package boundary | supported-by: reproduction pipeline; scope claim reframed |
| JOSS reproducibility text | Fixed seeds and selected cache checks guard the fast figure pipeline | supported-by: test suite; universal figure-output claim softened |
| JOSS package scope | External trajectories can be analysed by the diagnostics | supported-by: diagnostics accepting array-like trajectory/time-series inputs |
| Removed/softened claims | Hype or unqualified superlatives about performance/scope | unsupported-removed |
| sec12 intermittency; any manuscript text | The Lorenz laminar-channel fit recovers the type-I tangency slope (~1) | **unsupported-do-not-claim** — see below |


## `lorenz_channel_slope` is not a converged quantity

Type-I intermittency makes the return map tangent to the diagonal in the
laminar channel, so a channel-slope estimate near 1 is the expected physics.
That physics is not in question. What is not supported is the claim that the
Lorenz channel fit in `sec12_intermittency` *measures* it.

Measured on 2026-07-27 at the shipped configuration
(`lorenz_1662_oracle(t_span=(0, 80), dt=0.01)`, `channel_percentile=30`,
~274 extracted maxima):

- Perturbing the initial condition by `1e-12` moves the fitted slope across
  **[0.617, 1.540]**. The committed value 0.98549932 is one draw from that
  distribution, not a reproducible measurement. CI runners observed 2.048.
- Longer integration does not fix it: at `t_span=(0, 1500)` (2017 channel
  points) the spread across the same perturbations is still 0.48.
- The falsifying check fails. If the fit resolved the tangency, narrowing the
  channel would drive the slope to 1. At `t_span=(0, 1500)` the slope runs
  1.223, 0.519, 0.652, 0.725, 0.850, 0.822, 0.819 for `channel_percentile`
  30, 20, 10, 5, 2, 1, 0.5 — non-monotonic, tending to ~0.82, not 1.

The estimator is not simply wrong: applied to `logistic_f3_channel_slope`, a
deterministic 1-D map, it returns **1.00042829**. The Lorenz case is a chaotic
ODE where the extracted maxima do not determine the channel.

**Do not state a Lorenz tangency-slope value in the paper or the README**
without first reworking the fit and demonstrating convergence under refinement.
The test asserts only that the channel is found and the slope is finite and
positive; `tests/test_intermittency_figure.py` records the reasoning.
