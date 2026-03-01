# dynachaos

Python library for dynamical systems, chaos theory, and time series
analysis. Performance-critical algorithms accelerated via Rust (PyO3);
pure-Python fallbacks for all routines.

## Features

**Maps** — logistic, circle, coupled logistic, delayed logistic, coupled
delayed, modulated circle, torus doubling (Map I / Map IV)

**Coupled Map Lattices** — CML with nearest-neighbor diffusive coupling,
globally coupled maps (GCM), pattern dynamics, cluster statistics

**Diagnostics** — Lyapunov exponents (1D + QR spectrum), 0-1 test for chaos,
SALI/GALI alignment indices, permutation entropy + complexity–entropy planes,
recurrence quantification analysis (RQA), Grassberger–Procaccia correlation
dimension

**Rust backends** — recurrence quantification (diagonal/vertical line
extraction), ordinal distribution counting (more coming)

**Visualization** — bifurcation diagrams, cobweb plots, return maps, curated
Swiss-inspired style themes

## Installation

```bash
pip install dynachaos           # pure Python (all features work)
pip install dynachaos[viz]      # adds matplotlib plotting
```

## Quick Start

```python
import numpy as np
from dynachaos.maps import logistic, logistic_derivative
from dynachaos.diagnostics import lyapunov_exponent_1d, permutation_entropy

# Lyapunov exponent of the logistic map at the edge of chaos
f = lambda x: logistic(x, 1.99)
df = lambda x: logistic_derivative(x, 1.99)
lam = lyapunov_exponent_1d(f, df, x0=0.1, n_iter=100_000)
print(f"λ = {lam:.4f}")  # λ ≈ 0.69

# Permutation entropy of a chaotic series
series = np.empty(10_000)
series[0] = 0.1
for i in range(1, len(series)):
    series[i] = logistic(series[i - 1], 1.99)
H = permutation_entropy(series, d=5)
print(f"H_PE = {H:.4f}")  # H_PE ≈ 0.98 (near-random)
```

```python
from dynachaos.diagnostics import recurrence_matrix, rqa

# Recurrence quantification of a periodic orbit
t = np.linspace(0, 40, 600)
traj = np.column_stack([np.sin(t), np.cos(t)])
R, eps = recurrence_matrix(traj, percentile=5)
stats = rqa(R, l_min=2, v_min=2)
print(f"DET = {stats['DET']:.3f}, LAM = {stats['LAM']:.3f}")
```

## Algorithm Reference

| Algorithm | Module | Rust | Reference |
|-----------|--------|------|-----------|
| Lyapunov exponent (1D) | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| Lyapunov spectrum (QR) | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| 0-1 test for chaos | `diagnostics.zero_one_test` | — | Gottwald & Melbourne 2004 |
| SALI / GALI | `diagnostics.sali_gali` | — | Skokos et al. 2007 |
| Permutation entropy | `diagnostics.permutation` | ordinal dist. | Bandt & Pompe 2002 |
| Complexity–entropy plane | `diagnostics.permutation` | ordinal dist. | Rosso et al. 2007 |
| RQA (DET, LAM, ENTR, ...) | `diagnostics.recurrence` | line extraction | Marwan et al. 2007 |
| Correlation dimension | `diagnostics.correlation` | — | Grassberger & Procaccia 1983 |

## Showcase: Kaneko Atlas

A companion manuscript is the first major
application of dynachaos — a tribute to Kunihiko Kaneko's foundational work
on chaos, reproduced at 100–1000× finer resolution using this library.

```bash
dynachaos list                    # see paper sections
dynachaos run sec02_circle_map    # reproduce a figure
dynachaos run all                 # full pipeline
```

## Development

```bash
git clone https://github.com/ricardofrantz/dynachaos.git
cd dynachaos
uv sync
uv run pytest tests/ -q          # pure-Python tests

# With Rust extension:
uv run maturin develop --release
uv run pytest tests/ -q
```

## License

MIT
