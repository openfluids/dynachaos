# dynachaos

![dynachaos banner](assets/readme-banner-v1.png)

[![CI](https://github.com/ricardofrantz/dynachaos/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ricardofrantz/dynachaos/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Reusable dynamical-systems analysis for simulated or measured time signals, with Rust-accelerated kernels where they are tested.**

Status: this is a private research and development repository. Public release,
package publication, and final paper citation details are still future work.

> *"Chaos: When the present determines the future, but the approximate present*
> *does not approximately determine the future."*
> — Edward Lorenz

![Bifurcation diagram of the logistic map](assets/bifurcation.png)

## Why dynachaos?

dynachaos is a reusable Python/Rust package for inspecting simulated or
measured dynamical-systems time signals. It collects maps, coupled-map
lattices, recurrence analysis, entropy diagnostics, Grassberger-Procaccia
correlation dimension, multifractal spectra, and reproducible analysis
pipelines in one codebase. Performance-sensitive kernels use Rust backends
where they have parity tests, while pure-Python fallbacks support portability
and reference checks.

## Features

**Maps** — logistic, circle, coupled logistic, delayed logistic, coupled
delayed, modulated circle, torus doubling (Map I / Map IV)

**Coupled Map Lattices** — CML with nearest-neighbor diffusive coupling,
globally coupled maps (GCM), pattern dynamics, cluster statistics

**Diagnostics** — Lyapunov exponents (1D + QR spectrum + flow systems),
0-1 test for chaos, SALI/GALI alignment indices, permutation entropy +
complexity-entropy planes, sample/approximate/fuzzy/multiscale entropy,
recurrence quantification analysis (RQA), Grassberger-Procaccia
correlation dimension, multifractal spectra ($D_q$, $f(\alpha)$),
AMI + Cao + FNN embedding

**Rust backends** — correlation integral (Grassberger-Procaccia), fuzzy
entropy sum, recurrence line extraction, ordinal distribution counting,
AMI histograms, Cao dimension selection, multifractal moments

**Visualization** — bifurcation diagrams, cobweb plots, return maps,
curated Swiss-inspired style themes

## Private Development Setup

```bash
git clone https://github.com/ricardofrantz/dynachaos.git
cd dynachaos
uv sync
uv run --extra viz pytest tests/ -q
```

To exercise the installed Rust extension locally:

```bash
uv run maturin develop --release
uv run --extra viz pytest tests/ -q
```

To verify the pure-Python fallback path:

```bash
DYNACHAOS_NO_RUST=1 uv run --extra viz pytest tests/ -q
```

## Benchmarks

The reproducible scale-envelope benchmark for Rust Grassberger-Procaccia parity
and dense recurrence/RQA memory limits lives in `benchmarks/scale_envelope.py`.
Run CI mode with `uv run python benchmarks/scale_envelope.py benchmarks/scale_envelope.jsonc`
and inspect `benchmarks/results/scale_envelope.{json,md}`. The checked artifact
reports a 42.95x CI-mode Rust Grassberger-Procaccia speedup at N=1000 for the
largest common logistic case, and a dense-RQA predicted distance-matrix envelope
of 8*N^2 bytes (impracticality threshold N≈23170 at 4 GiB). The measured Rust
acceleration roadmap and local hotspot profiler are documented in
`docs/rust-acceleration-roadmap.md` and `benchmarks/rust_hotspot_profile.py`.

## Config-driven signal analysis workflow

Run a scalar/reduced time-series workflow with a JSONC config; all tuning lives in the config, not CLI flags:

```bash
uv run dynachaos analyze tests/data/workflow_fixture.jsonc
```

Config schema summary:
- `input`: either `{ "path": "signal.npy" }` / `{ "path": "signal.npz", "npz_key": "x" }` for a 1D finite scalar/reduced signal, or `{ "generated": { "name": "logistic", "n": 1000, "seed": 0 } }` for self-contained runs.
- `output.dir`: stable output directory, resolved relative to the config file.
- `diagnostics`: list of `{ "name": ... }` entries. Supported names are `permutation_entropy`, `correlation_dimension`, `rqa_streaming`, and `rqa_dense`.
- `scale_limits`: optional dense-RQA guard; `dense_rqa_max_bytes` defaults to 4 GiB using the `8*N^2` distance-matrix envelope, and `allow_dense_rqa_beyond_envelope: true` is required to override it.

Output layout is stable and referenceable by path: `results.json` (machine-readable diagnostic values), `metadata.json` (N, shape, wall time in seconds, peak RSS in MB, and per-diagnostic `ReliabilityRecord` metadata), and `summary.md` (human-readable report with relative artifact names).

For long-signal local RQA, avoid dense recurrence matrices and set an explicit threshold in config:

```jsonc
{
  "input": {"path": "long_signal.npy"},
  "output": {"dir": "results/long_signal_rqa"},
  "diagnostics": [
    {"name": "rqa_streaming", "embedding": {"d": 3, "tau": 2}, "eps": 0.08, "l_min": 2, "v_min": 2}
  ]
}
```

Run locally with `uv run dynachaos analyze long_signal_rqa.jsonc`; CI fixtures stay tiny and hermetic.

### Reliability metadata

Diagnostics can opt in to compact JSON-safe reliability metadata without changing default return values.
```python
D2, r, C, slopes, mask, meta = correlation_dimension(traj, return_metadata=True)
print(meta.to_json())  # backend, parameters, data shape, warnings, unresolved verdicts
```

## Quick Start

```python
import numpy as np
from dynachaos.maps import logistic, logistic_derivative
from dynachaos.diagnostics import lyapunov_exponent_1d, permutation_entropy

# Lyapunov exponent of the logistic map at the edge of chaos
f  = lambda x: logistic(x, 1.99)
df = lambda x: logistic_derivative(x, 1.99)
lam = lyapunov_exponent_1d(f, df, x0=0.1, n_iter=100_000)
print(f"lambda = {lam:.4f}")   # ~ 0.65

# Permutation entropy of a chaotic series
series = np.empty(10_000)
series[0] = 0.1
for i in range(1, len(series)):
    series[i] = logistic(series[i - 1], 1.99)
H = permutation_entropy(series, d=5)
print(f"H_PE = {H:.4f}")       # ~ 0.68
```

```python
import numpy as np
from dynachaos.maps import logistic
from dynachaos.diagnostics import correlation_dimension, sample_entropy
from dynachaos.diagnostics.recurrence import embed_time_delay

series = np.empty(10_000)
series[0] = 0.1
for i in range(1, len(series)):
    series[i] = logistic(series[i - 1], 1.99)

# Correlation dimension of the logistic map attractor
traj = embed_time_delay(series, d=3, tau=1)
D2, r, C, slopes, mask = correlation_dimension(traj, theiler_window=1)
print(f"D2 = {D2:.3f}")        # ~ 0.97

# Sample entropy (regularity measure)
se = sample_entropy(series, m=2)
print(f"SampEn = {se:.4f}")
```

```bash
# Compare Grassberger-Procaccia vs multifractal D2
uv run --extra viz python examples/benchmark_gp_vs_multifractal.py
```

## Rust-Accelerated Backends

Performance-critical algorithms are implemented as Rust kernels. The Rust
extension is required by default: `import dynachaos` fails loudly if it has not
been built.

Build the extension in editable mode:

```bash
uv run maturin develop --release
```

### Pure-Python fallback policy

The Rust kernels are the intended path for production-sized all-pairs and
large-N diagnostics. Pure-Python paths are an explicit opt-in for parity testing
and portability, not an automatic silent fallback. Set `DYNACHAOS_NO_RUST=1`
when you need to exercise them; the test suite checks parity on representative
small workloads. Some fallback implementations remain exact and quadratic by
design, so large pure-Python runs should be treated as diagnostic or development
runs unless a future release explicitly makes large fallback performance a
target.

### Correlation integral (Grassberger-Procaccia)

The all-pairs kernel evaluates all N(N−1)/2 pairs with Theiler-window
exclusion and multi-radius binning in a single pass.
The implementation is designed for correctness and low memory use,
offering several advantages over common baseline scripts (e.g., [notsebastiano/GP_algorithm](https://github.com/notsebastiano/GP_algorithm/blob/master/GP_algorithm.py)):

- **Algorithmic Correctness** — Supports the **Theiler window** ($|i-j| > w$) to
  exclude temporally correlated pairs (Theiler 1986) and uses proper
  normalization ($C(r) \le 1$).
- **Scaling Region Detection** — Employs a **stable plateau search** on local
  slopes instead of simple heuristics, making it robust to noise and saturation.
- **Memory Efficiency** — Uses **O(1) auxiliary memory** per pair (streaming)
  instead of an $O(N^2)$ distance matrix.

The Rust kernel uses:

- **Raw slice indexing** — C-contiguous slice access, avoiding ndarray's
  per-index `Index` overhead while staying in safe Rust
- **Prefix-sum binning** — one write per pair instead of up to 50;
  converted to cumulative counts with a single O(n_r) pass after the loop
- **Squared-distance comparison** — pre-squared thresholds eliminate
  `sqrt()` in Euclidean mode (saves 10–20 cycles per pair)
- **Branch-separated loops** — Chebyshev and Euclidean paths are fully
  separated, enabling independent auto-vectorization
- **Rayon parallelism** — outer loop distributed across all cores via
  Rayon; GIL released with `py.detach()` before the parallel region
- **Native compiler optimization** — release builds can use the local CPU
  target configured under `.cargo/`.

Benchmark numbers should be regenerated on the release target hardware before
being used in public documentation.

### Fuzzy entropy sum

Computes Σ exp(−(d/r)ⁿ) over all upper-triangle pairs on mean-centered
templates, using the same Rayon parallel fold + reduce pattern.

## Algorithm Reference

| Algorithm | Module | Rust kernel | Reference |
|-----------|--------|-------------|-----------|
| Lyapunov exponent (1D) | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| Lyapunov spectrum (QR) | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| Flow Lyapunov spectrum | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| 0-1 test for chaos | `diagnostics.zero_one_test` | — | Gottwald & Melbourne 2004 |
| SALI / GALI | `diagnostics.sali_gali` | — | Skokos et al. 2007 |
| Permutation entropy | `diagnostics.permutation` | ordinal dist. | Bandt & Pompe 2002 |
| Complexity-entropy plane | `diagnostics.permutation` | ordinal dist. | Rosso et al. 2007 |
| Sample entropy | `diagnostics.entropy` | correlation counts | Richman & Moorman 2000 |
| Approximate entropy | `diagnostics.entropy` | — | Pincus 1991 |
| Fuzzy entropy | `diagnostics.entropy` | fuzzy sum | Chen et al. 2007 |
| Multiscale entropy | `diagnostics.entropy` | correlation counts | Costa et al. 2002 |
| RQA (DET, LAM, ENTR, …) | `diagnostics.recurrence` | line extraction | Marwan et al. 2007 |
| Correlation dimension | `diagnostics.correlation` | all-pairs kernel | Grassberger & Procaccia 1983 |
| Multifractal spectrum ($D_q$, $f(\alpha)$) | `diagnostics.multifractal` | multifractal moments | Mukherjee et al. 2024 |
| AMI (embedding) | `diagnostics.embedding` | histogram | Fraser & Swinney 1986 |
| Cao's method | `diagnostics.embedding` | dimension selector only | Cao 1997 |
| False nearest neighbors | `diagnostics.embedding` | — | Kennel et al. 1992 |

`diagnostics.recurrence` keeps `recurrence_matrix()` for callers that need the
binary matrix. For large trajectories where only scalar RQA measures are needed,
use `rqa_from_trajectory()` to avoid materializing the dense recurrence matrix.
When starting from an existing recurrence matrix, compute public RQA metrics
through `rqa()`, which validates that the matrix is non-empty, square, and
symmetric. The direct line extractors, including the Rust-accelerated helpers,
are lower-level square-matrix scanners and do not replace that public RQA
validation boundary.

## Flagship application: Kaneko Atlas

A companion manuscript and figure
pipeline that serves as dynachaos's flagship application and stress test. It
revisits Kunihiko Kaneko's foundational work on chaos with the same reusable
Python/Rust diagnostics exposed by the package.

```bash
dynachaos list                    # list paper sections
dynachaos run sec02_circle_map    # reproduce a figure
dynachaos run all                 # full pipeline
```

To rebuild the tracked manuscript PDF:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Development

```bash
uv sync
uv run --extra viz pytest tests/ -q
uv run --extra viz ruff check src/ tests/
uv run --extra viz ruff format src/ tests/ --check

# With Rust extension:
uv run maturin develop --release
uv run --extra viz pytest tests/ -q
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines. All participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

Citation metadata is provisional while the repository and manuscript remain
private. Do not treat the companion paper entry as a published article until
the public release phase is explicitly completed.

```bibtex
@software{dynachaos2026,
  author  = {Frantz, Ricardo},
  title   = {dynachaos: Dynamical-systems analysis for time signals with Rust-accelerated kernels},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/ricardofrantz/dynachaos},
  license = {MIT}
}

@article{removed,
  author  = {Frantz, Ricardo},
  
             },
  
  year    = {2026}
}
```

## License

MIT
