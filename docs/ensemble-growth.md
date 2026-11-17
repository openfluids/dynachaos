# Finite-time growth of ensemble spread

`ensemble_growth` measures the RMS norm of trajectories or error curves in
linear units. For runs `x_i(t)`, it returns

$$S(t) = \sqrt{\frac{1}{N}\sum_i \|x_i(t)\|_2^2}.$$

This is scale about zero. It includes the mean and does not subtract it.
To measure separation from a reference trajectory, first subtract that
reference. Scalar error curves can be passed directly, without taking logs.

## Fit a stated window

```python
import numpy as np
from dynachaos.diagnostics import ensemble_growth

times = np.arange(21, dtype=float)
initial = np.linspace(0.001, 0.002, 10)
trajectories = initial[:, None] * np.exp(0.25 * times)
result = ensemble_growth(
    trajectories,
    times=times,
    window=(0, 21),
    initial_floor=1e-6,
    bootstrap_seed=42,
    permutation_seed=43,
)
print(result.g)             # 0.25, up to floating-point roundoff
print(result.g_interval)    # Percentile interval from resampling whole runs
print(result.G_max)         # exp(5)
print(result.null_exceedance)
```

Input shape is `(runs, steps, state)` or `(runs, steps)` for scalar states.
Values must be finite and real, every axis must be nonempty, and ensembles accept
float16, float32, float64, and integers exactly representable in float64. `times`
defaults to integer sample indices and must otherwise be strictly increasing.
The window uses sample indices, with an inclusive start and exclusive stop;
it defaults to the whole curve. At least five points are required. The rate
`g` is the ordinary least-squares slope of the natural logarithm of `S(t)`
against time, so it has inverse time units.

The bootstrap samples entire runs with replacement. `bootstrap_rates` holds
every fitted rate; `g_interval` holds its percentile interval, at `confidence`
(default 0.95). Spread must be positive throughout the fitted window, including
every bootstrap draw. A zero raises an error rather than introducing a log
floor or silently omitting a draw. Small ensembles can give poor coverage.

## Peak amplification and the null

`G_max = max(S) / S(0)` uses the whole supplied curve, regardless of the fit
window. `initial_floor` is required and must be positive. A baseline below it
raises an error; the denominator is never clamped. Slice the trajectories and
times explicitly to choose another baseline. The floor used is returned.

The null independently permutes the per-turn increments of each run and
reconstructs trajectories from the same initial states. Vector components
stay together. Each draw preserves a run's increments and endpoint while
removing their order. It assumes exchangeable additive increments with no
feedback. Unequal time gaps, drift, changing noise strength, bounded scores,
and stopping rules can invalidate that assumption. The null is conditional
on the observed increments, rather than a fitted model of all possible runs.
Because endpoints are fixed, monotone positive scalar growth gives a null
exceedance of one, even for exponential growth. This null assesses the ordering
of increments and transient peaks, rather than the magnitude of endpoint growth.

`null_samples` contains the peak ratios from these draws. `null_exceedance`
is the fraction at or above the observed ratio, including ties, without a
pseudocount. No bootstrap interval is given for the peak. Both random steps
require separate explicit integer seeds; identical inputs and seeds reproduce
the samples bit for bit. Draw counts default to 999 each.

Independent zero-mean noise added each turn with no feedback gives spread
that grows like the square root of the turn count. The null preserves this
accumulation; permuting state levels instead would remove it. Positive `g`
or a peak above one alone cannot distinguish diffusion from feedback.

The output is a finite-time growth rate of ensemble spread. It is not a
Lyapunov exponent for stochastic, discrete data, and does not establish that
a system is chaotic. Even a small null exceedance only describes disagreement
with the stated exchangeability assumption.
