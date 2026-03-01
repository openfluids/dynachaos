#!/usr/bin/env python3
"""
globally_coupled: Globally coupled map violating the law of large numbers.

Reproduces Kaneko (1990) PRL 65(12), 1391-1394.

GCM model (Eq. 1):
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + eps/N sum_j f(x_n(j))

with f(x) = 1 - a x^2 (logistic map).

Key result: MSD of mean field h_n stops decreasing with N.

OUTPUTS: figures/sec10_gcm/*.npz, *.png
"""

from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps.primitives import logistic
import numpy as np

FIG_DIR = section_dir("sec10_gcm")

GCM_NPZ = FIG_DIR / "gcm_results.npz"
GCM_PNG = FIG_DIR / "gcm_msd.png"
DIST_PNG = FIG_DIR / "gcm_distribution.png"


# ---------------------------------------------------------------------------
# GCM model
# ---------------------------------------------------------------------------

def gcm_step(x, a, eps):
    """One GCM step."""
    fx = logistic(x, a)
    mean_field = np.mean(fx)
    return (1.0 - eps) * fx + eps * mean_field


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute():
    """Compute MSD of mean field for various N."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    a = 1.99
    eps = 0.1
    N_values = [100, 400, 1000, 5000, 20_000]
    n_transient = 10_000
    n_sample = 100_000

    results = {}
    rng = np.random.default_rng(42)

    for N in N_values:
        print(f"  N={N}")
        x = rng.uniform(-1, 1, N)

        for _ in range(n_transient):
            x = gcm_step(x, a, eps)

        h_series = np.empty(n_sample)
        for t in range(n_sample):
            x = gcm_step(x, a, eps)
            fx = logistic(x, a)
            h_series[t] = np.mean(fx)

        results[f"N_{N}_h"] = h_series
        results[f"N_{N}_mean"] = np.array([np.mean(h_series)])
        results[f"N_{N}_msd"] = np.array([np.var(h_series)])

    results["N_values"] = np.array(N_values)
    results["a"] = np.array([a])
    results["eps"] = np.array([eps])

    # MSD vs N for multiple a values
    a_for_msd = [1.80, 1.85, 1.92, 1.95, 1.99]
    N_grid = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
    msd_grid = np.empty((len(a_for_msd), len(N_grid)))

    for ia, a_val in enumerate(a_for_msd):
        for jn, N_val in enumerate(N_grid):
            x = rng.uniform(-1, 1, N_val)
            for _ in range(5000):
                x = gcm_step(x, a_val, eps)
            h_vals = np.empty(50_000)
            for t in range(50_000):
                x = gcm_step(x, a_val, eps)
                fx = logistic(x, a_val)
                h_vals[t] = np.mean(fx)
            msd_grid[ia, jn] = np.var(h_vals)
        print(f"  MSD: a={a_val} done")

    results["a_for_msd"] = np.array(a_for_msd)
    results["N_grid"] = np.array(N_grid)
    results["msd_grid"] = msd_grid

    np.savez_compressed(GCM_NPZ, **results)
    print(f"Saved {GCM_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_msd(data):
    """Plot MSD vs N (main result of the paper)."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        series_style,
        setup,
    )
    setup()

    N_grid = data["N_grid"]
    msd_grid = data["msd_grid"]
    a_vals = data["a_for_msd"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    for ia, a_val in enumerate(a_vals):
        sty = series_style(ia)
        ax.loglog(
            N_grid,
            msd_grid[ia],
            label=f"$a = {a_val}$",
            color=sty["color"],
            marker=sty["marker"],
            markersize=3.0,
            markerfacecolor=COLORS["offwhite"],
            markeredgewidth=0.75,
            linewidth=1.1,
        )

    N_ref = np.array([100, 20_000])
    ax.loglog(
        N_ref,
        0.05 / N_ref,
        color=COLORS["black"],
        linestyle="--",
        lw=0.8,
        label=r"$\propto 1/N$",
    )

    ax.set_xlabel(r"System size $N$")
    ax.set_ylabel(r"MSD $\langle (\delta h)^2 \rangle$")
    ax.set_title(rf"Globally coupled map, $\varepsilon = {data['eps'][0]}$", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left")
    finalize_legend(ax, kind="double", loc="lower left", ncol=2, columnspacing=1.1)

    fig.savefig(GCM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {GCM_PNG}")


def plot_distribution(data):
    """Plot distribution P(h) for different N."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        COLOR_CYCLE,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
    )
    setup()

    N_values = data["N_values"]

    spec = figure_spec("single")
    fig, ax = plt.subplots(figsize=spec.figsize)
    for idx, N in enumerate(N_values):
        h = data[f"N_{N}_h"]
        ax.hist(h, bins=100, density=True, alpha=0.9,
                color=COLOR_CYCLE[idx], label=f"$N = {N}$",
                histtype="step", lw=1.0)

    ax.set_xlabel(r"Mean field $h$")
    ax.set_ylabel(r"$P(h)$")
    ax.set_title(rf"$a = {data['a'][0]}$, $\varepsilon = {data['eps'][0]}$", loc="left")
    apply_axes_polish(ax, kind="single", title_loc="left")
    finalize_legend(ax, kind="single", loc="upper left")

    fig.savefig(DIST_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {DIST_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(GCM_NPZ)
        print(f"Loaded {GCM_NPZ}")
    except FileNotFoundError:
        print("Computing GCM results...")
        compute()
        data = safe_load(GCM_NPZ)
    plot_msd(data)
    plot_distribution(data)


if __name__ == "__main__":
    main()
