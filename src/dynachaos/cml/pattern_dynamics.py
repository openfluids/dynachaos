#!/usr/bin/env python3
"""
pattern_dynamics: CML pattern dynamics and global phase diagram.

Reproduces Kaneko (1989) "Pattern dynamics in spatiotemporal chaos", Physica D
34, 1-41.

CML model (Eq. 1):
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + eps/2 [f(x_n(i+1)) + f(x_n(i-1))]

with f(x) = 1 - a x^2 (logistic map) and periodic boundary conditions.

Phase diagram: a in [1.5, 2.0], eps in [0, 0.4]

OUTPUTS: figures/sec09_pattern/*.npz, *.png
"""

from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps.primitives import logistic
import numpy as np

from dynachaos.cml.primitives import cml_step_logistic as cml_step

FIG_DIR = section_dir("sec09_pattern")

PHASE_NPZ = FIG_DIR / "phase_diagram.npz"
PHASE_PNG = FIG_DIR / "phase_diagram.png"
SPACE_NPZ = FIG_DIR / "space_amplitude.npz"
SPACE_PNG = FIG_DIR / "space_amplitude.png"


# ---------------------------------------------------------------------------
# Phase diagram computation
# ---------------------------------------------------------------------------

def _cml_step_batch(x, a_col, eps):
    """Batched CML step: x is (n_a, N), a_col is (n_a, 1)."""
    fx = logistic(x, a_col)
    fx_left = np.roll(fx, -1, axis=1)
    fx_right = np.roll(fx, 1, axis=1)
    return (1.0 - eps) * fx + eps / 2.0 * (fx_left + fx_right)


def compute_phase_diagram():
    """Compute phase diagram by measuring temporal variance.

    Vectorized: for each eps, all a values run in parallel as a
    (n_a, N) array.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_a = 200
    n_eps = 160
    a_values = np.linspace(1.5, 2.0, n_a)
    eps_values = np.linspace(0.0, 0.4, n_eps)
    N = 100
    n_transient = 5000
    n_sample = 2000

    # a values as column vector for broadcasting with (n_a, N) arrays
    a_col = a_values[:, np.newaxis]  # shape (n_a, 1)

    lam_grid = np.empty((n_eps, n_a))
    rng = np.random.default_rng(42)

    for j, eps in enumerate(eps_values):
        # Each row is one CML instance at a different a
        x = rng.uniform(-1, 1, (n_a, N))

        for _ in range(n_transient):
            x = _cml_step_batch(x, a_col, eps)

        # Accumulate variance of the central site
        sum_v = np.zeros(n_a)
        sum_v2 = np.zeros(n_a)
        for _ in range(n_sample):
            x = _cml_step_batch(x, a_col, eps)
            mid = x[:, N // 2]
            sum_v += mid
            sum_v2 += mid * mid

        mean_v = sum_v / n_sample
        lam_grid[j] = sum_v2 / n_sample - mean_v * mean_v

        if (j + 1) % 40 == 0:
            print(f"  Phase diagram: {j + 1}/{n_eps}")
            np.savez_compressed(PHASE_NPZ, a=a_values, eps=eps_values,
                                lam=lam_grid)

    np.savez_compressed(PHASE_NPZ, a=a_values, eps=eps_values,
                        lam=lam_grid)
    print(f"Saved {PHASE_NPZ}")


# ---------------------------------------------------------------------------
# Space-amplitude plots
# ---------------------------------------------------------------------------

def compute_space_amplitude():
    """Compute space-amplitude snapshots for representative phases."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    N = 100
    params = [
        (1.44, 0.1, "Frozen random"),
        (1.64, 0.1, "Pattern selection"),
        (1.72, 0.1, "Defect turbulence"),
        (1.80, 0.1, "Brownian defects"),
        (1.90, 0.1, "Fully dev. turbulence"),
    ]

    results = {}
    rng = np.random.default_rng(42)

    for a, eps, label in params:
        print(f"  a={a}, eps={eps} ({label})")
        x = rng.uniform(-1, 1, N)
        for _ in range(10_000):
            x = cml_step(x, a, eps)

        snapshots = np.empty((50, N))
        for t in range(50):
            for _ in range(20):
                x = cml_step(x, a, eps)
            snapshots[t] = x.copy()

        results[f"a_{a}_eps_{eps}_snap"] = snapshots
        results[f"a_{a}_eps_{eps}_label"] = np.array([label])

    results["params"] = np.array([(a, eps) for a, eps, _ in params])
    np.savez_compressed(SPACE_NPZ, **results)
    print(f"Saved {SPACE_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_diagram(data):
    """Plot the global phase diagram."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        COLORS,
        apply_axes_polish,
        figure_spec,
        setup,
    )
    setup()

    a = data["a"]
    eps = data["eps"]
    lam = data["lam"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    im = ax.pcolormesh(a, eps, np.log10(lam + 1e-10), cmap=CMAP_SEQUENTIAL,
                       rasterized=True)
    ax.set_xlabel(r"Nonlinearity $a$")
    ax.set_ylabel(r"Coupling $\varepsilon$")
    ax.set_title("CML phase diagram (log temporal variance)", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(\mathrm{Var})$")
    cb.ax.tick_params(labelsize=spec.tick_size)
    apply_axes_polish(ax, kind="double", title_loc="left")

    ax.axvline(1.401, color=COLORS["offwhite"], lw=0.5, ls="--", alpha=0.7)
    ax.text(
        1.42,
        0.35,
        "period-2\nband",
        fontsize=spec.tick_size * 0.8,
        color=COLORS["offwhite"],
    )

    fig.savefig(PHASE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PHASE_PNG}")


def plot_space_amplitude(data):
    """Plot space-amplitude snapshots."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    params = data["params"]
    n_panels = len(params)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(1, n_panels, figsize=(spec.figsize[0], spec.figsize[1] * 0.54))

    for idx, (a, eps) in enumerate(params):
        ax = axes[idx]
        key = f"a_{a}_eps_{eps}_snap"
        label_key = f"a_{a}_eps_{eps}_label"
        if key in data:
            snapshots = data[key]
            N = snapshots.shape[1]
            sites = np.arange(N)
            for t in range(snapshots.shape[0]):
                ax.plot(
                    sites,
                    snapshots[t],
                    color=COLORS["black"],
                    linestyle="-",
                    lw=0.1,
                    alpha=0.3,
                )
        label = str(data[label_key][0]) if label_key in data else ""
        ax.set_title(f"$a={a}$\n{label}", loc="left")
        ax.set_xlabel("$i$")
        if idx == 0:
            ax.set_ylabel("$x(i)$")
        apply_axes_polish(ax, kind="grid", title_loc="left")

    fig.suptitle(
        r"Space-amplitude plots, $\varepsilon = 0.1$, $N=100$",
        fontsize=spec.title_size,
        y=1.05,
    )
    fig.savefig(SPACE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SPACE_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        phase_data = safe_load(PHASE_NPZ)
        print(f"Loaded {PHASE_NPZ}")
    except FileNotFoundError:
        print("Computing phase diagram...")
        compute_phase_diagram()
        phase_data = safe_load(PHASE_NPZ)
    plot_phase_diagram(phase_data)

    try:
        space_data = safe_load(SPACE_NPZ)
        print(f"Loaded {SPACE_NPZ}")
    except FileNotFoundError:
        print("Computing space-amplitude plots...")
        compute_space_amplitude()
        space_data = safe_load(SPACE_NPZ)
    plot_space_amplitude(space_data)


if __name__ == "__main__":
    main()
