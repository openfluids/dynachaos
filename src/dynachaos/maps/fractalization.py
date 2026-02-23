#!/usr/bin/env python3
"""
fractalization: Fractalization of torus near the onset of chaos.

Reproduces Kaneko (1984) "Fractalization of Torus", PTP 71(5), 1112-1115.

As a control parameter approaches the chaotic threshold, a smooth torus
develops wrinkles at ever finer scales and becomes fractal.  The fractal
(correlation) dimension increases from 1 (smooth torus) toward ~1.3-1.5
(fractal torus) at the chaos onset.

Uses the delayed logistic map (Kaneko 1984):
    x_{n+1} = A x_n + (1 - A)(1 - D y_n^2)
    y_{n+1} = x_n

with A = 0.3.  Torus appears at D_c = 1/(1-A) ≈ 1.429.
Fractalization visible for D ∈ [1.90, 1.95].

Figures:
  - Attractor portraits showing progressive fractalization
  - Correlation dimension D_2 vs D (Grassberger-Procaccia)

OUTPUTS: figures/sec07_fractalization/*.npz, *.png
USAGE:   python src/dynachaos/maps/fractalization.py
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec07_fractalization")

FRAC_NPZ = FIG_DIR / "fractal_attractors.npz"
FRAC_PNG = FIG_DIR / "fractal_attractors.png"
DIM_NPZ = FIG_DIR / "correlation_dimension.npz"
DIM_PNG = FIG_DIR / "correlation_dimension.png"
ANIM_NPZ = FIG_DIR / "fractalization_animation.npz"
ANIM_GIF = FIG_DIR / "fractalization_animation.gif"


def _safe_load(path):
    """Load .npz safely (no deserialization of arbitrary objects)."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# Map (reused from delayed_logistic.py)
# ---------------------------------------------------------------------------

def delayed_logistic(state, A, D):
    """One iteration of the delayed logistic map."""
    x, y = state
    x_new = A * x + (1.0 - A) * (1.0 - D * y * y)
    y_new = x
    return np.array([x_new, y_new])


def iterate(A, D, n_transient=20_000, n_record=100_000, x0=None):
    """Iterate and return trajectory points."""
    if x0 is None:
        fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
        x0 = np.array([fp + 0.01, fp - 0.01])

    state = x0.copy()
    for _ in range(n_transient):
        state = delayed_logistic(state, A, D)

    traj = np.empty((n_record, 2))
    for i in range(n_record):
        state = delayed_logistic(state, A, D)
        traj[i] = state
    return traj


# ---------------------------------------------------------------------------
# Correlation dimension (Grassberger-Procaccia)
# ---------------------------------------------------------------------------

def correlation_integral(traj, r_values, max_pairs=500_000):
    """Compute the correlation integral C(r) for an array of r values.

    C(r) = (2 / N(N-1)) * #{pairs with ||x_i - x_j|| < r}

    For efficiency, randomly samples pairs when N is large.
    """
    N = len(traj)
    n_pairs = N * (N - 1) // 2

    if n_pairs > max_pairs:
        # Random sampling of pairs
        rng = np.random.default_rng(42)
        idx_i = rng.integers(0, N, max_pairs)
        idx_j = rng.integers(0, N, max_pairs)
        # Avoid self-pairs
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        diffs = traj[idx_i] - traj[idx_j]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        n_used = len(dists)
    else:
        from scipy.spatial.distance import pdist
        dists = pdist(traj)
        n_used = len(dists)

    C = np.empty(len(r_values))
    for k, r in enumerate(r_values):
        C[k] = np.sum(dists < r) / n_used

    return C


def correlation_dimension(traj, n_r=30, r_range=None, max_pairs=500_000):
    """Estimate the correlation dimension D_2 from trajectory points.

    Fits the scaling C(r) ~ r^{D_2} in the linear region of
    log(C) vs log(r).

    Returns
    -------
    D2 : float
        Estimated correlation dimension.
    r_values : ndarray
        The r values used.
    C_values : ndarray
        Corresponding C(r).
    """
    if r_range is None:
        # Estimate scale from data
        std = np.std(traj, axis=0)
        scale = np.mean(std)
        r_min = scale * 0.005
        r_max = scale * 0.5
        r_range = (r_min, r_max)

    r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_r)
    C_values = correlation_integral(traj, r_values, max_pairs)

    # Fit D_2 in the scaling region (where C > 0 and < 1)
    mask = (C_values > 1e-4) & (C_values < 0.5)
    if np.sum(mask) < 3:
        return np.nan, r_values, C_values

    log_r = np.log(r_values[mask])
    log_C = np.log(C_values[mask])

    # Linear regression
    coeffs = np.polyfit(log_r, log_C, 1)
    D2 = coeffs[0]

    return D2, r_values, C_values


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_attractors():
    """Compute attractors at D values showing fractalization."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 0.3
    # D values: smooth torus -> wrinkled -> fractal -> chaos
    D_values = [1.75, 1.86, 1.90, 1.92, 1.94, 1.945]
    labels = ["smooth torus", "torus",
              "wrinkled torus", "fractal torus",
              "onset of chaos", "chaos"]

    results = {}
    for D, label in zip(D_values, labels):
        print(f"  D={D} ({label})")
        traj = iterate(A, D, n_transient=30_000, n_record=200_000)
        results[f"D_{D}_traj"] = traj

    results["D_values"] = np.array(D_values)
    results["A"] = np.array([A])
    np.savez_compressed(FRAC_NPZ, **results)
    print(f"Saved {FRAC_NPZ}")


def compute_dimensions():
    """Compute correlation dimension D_2 vs D."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 0.3
    n_params = 100
    D_values = np.linspace(1.70, 2.00, n_params)
    D2_values = np.empty(n_params)

    for i, D in enumerate(D_values):
        traj = iterate(A, D, n_transient=20_000, n_record=50_000)
        D2, _, _ = correlation_dimension(traj, n_r=25, max_pairs=300_000)
        D2_values[i] = D2

        if (i + 1) % 20 == 0:
            print(f"  Dimension: {i + 1}/{n_params}")
            np.savez_compressed(DIM_NPZ, D=D_values[:i+1],
                                D2=D2_values[:i+1], A=np.array([A]))

    np.savez_compressed(DIM_NPZ, D=D_values, D2=D2_values, A=np.array([A]))
    print(f"Saved {DIM_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attractors(data):
    """Plot attractor portraits showing fractalization."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, DOUBLE_COL, setup
    setup()

    D_values = data["D_values"]
    labels = ["smooth torus", "torus",
              "wrinkled torus", "fractal torus",
              "onset of chaos", "chaos"]

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 5.0))
    axes_flat = axes.flatten()

    for idx, D in enumerate(D_values):
        ax = axes_flat[idx]
        traj = data[f"D_{D}_traj"]
        ax.scatter(traj[:, 0], traj[:, 1], s=0.01, c=COLORS["black"],
                   alpha=0.15, rasterized=True)
        ax.set_title(f"$D = {D}$\n{labels[idx]}", fontsize=8)
        ax.set_xlabel("$x$", fontsize=7)
        ax.set_ylabel("$y$", fontsize=7)
        ax.tick_params(labelsize=6)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    fig.suptitle(r"Fractalization of torus, $\alpha = 0.3$", fontsize=10, y=1.02)
    fig.savefig(FRAC_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FRAC_PNG}")


def plot_dimension(data):
    """Plot correlation dimension D_2 vs D."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, DOUBLE_COL, setup
    setup()

    D = data["D"]
    D2 = data["D2"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
    ax.plot(D, D2, color=COLORS["black"], linestyle="-", lw=0.8)
    ax.axhline(1.0, color=COLORS["blue"], lw=0.5, ls="--", alpha=0.6,
               label="$D_2 = 1$ (smooth curve)")
    ax.axhline(2.0, color=COLORS["red"], lw=0.5, ls="--", alpha=0.6,
               label="$D_2 = 2$ (area-filling)")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"Correlation dimension $D_2$")
    ax.set_title(r"Delayed logistic map, $\alpha = 0.3$", fontsize=10)
    ax.set_ylim(0.5, 2.5)
    ax.legend(fontsize=7)

    fig.savefig(DIM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {DIM_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def compute_animation_data():
    """Sweep D from 1.75 to 1.96 for fractalization animation."""
    from dynachaos.utils.animation import compute_animation_sweep

    A = 0.3
    D_sweep = np.linspace(1.75, 1.96, 200)

    def iterate_fn(D):
        return iterate(A, D, n_transient=30_000, n_record=5_000)

    compute_animation_sweep(iterate_fn, D_sweep, ANIM_NPZ, n_plot=5_000)


def make_animation_gif(data):
    """Create GIF of fractalization progression."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"], data["all_x"], data["all_y"], ANIM_GIF,
        title_template=r"Fractalization of torus, $\alpha = 0.3$, $D = {param_value}$",
        param_name="D", param_fmt=".4f",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        frac_data = _safe_load(FRAC_NPZ)
        print(f"Loaded {FRAC_NPZ}")
    except FileNotFoundError:
        print("Computing fractal attractors...")
        compute_attractors()
        frac_data = _safe_load(FRAC_NPZ)
    plot_attractors(frac_data)

    try:
        dim_data = _safe_load(DIM_NPZ)
        print(f"Loaded {DIM_NPZ}")
    except FileNotFoundError:
        print("Computing correlation dimensions...")
        compute_dimensions()
        dim_data = _safe_load(DIM_NPZ)
    plot_dimension(dim_data)

    # Animation
    try:
        anim_data = _safe_load(ANIM_NPZ)
        print(f"Loaded {ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing fractalization animation data...")
        compute_animation_data()
        anim_data = _safe_load(ANIM_NPZ)
    make_animation_gif(anim_data)


if __name__ == "__main__":
    main()
