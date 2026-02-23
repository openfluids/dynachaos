#!/usr/bin/env python3
"""
circle_map: Devil's staircase and Lyapunov exponents for the Kaneko circle map.

Reproduces Kaneko (1982) "On the period-adding phenomena at the frequency
locking in a one-dimensional mapping".

Map:  θ_{n+1} = θ_n + D + A sin(2πθ_n)  (mod 1)

With D = 0.25 fixed and A as the bifurcation parameter.

OUTPUTS: figures/sec02_circle_map/devils_staircase.npz,
         figures/sec02_circle_map/devils_staircase.png
USAGE:   python src/dynachaos/maps/circle_map.py                        # .npz exists → plot only
         rm figures/sec02_circle_map/devils_staircase.npz && python src/dynachaos/maps/circle_map.py  # recompute
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec02_circle_map")
OUTPUT_NPZ = FIG_DIR / "devils_staircase.npz"
OUTPUT_PNG = FIG_DIR / "devils_staircase.png"
ZOOM_NPZ = FIG_DIR / "staircase_zoom.npz"
ZOOM_PNG = FIG_DIR / "staircase_zoom.png"


def _safe_load(path):
    """Load .npz with pickle disabled for safety."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------

def circle_map(theta, A, D=0.25):
    """One iteration of the Kaneko circle map."""
    return (theta + D + A * np.sin(2 * np.pi * theta)) % 1.0


def circle_map_derivative(theta, A, D=0.25):
    """Derivative dF/dθ of the circle map."""
    return 1.0 + 2 * np.pi * A * np.cos(2 * np.pi * theta)


# ---------------------------------------------------------------------------
# Rotation number computation
# ---------------------------------------------------------------------------

def rotation_number(A, D=0.25, n_transient=5000, n_iter=50_000, theta0=0.1):
    """Compute the rotation number for given parameters.

    The rotation number is the average advance per iteration (without mod 1).
    """
    theta = theta0
    # Transient — iterate without mod to track total advance, then reset
    theta_unwrapped = theta
    for _ in range(n_transient):
        theta_unwrapped += D + A * np.sin(2 * np.pi * theta_unwrapped)
        # Keep the "wrapped" version for the sin evaluation
        # but track the unwrapped total

    # Now accumulate for the rotation number
    theta_start = theta_unwrapped
    for _ in range(n_iter):
        theta_unwrapped += D + A * np.sin(2 * np.pi * theta_unwrapped)

    return (theta_unwrapped - theta_start) / n_iter


def lyapunov_exponent(A, D=0.25, n_transient=5000, n_iter=50_000, theta0=0.1):
    """Compute Lyapunov exponent of the circle map at given A, D."""
    theta = theta0
    for _ in range(n_transient):
        theta = circle_map(theta, A, D)

    log_sum = 0.0
    for _ in range(n_iter):
        deriv = abs(circle_map_derivative(theta, A, D))
        if deriv > 0:
            log_sum += np.log(deriv)
        else:
            log_sum += -100.0
        theta = circle_map(theta, A, D)

    return log_sum / n_iter


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute():
    """Sweep A and compute rotation numbers + Lyapunov exponents.

    Vectorized: all parameter values are iterated simultaneously as a
    single NumPy array, giving ~100x speedup over scalar loops.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_params = 100_000
    n_transient = 2000
    n_iter = 20_000
    D = 0.25

    A_values = np.linspace(0.0, 0.25, n_params)
    TWO_PI = 2.0 * np.pi

    # --- Vectorised iteration: all 100k thetas in parallel ---
    theta = np.full(n_params, 0.1)

    # Transient (unwrapped)
    for _ in range(n_transient):
        theta += D + A_values * np.sin(TWO_PI * theta)
    print("  Transient done")

    # Record start for rotation number
    theta_start = theta.copy()

    # Accumulate Lyapunov and iterate for rotation number
    log_sum = np.zeros(n_params)
    for step in range(n_iter):
        deriv = np.abs(1.0 + TWO_PI * A_values * np.cos(TWO_PI * theta))
        log_sum += np.where(deriv > 0, np.log(deriv), -100.0)
        theta += D + A_values * np.sin(TWO_PI * theta)
        if (step + 1) % 5000 == 0:
            print(f"  {step + 1}/{n_iter} iterations")

    rho = (theta - theta_start) / n_iter
    lam = log_sum / n_iter

    np.savez_compressed(OUTPUT_NPZ, A=A_values, rho=rho, lam=lam)
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data):
    """Create the devil's staircase with Lyapunov overlay."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        CMAP_DIVERGING,
        COLORS,
        apply_axes_polish,
        figure_spec,
        setup,
    )
    setup()

    A = data["A"]
    rho = data["rho"]
    lam = data["lam"]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(spec.figsize[0], spec.figsize[1] * 1.64),
        sharex=True,
        height_ratios=[3, 1],
    )
    fig.subplots_adjust(hspace=0.08)

    # --- Top panel: Devil's staircase coloured by Lyapunov exponent ---
    # Use scatter with small points for colour mapping
    vmax = max(abs(lam.min()), abs(lam.max())) * 0.5
    sc = ax1.scatter(A, rho, c=lam, cmap=CMAP_DIVERGING, s=0.05,
                     vmin=-vmax, vmax=vmax, rasterized=True)
    ax1.set_ylabel(r"Rotation number $\rho$")
    ax1.set_ylim(0, 0.5)

    # Mark some key lockings
    lockings = {
        r"$\frac{1}{5}$": 0.2,
        r"$\frac{1}{4}$": 0.25,
        r"$\frac{2}{9}$": 2/9,
        r"$\frac{1}{3}$": 1/3,
    }
    for label, val in lockings.items():
        ax1.axhline(val, color=COLORS["grey"], lw=0.3, ls="--", alpha=0.5)
        ax1.text(0.002, val + 0.005, label, fontsize=spec.tick_size, color=COLORS["grey"])

    cb = fig.colorbar(sc, ax=ax1, pad=0.02, aspect=30)
    cb.set_label(r"Lyapunov exponent $\lambda$", fontsize=spec.label_size)
    cb.ax.tick_params(labelsize=spec.tick_size)

    # --- Bottom panel: Lyapunov exponent vs A ---
    ax2.plot(A, lam, color=COLORS["black"], lw=0.2, linestyle="-", rasterized=True)
    ax2.axhline(0, color=COLORS["red"], lw=0.5, ls="--")
    ax2.set_xlabel(r"Nonlinearity parameter $K$")
    ax2.set_ylabel(r"$\lambda$")
    ax2.set_xlim(0, 0.25)

    # Mark Kaneko's K_∞ ≈ 0.15671685 and K_c ≈ 0.18189
    ax2.axvline(0.15671685, color=COLORS["blue"], lw=0.5, ls=":", alpha=0.7)
    ax2.text(
        0.157,
        ax2.get_ylim()[1] * 0.8,
        r"$K_\infty$",
        fontsize=spec.tick_size,
        color=COLORS["blue"],
    )

    apply_axes_polish(ax1, kind="double")
    apply_axes_polish(ax2, kind="double")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Staircase zoom: self-similar nesting in the Farey region
# ---------------------------------------------------------------------------

def compute_zoom():
    """Recompute the staircase at high resolution over K ∈ [0.10, 0.17]."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_params = 50_000
    n_transient = 2000
    n_iter = 20_000
    D = 0.25

    A_values = np.linspace(0.10, 0.17, n_params)
    TWO_PI = 2.0 * np.pi

    theta = np.full(n_params, 0.1)
    for _ in range(n_transient):
        theta += D + A_values * np.sin(TWO_PI * theta)
    print("  Zoom transient done")

    theta_start = theta.copy()
    for step in range(n_iter):
        theta += D + A_values * np.sin(TWO_PI * theta)
        if (step + 1) % 5000 == 0:
            print(f"  Zoom: {step + 1}/{n_iter}")

    rho = (theta - theta_start) / n_iter
    np.savez_compressed(ZOOM_NPZ, A=A_values, rho=rho)
    print(f"Saved {ZOOM_NPZ}")


def plot_zoom(zoom_data, full_data):
    """Two-panel figure: full staircase with zoom box + zoomed Farey region."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    spec = figure_spec("double")
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2,
        figsize=(spec.figsize[0], spec.figsize[1]),
        width_ratios=[1, 1.2],
    )
    fig.subplots_adjust(wspace=0.30)

    # --- Left panel: full staircase with zoom box ---
    A_full = full_data["A"]
    rho_full = full_data["rho"]
    ax_full.plot(A_full, rho_full, color=COLORS["black"], lw=0.2, rasterized=True)
    ax_full.set_xlabel(r"Nonlinearity $K$")
    ax_full.set_ylabel(r"Rotation number $\rho$")
    ax_full.set_title("(a) Full staircase", loc="left")
    ax_full.set_xlim(0, 0.25)
    ax_full.set_ylim(0, 0.5)

    # Zoom box
    zoom_K = (0.10, 0.17)
    zoom_rho = (1.0 / 3.0 - 0.01, 1.0 / 2.0 + 0.01)
    rect = Rectangle(
        (zoom_K[0], zoom_rho[0]),
        zoom_K[1] - zoom_K[0],
        zoom_rho[1] - zoom_rho[0],
        linewidth=0.8, edgecolor=COLORS["red"], fill=False, ls="--",
    )
    ax_full.add_patch(rect)

    # --- Right panel: zoomed Farey region ---
    A_zoom = zoom_data["A"]
    rho_zoom = zoom_data["rho"]
    ax_zoom.plot(A_zoom, rho_zoom, color=COLORS["black"], lw=0.15, rasterized=True)
    ax_zoom.set_xlabel(r"Nonlinearity $K$")
    ax_zoom.set_ylabel(r"Rotation number $\rho$")
    ax_zoom.set_title("(b) Farey zoom", loc="left")
    ax_zoom.set_xlim(*zoom_K)
    ax_zoom.set_ylim(*zoom_rho)

    # Mark key Farey mediants
    mediants = {
        r"$\frac{1}{3}$": 1.0 / 3.0,
        r"$\frac{2}{5}$": 2.0 / 5.0,
        r"$\frac{3}{7}$": 3.0 / 7.0,
        r"$\frac{1}{2}$": 1.0 / 2.0,
    }
    for label, val in mediants.items():
        ax_zoom.axhline(val, color=COLORS["grey"], lw=0.3, ls="--", alpha=0.5)
        ax_zoom.text(
            zoom_K[0] + 0.001, val + 0.003,
            label, fontsize=spec.tick_size - 1, color=COLORS["grey"],
        )

    apply_axes_polish(ax_full, kind="double", title_loc="left")
    apply_axes_polish(ax_zoom, kind="double", title_loc="left")

    fig.savefig(ZOOM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ZOOM_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Devil's staircase
    try:
        data = _safe_load(OUTPUT_NPZ)
        print(f"Loaded {OUTPUT_NPZ} ({len(data['A'])} points)")
    except FileNotFoundError:
        print("Computing devil's staircase...")
        compute()
        data = _safe_load(OUTPUT_NPZ)
    plot(data)

    # Staircase zoom
    try:
        zoom_data = _safe_load(ZOOM_NPZ)
        print(f"Loaded {ZOOM_NPZ}")
    except FileNotFoundError:
        print("Computing staircase zoom...")
        compute_zoom()
        zoom_data = _safe_load(ZOOM_NPZ)
    plot_zoom(zoom_data, data)


if __name__ == "__main__":
    main()
