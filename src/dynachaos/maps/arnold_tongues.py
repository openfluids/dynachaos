#!/usr/bin/env python3
"""
arnold_tongues: Arnold tongue structure of the circle map in the (Ω, K) plane.

Displays the rotation number ρ(Ω, K) as a 2D color map, revealing the
wedge-shaped tongue structure.  Above the critical line K_c = 1/(2π),
tongues overlap and chaos appears.

Map:  θ_{n+1} = θ_n + Ω + K sin(2πθ_n)  (mod 1)

OUTPUTS: figures/sec02_circle_map/arnold_tongues.npz,
         figures/sec02_circle_map/arnold_tongues.png
USAGE:   python src/dynachaos/maps/arnold_tongues.py                        # .npz exists → plot only
         rm figures/sec02_circle_map/arnold_tongues.npz && python src/dynachaos/maps/arnold_tongues.py  # recompute
"""

from dynachaos.io.paths import safe_load, section_dir
import numpy as np

FIG_DIR = section_dir("sec02_circle_map")
OUTPUT_NPZ = FIG_DIR / "arnold_tongues.npz"
OUTPUT_PNG = FIG_DIR / "arnold_tongues.png"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute():
    """Sweep (Ω, K) and compute rotation numbers — fully vectorized."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_omega = 1000
    n_K = 500
    n_transient = 2000
    n_iter = 10_000

    Omega_values = np.linspace(0.0, 1.0, n_omega)
    K_values = np.linspace(0.0, 0.3, n_K)

    # 2D grid: shape (n_K, n_omega) — K varies along rows, Ω along columns
    Omega_grid, K_grid = np.meshgrid(Omega_values, K_values)
    Omega_flat = Omega_grid.ravel()
    K_flat = K_grid.ravel()
    n_total = len(Omega_flat)

    TWO_PI = 2.0 * np.pi
    theta = np.full(n_total, 0.1)

    # Transient (unwrapped)
    for step in range(n_transient):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)
        if (step + 1) % 1000 == 0:
            print(f"  Transient: {step + 1}/{n_transient}")

    theta_start = theta.copy()

    # Main iteration for rotation number
    for step in range(n_iter):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)
        if (step + 1) % 5000 == 0:
            print(f"  Iteration: {step + 1}/{n_iter}")

    rho = (theta - theta_start) / n_iter
    rho_2d = rho.reshape(n_K, n_omega)

    np.savez_compressed(
        OUTPUT_NPZ,
        Omega=Omega_values,
        K=K_values,
        rho=rho_2d,
    )
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data):
    """Plot the Arnold tongue structure as a 2D color map."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        COLORS,
        apply_axes_polish,
        figure_spec,
        setup,
    )
    setup()

    Omega = data["Omega"]
    K = data["K"]
    rho = data["rho"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    pcm = ax.pcolormesh(
        Omega, K, rho,
        cmap=CMAP_SEQUENTIAL,
        shading="auto",
        rasterized=True,
    )

    # Critical line K_c = 1/(2π)
    K_c = 1.0 / (2.0 * np.pi)
    ax.axhline(K_c, color=COLORS["red"], lw=0.8, ls="--", alpha=0.8)
    ax.text(
        0.02, K_c + 0.005,
        r"$K_c = 1/(2\pi)$",
        fontsize=spec.tick_size,
        color=COLORS["red"],
    )

    # Label main tongues
    tongue_labels = {
        r"$\frac{0}{1}$": 0.0,
        r"$\frac{1}{3}$": 1.0 / 3.0,
        r"$\frac{1}{2}$": 0.5,
        r"$\frac{2}{3}$": 2.0 / 3.0,
        r"$\frac{1}{1}$": 1.0,
    }
    for label, omega_val in tongue_labels.items():
        ax.text(
            omega_val, K[-1] + 0.005,
            label,
            fontsize=spec.tick_size,
            ha="center",
            color=COLORS["black"],
        )

    ax.set_xlabel(r"Bare frequency $\Omega$")
    ax.set_ylabel(r"Nonlinearity $K$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, K[-1])

    cb = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=30)
    cb.set_label(r"Rotation number $\rho$", fontsize=spec.label_size)
    cb.ax.tick_params(labelsize=spec.tick_size)

    apply_axes_polish(ax, kind="double")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(OUTPUT_NPZ)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing Arnold tongues...")
        compute()
        data = safe_load(OUTPUT_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
