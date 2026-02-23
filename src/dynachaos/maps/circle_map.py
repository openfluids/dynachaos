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
    from dynachaos.utils.style import CMAP_DIVERGING, COLORS, DOUBLE_COL, setup
    setup()

    A = data["A"]
    rho = data["rho"]
    lam = data["lam"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, 5.5),
                                    sharex=True, height_ratios=[3, 1])
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
        ax1.text(0.002, val + 0.005, label, fontsize=7, color=COLORS["grey"])

    cb = fig.colorbar(sc, ax=ax1, pad=0.02, aspect=30)
    cb.set_label(r"Lyapunov exponent $\lambda$", fontsize=9)

    # --- Bottom panel: Lyapunov exponent vs A ---
    ax2.plot(A, lam, color=COLORS["black"], lw=0.2, linestyle="-", rasterized=True)
    ax2.axhline(0, color=COLORS["red"], lw=0.5, ls="--")
    ax2.set_xlabel(r"Nonlinearity parameter $K$")
    ax2.set_ylabel(r"$\lambda$")
    ax2.set_xlim(0, 0.25)

    # Mark Kaneko's K_∞ ≈ 0.15671685 and K_c ≈ 0.18189
    ax2.axvline(0.15671685, color=COLORS["blue"], lw=0.5, ls=":", alpha=0.7)
    ax2.text(0.157, ax2.get_ylim()[1] * 0.8, r"$K_\infty$", fontsize=8,
             color=COLORS["blue"])

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ} ({len(data['A'])} points)")
    except FileNotFoundError:
        print("Computing devil's staircase...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
