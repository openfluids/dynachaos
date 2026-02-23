#!/usr/bin/env python3
"""
modulated_circle: Double devil's staircase in the modulated circle map.

Reproduces Kaneko (1984) "Fates of Three-Torus. I", PTP 71(2), 282-294, Sec 3.

Modulated circle map (Eq. 1.4):
    theta_{n+1} = theta_n + A sin(2 pi theta_n) + D + eps sin(2 pi phi_n)
    phi_{n+1}   = phi_n   + B sin(2 pi phi_n)   + C + eps' sin(2 pi theta_n)

This is the simplest model exhibiting the "double devil's staircase" -- the
rotation numbers for both theta and phi form independent devil's staircases
as a function of the bifurcation parameter.

OUTPUTS: figures/sec06_three_torus/double_staircase.npz, .png
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec06_three_torus")
OUTPUT_NPZ = FIG_DIR / "double_staircase.npz"
OUTPUT_PNG = FIG_DIR / "double_staircase.png"


def _safe_load(path):
    """Load .npz with pickle disabled for safety."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------

def modulated_circle(state, A, B, C, D, eps, eps_prime):
    """One iteration of the modulated circle map."""
    theta, phi = state
    theta_new = (theta + A * np.sin(2 * np.pi * theta) + D
                 + eps * np.sin(2 * np.pi * phi)) % 1.0
    phi_new = (phi + B * np.sin(2 * np.pi * phi) + C
               + eps_prime * np.sin(2 * np.pi * theta)) % 1.0
    return np.array([theta_new, phi_new])


# ---------------------------------------------------------------------------
# Rotation numbers
# ---------------------------------------------------------------------------

def rotation_numbers(A, B, C, D, eps, eps_prime,
                     n_transient=5000, n_iter=30_000, state0=None):
    """Compute both rotation numbers (rho_theta, rho_phi)."""
    if state0 is None:
        state0 = np.array([0.1, 0.1])

    theta_uw, phi_uw = float(state0[0]), float(state0[1])

    # Transient (unwrapped)
    for _ in range(n_transient):
        dt = A * np.sin(2 * np.pi * theta_uw) + D + eps * np.sin(2 * np.pi * phi_uw)
        dp = B * np.sin(2 * np.pi * phi_uw) + C + eps_prime * np.sin(2 * np.pi * theta_uw)
        theta_uw += dt
        phi_uw += dp

    theta_start, phi_start = theta_uw, phi_uw
    for _ in range(n_iter):
        dt = A * np.sin(2 * np.pi * theta_uw) + D + eps * np.sin(2 * np.pi * phi_uw)
        dp = B * np.sin(2 * np.pi * phi_uw) + C + eps_prime * np.sin(2 * np.pi * theta_uw)
        theta_uw += dt
        phi_uw += dp

    rho_theta = (theta_uw - theta_start) / n_iter
    rho_phi = (phi_uw - phi_start) / n_iter
    return rho_theta, rho_phi


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute():
    """Sweep C and compute the double devil's staircase."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters following Kaneko's choices
    A = 0.15
    B = 0.12
    D = 0.25
    eps = 0.02
    eps_prime = 0.015

    n_params = 10_000
    C_values = np.linspace(0.0, 0.5, n_params)

    rho_theta = np.empty(n_params)
    rho_phi = np.empty(n_params)

    for i, C in enumerate(C_values):
        rt, rp = rotation_numbers(A, B, C, D, eps, eps_prime,
                                  n_transient=3000, n_iter=20_000)
        rho_theta[i] = rt
        rho_phi[i] = rp

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_params}")
            np.savez_compressed(OUTPUT_NPZ, C=C_values[:i+1],
                                rho_theta=rho_theta[:i+1],
                                rho_phi=rho_phi[:i+1],
                                A=np.array([A]), B=np.array([B]),
                                D=np.array([D]), eps=np.array([eps]),
                                eps_prime=np.array([eps_prime]))

    np.savez_compressed(OUTPUT_NPZ, C=C_values,
                        rho_theta=rho_theta, rho_phi=rho_phi,
                        A=np.array([A]), B=np.array([B]),
                        D=np.array([D]), eps=np.array([eps]),
                        eps_prime=np.array([eps_prime]))
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data):
    """Plot the double devil's staircase."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, DOUBLE_COL, setup
    setup()

    C = data["C"]
    rho_theta = data["rho_theta"]
    rho_phi = data["rho_phi"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, 4.0),
                                    sharex=True)
    fig.subplots_adjust(hspace=0.15)

    ax1.plot(C, rho_theta, color=COLORS["black"], linestyle="-", lw=0.2, rasterized=True)
    ax1.set_ylabel(r"$\rho_\theta$")
    ax1.set_title(r"Double devil's staircase (modulated circle map)", fontsize=10)

    ax2.plot(C, rho_phi, color=COLORS["blue"], linestyle="-", lw=0.2, rasterized=True)
    ax2.set_ylabel(r"$\rho_\varphi$")
    ax2.set_xlabel(r"$\Omega_2$")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = _safe_load(OUTPUT_NPZ)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing double devil's staircase...")
        compute()
        data = _safe_load(OUTPUT_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
