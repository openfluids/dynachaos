#!/usr/bin/env python3
"""
modulated_circle: Double devil's staircase in the modulated circle map.

Reproduces Kaneko (1984) "Fates of Three-Torus. I", PTP 71(2), 282-294, Sec 3.

Modulated circle map (Eq. 3.1):
    theta_{n+1} = theta_n + A sin(2 pi theta_n) + D + eps sin(2 pi phi_n)
    phi_{n+1}   = phi_n + C

where phi undergoes pure rigid rotation at irrational frequency C = (sqrt(5)-1)/2.

OUTPUTS: figures/sec06_three_torus/double_staircase.npz, .png
"""

from dynachaos.io.paths import safe_load, section_dir
import numpy as np

FIG_DIR = section_dir("sec06_three_torus")
OUTPUT_NPZ = FIG_DIR / "double_staircase.npz"
OUTPUT_PNG = FIG_DIR / "double_staircase.png"
ZOOM_PNG = FIG_DIR / "double_staircase_zoom.png"

# Golden-mean inverse (irrational modulation frequency)
C_GOLDEN = (np.sqrt(5) - 1) / 2


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------

def modulated_circle(state, A, C, D, eps):
    """One iteration of the modulated circle map (Kaneko Eq. 3.1)."""
    theta, phi = state
    theta_new = (theta + A * np.sin(2 * np.pi * theta) + D
                 + eps * np.sin(2 * np.pi * phi)) % 1.0
    phi_new = (phi + C) % 1.0
    return np.array([theta_new, phi_new])


# ---------------------------------------------------------------------------
# Rotation numbers
# ---------------------------------------------------------------------------

def rotation_numbers(A, C, D, eps,
                     n_transient=5000, n_iter=30_000, state0=None):
    """Compute both rotation numbers (rho_theta, rho_phi)."""
    if state0 is None:
        state0 = np.array([0.1, 0.1])

    theta_uw, phi_uw = float(state0[0]), float(state0[1])

    # Transient (unwrapped)
    for _ in range(n_transient):
        dt = A * np.sin(2 * np.pi * theta_uw) + D + eps * np.sin(2 * np.pi * phi_uw)
        theta_uw += dt
        phi_uw += C

    theta_start, phi_start = theta_uw, phi_uw
    for _ in range(n_iter):
        dt = A * np.sin(2 * np.pi * theta_uw) + D + eps * np.sin(2 * np.pi * phi_uw)
        theta_uw += dt
        phi_uw += C

    rho_theta = (theta_uw - theta_start) / n_iter
    rho_phi = (phi_uw - phi_start) / n_iter
    return rho_theta, rho_phi


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute():
    """Sweep D (bare frequency) and compute the double devil's staircase."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters following Kaneko Eq. 3.1
    A = 0.10          # subcritical (A < 1/2pi ~ 0.159)
    eps = 0.05        # moderate forcing amplitude
    C = C_GOLDEN      # golden-mean inverse

    n_params = 10_000
    D_values = np.linspace(0.0, 1.0, n_params)

    rho_theta = np.empty(n_params)
    rho_phi = np.empty(n_params)

    for i, D in enumerate(D_values):
        rt, rp = rotation_numbers(A, C, D, eps,
                                  n_transient=3000, n_iter=20_000)
        rho_theta[i] = rt
        rho_phi[i] = rp

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_params}")
            np.savez_compressed(OUTPUT_NPZ, D=D_values[:i+1],
                                rho_theta=rho_theta[:i+1],
                                rho_phi=rho_phi[:i+1],
                                A=np.array([A]), C=np.array([C]),
                                eps=np.array([eps]))

    np.savez_compressed(OUTPUT_NPZ, D=D_values,
                        rho_theta=rho_theta, rho_phi=rho_phi,
                        A=np.array([A]), C=np.array([C]),
                        eps=np.array([eps]))
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data):
    """Plot the double devil's staircase."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    D = data["D"]
    rho_theta = data["rho_theta"]
    rho_phi = data["rho_phi"]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(spec.figsize[0], spec.figsize[1] * 1.19),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.subplots_adjust(hspace=0.08)

    ax1.plot(D, rho_theta, color=COLORS["black"], linestyle="-", lw=0.2, rasterized=True)
    ax1.set_ylabel(r"$\rho_\theta$")
    ax1.set_ylim(-0.02, 1.02)

    ax2.plot(D, rho_phi, color=COLORS["blue"], linestyle="-", lw=0.2, rasterized=True)
    ax2.axhline(C_GOLDEN, color=COLORS["red"], ls="--", lw=0.5, zorder=0)
    ax2.set_ylabel(r"$\rho_\varphi$")
    ax2.set_xlabel(r"$D$")
    ax2.set_ylim(0.615, 0.622)

    apply_axes_polish(ax1, kind="double", title_loc="left")
    apply_axes_polish(ax2, kind="double")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Zoomed double staircase: partial locking windows
# ---------------------------------------------------------------------------

def plot_zoom(data):
    """Two-panel zoom into different D windows showing rho_theta staircase detail."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    D = data["D"]
    rho_theta = data["rho_theta"]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(spec.figsize[0], spec.figsize[1]),
    )
    fig.subplots_adjust(wspace=0.30)

    # --- Left panel: window near D ∈ [0.10, 0.30] ---
    mask1 = (D >= 0.10) & (D <= 0.30)
    rt1 = rho_theta[mask1]
    ax1.plot(D[mask1], rt1, color=COLORS["black"], lw=0.3, rasterized=True)
    ax1.set_xlabel(r"$D$")
    ax1.set_ylabel(r"$\rho_\theta$")
    ax1.set_title(r"(a) $D \in [0.10, 0.30]$", loc="left")
    if rt1.size > 0:
        ax1.set_ylim(rt1.min() - 0.02, rt1.max() + 0.02)

    # --- Right panel: window near D ∈ [0.40, 0.60] ---
    mask2 = (D >= 0.40) & (D <= 0.60)
    rt2 = rho_theta[mask2]
    ax2.plot(D[mask2], rt2, color=COLORS["black"], lw=0.3, rasterized=True)
    ax2.set_xlabel(r"$D$")
    ax2.set_title(r"(b) $D \in [0.40, 0.60]$", loc="left")
    if rt2.size > 0:
        ax2.set_ylim(rt2.min() - 0.02, rt2.max() + 0.02)

    apply_axes_polish(ax1, kind="double", title_loc="left")
    apply_axes_polish(ax2, kind="double", title_loc="left")

    fig.savefig(ZOOM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ZOOM_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(OUTPUT_NPZ)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing double devil's staircase...")
        compute()
        data = safe_load(OUTPUT_NPZ)
    plot(data)
    plot_zoom(data)


if __name__ == "__main__":
    main()
