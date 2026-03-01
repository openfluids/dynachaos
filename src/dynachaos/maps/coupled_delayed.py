#!/usr/bin/env python3
"""
coupled_delayed: 4D coupled delayed logistic map for three-torus dynamics.

Reproduces Kaneko (1984) "Fates of Three-Torus. I", PTP 71(2), 282-294.

Map (Eq. 2.2):
    x_{n+1} = A x_n + D_A x_{n-1}(1 - x_{n-1}) + eps h_1(x_n, x_{n-1}, z_n, z_{n-1})
    z_{n+1} = A z_n + D_B z_{n-1}(1 - z_{n-1}) + eps h_2(x_n, x_{n-1}, z_n, z_{n-1})

With y_n = x_{n-1}, w_n = z_{n-1}, the state is (x, y, z, w).
Perturbations: h_1 = z_n - z_{n-1}, h_2 = x_{n-1} - x_n.
D_A = D_B + 0.1, A = 0.4 fixed.

Figures:
  - Lyapunov exponents vs D_B for different eps
  - (x_n, z_n) projections showing 3-torus, lockings, chaos

OUTPUTS: figures/sec06_three_torus/*.npz, *.png
"""

from dynachaos.io.paths import safe_load, section_dir
import numpy as np

FIG_DIR = section_dir("sec06_three_torus")

LYAP_NPZ = FIG_DIR / "lyapunov_vs_DB.npz"
LYAP_PNG = FIG_DIR / "lyapunov_vs_DB.png"
PROJ_NPZ = FIG_DIR / "xz_projections.npz"
PROJ_PNG = FIG_DIR / "xz_projections.png"
ANIM_NPZ = FIG_DIR / "three_torus_animation.npz"
ANIM_GIF = FIG_DIR / "three_torus_animation.gif"


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------

def coupled_delayed(state, A, DA, DB, eps):
    """One iteration of the 4D coupled delayed logistic map.

    State = (x, y, z, w) where y = x_{n-1}, w = z_{n-1}.
    """
    x, y, z, w = state
    h1 = z - w     # z_n - z_{n-1}
    h2 = y - x     # x_{n-1} - x_n
    x_new = A * x + DA * y * (1.0 - y) + eps * h1
    z_new = A * z + DB * w * (1.0 - w) + eps * h2
    y_new = x
    w_new = z
    return np.array([x_new, y_new, z_new, w_new])


def coupled_delayed_jac(state, A, DA, DB, eps):
    """Jacobian of the coupled delayed logistic map."""
    x, y, z, w = state
    return np.array([
        [A, DA * (1.0 - 2.0 * y), eps, -eps],
        [1.0, 0.0, 0.0, 0.0],
        [-eps, eps, A, DB * (1.0 - 2.0 * w)],
        [0.0, 0.0, 1.0, 0.0]
    ])


# ---------------------------------------------------------------------------
# Lyapunov computation
# ---------------------------------------------------------------------------

def compute_lyapunov():
    """Compute Lyapunov exponents vs D_B for several epsilon values."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    A = 0.4
    eps_values = [1e-3, 5e-3, 1e-2]
    n_params = 500
    DB_values = np.linspace(2.1, 2.65, n_params)

    all_spectra = {}
    for eps in eps_values:
        print(f"  eps={eps}")
        spectra = np.empty((n_params, 4))
        for i, DB in enumerate(DB_values):
            DA = DB + 0.1
            x0 = np.array([0.5, 0.5, 0.3, 0.3])
            f = lambda s, _DA=DA, _DB=DB, _e=eps: coupled_delayed(s, A, _DA, _DB, _e)
            jac = lambda s, _DA=DA, _DB=DB, _e=eps: coupled_delayed_jac(s, A, _DA, _DB, _e)
            spectra[i] = lyapunov_spectrum(f, jac, x0, n_iter=30_000,
                                           n_transient=10_000)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{n_params}")
        all_spectra[f"eps_{eps}_spectra"] = spectra

    np.savez_compressed(LYAP_NPZ, DB=DB_values,
                        eps_values=np.array(eps_values), **all_spectra)
    print(f"Saved {LYAP_NPZ}")


# ---------------------------------------------------------------------------
# (x, z) projections
# ---------------------------------------------------------------------------

def compute_projections():
    """Compute (x_n, z_n) projections at fixed eps, varying D_B."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 0.4
    eps = 5e-3
    DB_values = [2.35, 2.37, 2.45, 2.46, 2.47, 2.48]
    labels = ["3-torus", "locking", "locking", "torus", "chaos", "chaos"]

    results = {}
    for DB, label in zip(DB_values, labels):
        DA = DB + 0.1
        print(f"  DB={DB} ({label})")
        x0 = np.array([0.5, 0.5, 0.3, 0.3])
        state = x0.copy()
        for _ in range(30_000):
            state = coupled_delayed(state, A, DA, DB, eps)

        n_plot = 50_000
        traj = np.empty((n_plot, 4))
        for i in range(n_plot):
            state = coupled_delayed(state, A, DA, DB, eps)
            traj[i] = state
        results[f"DB_{DB}_xz"] = traj[:, [0, 2]]

    results["DB_values"] = np.array(DB_values)
    results["labels"] = np.array(labels)
    np.savez_compressed(PROJ_NPZ, **results)
    print(f"Saved {PROJ_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lyapunov(data):
    """Plot Lyapunov exponents vs D_B."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
    )
    setup()

    DB = data["DB"]
    eps_values = data["eps_values"]

    panel_spec = figure_spec("grid")
    fig, axes = plt.subplots(
        1,
        len(eps_values),
        figsize=(panel_spec.figsize[0], panel_spec.figsize[1] * 0.58),
        sharey=True,
    )
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        spectra = data[f"eps_{eps}_spectra"]
        for k in range(3):
            ax.plot(DB, spectra[:, k], lw=0.3,
                    label=rf"$\lambda_{k+1}$")
        ax.axhline(0, color=COLORS["red"], lw=0.3, ls="--")
        ax.set_xlabel(r"$D_B$")
        ax.set_title(rf"$\varepsilon = {eps}$", loc="left")
        apply_axes_polish(ax, kind="grid", title_loc="left")
        if idx == 0:
            ax.set_ylabel("Lyapunov exponent")
            finalize_legend(ax, kind="grid", loc="lower left")

    fig.suptitle(
        "Coupled delayed logistic map, $\\alpha = 0.4$, $D_A = D_B + 0.1$",
        fontsize=panel_spec.title_size,
        y=1.02,
    )
    fig.savefig(LYAP_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LYAP_PNG}")


def plot_projections(data):
    """Plot (x_n, z_n) projections."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    DB_values = data["DB_values"]
    labels = data["labels"]

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 3, figsize=spec.figsize)
    axes_flat = axes.flatten()

    for idx, DB in enumerate(DB_values):
        ax = axes_flat[idx]
        xz = data[f"DB_{DB}_xz"]
        ax.scatter(xz[:, 0], xz[:, 1], s=0.01, c=COLORS["black"], alpha=0.3,
                   rasterized=True)
        ax.set_title(f"$D_B={DB}$ ({labels[idx]})", loc="left")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
        apply_axes_polish(ax, kind="grid", title_loc="left")

    fig.suptitle(
        r"$(x_n, z_n)$ projections, $\varepsilon = 5 \times 10^{-3}$",
        fontsize=spec.title_size,
        y=1.02,
    )
    fig.savefig(PROJ_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PROJ_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def compute_animation_data():
    """Sweep DB from 2.1 to 2.65 for the three-torus animation."""
    from dynachaos.utils.animation import compute_animation_sweep

    A = 0.4
    eps = 5e-3
    DB_sweep = np.linspace(2.1, 2.65, 300)
    x0 = np.array([0.5, 0.5, 0.3, 0.3])

    def iterate_fn(DB):
        DA = DB + 0.1
        state = x0.copy()
        for _ in range(30_000):
            state = coupled_delayed(state, A, DA, DB, eps)
        traj = np.empty((10_000, 2))
        for i in range(10_000):
            state = coupled_delayed(state, A, DA, DB, eps)
            traj[i] = state[[0, 2]]  # project (x, z)
        return traj

    compute_animation_sweep(
        iterate_fn, DB_sweep, ANIM_NPZ, n_plot=10_000, progress_interval=50,
    )


def make_animation_gif(data):
    """Create the three-torus dynamics GIF."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"], data["all_x"], data["all_y"], ANIM_GIF,
        title_template=r"Three-torus dynamics, $\varepsilon = 0.005$, $D_B = {param_value}$",
        param_name="D_B", param_fmt=".4f",
        xlabel="$x$", ylabel="$z$",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        lyap_data = safe_load(LYAP_NPZ)
        print(f"Loaded {LYAP_NPZ}")
    except FileNotFoundError:
        print("Computing Lyapunov spectra...")
        compute_lyapunov()
        lyap_data = safe_load(LYAP_NPZ)
    plot_lyapunov(lyap_data)

    try:
        proj_data = safe_load(PROJ_NPZ)
        print(f"Loaded {PROJ_NPZ}")
    except FileNotFoundError:
        print("Computing projections...")
        compute_projections()
        proj_data = safe_load(PROJ_NPZ)
    plot_projections(proj_data)

    # Animation
    try:
        anim_data = safe_load(ANIM_NPZ)
        print(f"Loaded {ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing three-torus animation data...")
        compute_animation_data()
        anim_data = safe_load(ANIM_NPZ)
    make_animation_gif(anim_data)


if __name__ == "__main__":
    main()
