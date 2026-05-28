#!/usr/bin/env python3
"""
torus_doubling: Doubling of torus in 3- and 4-dimensional delayed logistic maps.

Reproduces Kaneko (1983) "Doubling of Torus", PTP 69(6), 1806-1810.

Maps investigated:
  (I)  X_{n+1} = AX_n + (1-A)(1-DY_n^2), Y_{n+1} = Z_n, Z_{n+1} = X_n;  A=0.4
  (IV) X_{n+1} = AX_n + (1-A)(1-DY_n^2), Y_{n+1} = Z_n,
       Z_{n+1} = AZ_n + (1-A)(1-DW_n^2), W_{n+1} = X_n;  A=0.3

Figures:
  - Projections onto (X,Y) plane for Map (I): D=2.11 (torus), 2.16 (2x torus), 2.19 (chaos)
  - Projections for Map (IV): D=1.515 (4x torus), 1.5206 (8x torus), 1.5212 (chaos)
  - Lyapunov exponents for Map (IV) vs D

OUTPUTS: figures/sec04_doubling/*.npz, *.png
"""

import numpy as np

from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps._iter import run_animation_sweep, trajectory_after_transient
from dynachaos.maps.primitives import logistic, logistic_derivative

FIG_DIR = section_dir("sec04_doubling")

MAP1_NPZ = FIG_DIR / "map_I_attractors.npz"
MAP1_PNG = FIG_DIR / "map_I_attractors.png"
MAP4_NPZ = FIG_DIR / "map_IV_attractors.npz"
MAP4_PNG = FIG_DIR / "map_IV_attractors.png"
LYAP_NPZ = FIG_DIR / "map_IV_lyapunov.npz"
LYAP_PNG = FIG_DIR / "map_IV_lyapunov.png"
MAP1_ANIM_NPZ = FIG_DIR / "map_I_animation.npz"
MAP1_ANIM_GIF = FIG_DIR / "map_I_animation.gif"
MAP4_ANIM_NPZ = FIG_DIR / "map_IV_animation.npz"
MAP4_ANIM_GIF = FIG_DIR / "map_IV_animation.gif"


def _write_payload(output_path, payload):
    """Write a compute payload when requested and return it unchanged."""
    if output_path is None:
        return payload
    output_path = FIG_DIR / output_path if isinstance(output_path, str) else output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"Saved {output_path}")
    return payload


def _load_or_compute_payload(npz_path, section_name, compute_fn, *, required_keys=()):
    """Load a cache file, or use the returned compute payload when recomputing."""
    path = npz_path
    keys = tuple(required_keys)
    try:
        data = safe_load(path)
        missing = tuple(key for key in keys if key not in data.files)
        if not missing:
            print(f"Loaded {path}")
            return data
        data.close()
        missing_text = ", ".join(missing)
        print(f"Cache {path} missing keys ({missing_text}); recomputing {section_name}...")
    except FileNotFoundError:
        print(f"Computing {section_name}...")

    payload = compute_fn()
    missing = tuple(key for key in keys if key not in payload)
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Cache {path} is missing required keys after compute: {missing_text}")
    return payload


# ---------------------------------------------------------------------------
# Map definitions
# ---------------------------------------------------------------------------


def map_I(state, A, D):
    """Map (I): 3D delayed logistic. State = (X, Y, Z)."""
    X, Y, Z = state
    X_new = A * X + (1.0 - A) * logistic(Y, D)
    Y_new = Z
    Z_new = X
    return np.array([X_new, Y_new, Z_new])


def map_I_jac(state, A, D):
    """Jacobian of Map (I)."""
    X, Y, Z = state
    return np.array(
        [[A, (1.0 - A) * logistic_derivative(Y, D), 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )


def map_IV(state, A, D):
    """Map (IV): 4D delayed logistic. State = (X, Y, Z, W)."""
    X, Y, Z, W = state
    X_new = A * X + (1.0 - A) * logistic(Y, D)
    Y_new = Z
    Z_new = A * Z + (1.0 - A) * logistic(W, D)
    W_new = X
    return np.array([X_new, Y_new, Z_new, W_new])


def map_IV_jac(state, A, D):
    """Jacobian of Map (IV)."""
    X, Y, Z, W = state
    return np.array(
        [
            [A, (1.0 - A) * logistic_derivative(Y, D), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, A, (1.0 - A) * logistic_derivative(W, D)],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )


# ---------------------------------------------------------------------------
# Attractor computation
# ---------------------------------------------------------------------------


def iterate_map(f, x0, A, D, n_transient=20_000, n_plot=100_000):
    """Iterate a map and return trajectory after transient."""
    return trajectory_after_transient(
        x0,
        lambda state: f(state, A, D),
        n_transient,
        n_plot,
        diverged_fn=lambda state: np.any(np.abs(state) > 1e10),
        allow_partial=True,
    )


def compute_map_I(
    *,
    A=0.4,
    D_values=None,
    n_transient=20_000,
    n_plot=100_000,
    output_path=MAP1_NPZ,
):
    """Compute Map (I) attractors at A=0.4."""
    if D_values is None:
        D_values = [2.11, 2.16, 2.19]
    else:
        D_values = list(np.atleast_1d(np.asarray(D_values, dtype=np.float64)))
    labels = ["torus", "2x torus", "chaos"]

    results = {}
    x0 = np.array([0.5, 0.5, 0.5])
    for idx, D in enumerate(D_values):
        label = labels[idx] if idx < len(labels) else ""
        print(f"  Map (I): D={D} ({label})")
        traj = iterate_map(map_I, x0, A, D, n_transient=n_transient, n_plot=n_plot)
        if traj is not None:
            results[f"D_{D}_traj"] = traj
    results["D_values"] = np.array(D_values)
    return _write_payload(output_path, results)


def compute_map_IV(
    *,
    A=0.3,
    D_values=None,
    n_transient=20_000,
    n_plot=100_000,
    output_path=MAP4_NPZ,
):
    """Compute Map (IV) attractors at A=0.3."""
    if D_values is None:
        D_values = [1.515, 1.5206, 1.5212]
    else:
        D_values = list(np.atleast_1d(np.asarray(D_values, dtype=np.float64)))
    labels = ["4x torus", "8x torus", "chaos"]

    results = {}
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    for idx, D in enumerate(D_values):
        label = labels[idx] if idx < len(labels) else ""
        print(f"  Map (IV): D={D} ({label})")
        traj = iterate_map(map_IV, x0, A, D, n_transient=n_transient, n_plot=n_plot)
        if traj is not None:
            results[f"D_{D}_traj"] = traj
    results["D_values"] = np.array(D_values)
    return _write_payload(output_path, results)


def compute_map_IV_lyapunov(
    *,
    A=0.3,
    D_values=None,
    n_iter=50_000,
    n_transient=20_000,
    output_path=LYAP_NPZ,
    progress_interval=200,
):
    """Compute Lyapunov exponents for Map (IV) vs D."""
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    if D_values is None:
        D_values = np.linspace(1.48, 1.53, 1000)
    else:
        D_values = np.atleast_1d(np.asarray(D_values, dtype=np.float64))
    n_params = len(D_values)

    spectra = np.empty((n_params, 4))
    x0 = np.array([0.5, 0.5, 0.5, 0.5])

    for i, D in enumerate(D_values):

        def f(state, _D=D):
            return map_IV(state, A, _D)

        def jac(state, _D=D):
            return map_IV_jac(state, A, _D)

        spectra[i] = lyapunov_spectrum(f, jac, x0, n_iter=n_iter, n_transient=n_transient)
        if output_path is not None and progress_interval and (i + 1) % progress_interval == 0:
            print(f"  Lyapunov: {i + 1}/{n_params}")
            _write_payload(output_path, {"D": D_values[: i + 1], "spectra": spectra[: i + 1]})

    return _write_payload(output_path, {"D": D_values, "spectra": spectra})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_map_I(data):
    """Plot Map (I) attractor projections onto (X, Y) plane."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup

    setup()

    D_values = data["D_values"]
    labels = ["torus", r"$2\times$torus", "chaos"]

    spec = figure_spec("grid")
    fig, axes = plt.subplots(1, 3, figsize=(spec.figsize[0], 2.75))
    fig.subplots_adjust(wspace=0.24)
    for idx, D in enumerate(D_values):
        ax = axes[idx]
        key = f"D_{D}_traj"
        if key in data:
            traj = data[key]
            ax.scatter(
                traj[:, 0], traj[:, 1], s=0.012, c=COLORS["black"], alpha=0.22, rasterized=True
            )
        ax.set_title(f"$D={D}$ ({labels[idx]})", loc="left")
        ax.set_xlabel("$X$")
        if idx == 0:
            ax.set_ylabel("$Y$")
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False, equal=True)

    fig.suptitle(
        "Map (I), $\\alpha = 0.4$: projection onto $(X, Y)$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )
    fig.savefig(MAP1_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {MAP1_PNG}")


def plot_map_IV(data):
    """Plot Map (IV) attractor projections onto (X, Y) plane with zoom insets."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup

    setup()

    D_values = data["D_values"]
    labels = [r"$4\times$torus", r"$8\times$torus", "chaos"]

    spec = figure_spec("grid")
    fig, axes = plt.subplots(1, 3, figsize=(spec.figsize[0], 3.0))
    fig.subplots_adjust(wspace=0.34)
    for idx, D in enumerate(D_values):
        ax = axes[idx]
        key = f"D_{D}_traj"
        if key in data:
            traj = data[key]
            ax.scatter(
                traj[:, 0], traj[:, 1], s=0.01, c=COLORS["black"], alpha=0.3, rasterized=True
            )

            # Add zoom inset to reveal strand structure
            axins = inset_axes(ax, width="40%", height="40%", loc="upper left", borderpad=0.5)
            axins.scatter(
                traj[:, 0], traj[:, 1], s=0.02, c=COLORS["black"], alpha=0.5, rasterized=True
            )
            # Zoom into a region showing strand separation
            xmid = np.median(traj[:, 0])
            ymid = np.median(traj[:, 1])
            xspan = np.ptp(traj[:, 0]) * 0.12
            yspan = np.ptp(traj[:, 1]) * 0.12
            axins.set_xlim(xmid - xspan, xmid + xspan)
            axins.set_ylim(ymid - yspan, ymid + yspan)
            axins.tick_params(labelsize=5)
            axins.set_xticks([])
            axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_edgecolor(COLORS["red"])
                spine.set_linewidth(0.6)

        ax.set_title(f"$D={D}$ ({labels[idx]})", loc="left")
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        apply_axes_polish(ax, kind="grid", title_loc="left")

    fig.suptitle(
        "Map (IV), $\\alpha = 0.3$: projection onto $(X, Y)$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )
    fig.savefig(MAP4_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {MAP4_PNG}")


def plot_lyapunov(data):
    """Plot Lyapunov exponents for Map (IV) vs D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
    )

    setup()

    D = data["D"]
    spectra = data["spectra"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    labels_le = [r"$\lambda_1$", r"$\lambda_2$"]
    colors = [COLORS["black"], COLORS["blue"]]
    linewidths = [0.6, 1.0]
    # Only plot first two (3rd and 4th are large negative, as Kaneko notes)
    for k in range(2):
        ax.plot(
            D, spectra[:, k], color=colors[k], lw=linewidths[k], label=labels_le[k], zorder=3 - k
        )
    ax.axhline(0, color=COLORS["red"], lw=0.4, ls="--", zorder=1)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title("Map (IV), $\\alpha = 0.3$", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
    finalize_legend(ax, kind="double", loc="upper right")

    axins = inset_axes(ax, width="38%", height="42%", loc="lower left", borderpad=1.0)
    axins.plot(D, spectra[:, 1], color=COLORS["blue"], lw=0.8)
    axins.axhline(0, color=COLORS["red"], lw=0.4, ls="--")
    axins.set_xlim(D.min(), D.max())
    axins.set_ylim(-1.8e-4, 2.0e-5)
    axins.set_title(r"$\lambda_2$ near zero", loc="left", fontsize=spec.tick_size)
    apply_axes_polish(axins, kind="grid", title_loc="left", grid=False)
    axins.tick_params(labelsize=spec.tick_size - 0.8)

    fig.savefig(LYAP_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LYAP_PNG}")


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------


def compute_map_I_animation():
    """Sweep D from 1.9 to 2.25 for Map (I) animation at A=0.4."""
    A = 0.4
    D_sweep = np.linspace(1.9, 2.25, 200)
    x0 = np.array([0.5, 0.5, 0.5])

    def iterate_fn(D):
        traj = iterate_map(map_I, x0, A, D, n_transient=20_000, n_plot=5_000)
        if traj is None:
            return None
        return traj[:, :2]

    return run_animation_sweep(iterate_fn, D_sweep, MAP1_ANIM_NPZ, n_plot=5_000)


def make_map_I_animation_gif(data):
    """Create GIF for Map (I) attractor evolution."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"],
        data["all_x"],
        data["all_y"],
        MAP1_ANIM_GIF,
        title_template=r"Map (I), $\alpha = 0.4$, $D = {param_value}$",
        param_name="D",
        param_fmt=".3f",
        xlabel="$X$",
        ylabel="$Y$",
    )


def compute_map_IV_animation():
    """Sweep D from 1.48 to 1.53 for Map (IV) animation at A=0.3."""
    A = 0.3
    D_sweep = np.linspace(1.48, 1.53, 200)
    x0 = np.array([0.5, 0.5, 0.5, 0.5])

    def iterate_fn(D):
        traj = iterate_map(map_IV, x0, A, D, n_transient=20_000, n_plot=5_000)
        if traj is None:
            return None
        return traj[:, :2]

    return run_animation_sweep(iterate_fn, D_sweep, MAP4_ANIM_NPZ, n_plot=5_000)


def make_map_IV_animation_gif(data):
    """Create GIF for Map (IV) attractor evolution."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"],
        data["all_x"],
        data["all_y"],
        MAP4_ANIM_GIF,
        title_template=r"Map (IV), $\alpha = 0.3$, $D = {param_value}$",
        param_name="D",
        param_fmt=".4f",
        xlabel="$X$",
        ylabel="$Y$",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    m1_data = _load_or_compute_payload(
        MAP1_NPZ,
        "Map (I) attractors",
        compute_map_I,
        required_keys=("D_values",),
    )
    plot_map_I(m1_data)

    m4_data = _load_or_compute_payload(
        MAP4_NPZ,
        "Map (IV) attractors",
        compute_map_IV,
        required_keys=("D_values",),
    )
    plot_map_IV(m4_data)

    lyap_data = _load_or_compute_payload(
        LYAP_NPZ,
        "Map (IV) Lyapunov exponents",
        compute_map_IV_lyapunov,
        required_keys=("D", "spectra"),
    )
    plot_lyapunov(lyap_data)

    # Map (I) animation
    try:
        m1_anim = safe_load(MAP1_ANIM_NPZ)
        print(f"Loaded {MAP1_ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing Map (I) animation data...")
        m1_anim = compute_map_I_animation()
    make_map_I_animation_gif(m1_anim)

    # Map (IV) animation
    try:
        m4_anim = safe_load(MAP4_ANIM_NPZ)
        print(f"Loaded {MAP4_ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing Map (IV) animation data...")
        m4_anim = compute_map_IV_animation()
    make_map_IV_animation_gif(m4_anim)


if __name__ == "__main__":
    main()
