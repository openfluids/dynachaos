#!/usr/bin/env python3
"""
coupled_logistic: Transition from torus to chaos in the coupled logistic map.

Reproduces Kaneko (1983) "Transition from Torus to Chaos Accompanied by
Frequency Lockings with Symmetry Breaking", PTP 69(5), 1427-1442.

Map (Eq. 1.1):
    x_{n+1} = 1 - A x_n^2 + D(y_n - x_n)
    y_{n+1} = 1 - A y_n^2 + D(x_n - y_n)

Figures:
  - Phase diagram in (A, D) space
  - Attractor portraits at D=0.1 for several A values
  - Basin of attraction showing self-similar stripe structure

OUTPUTS: figures/sec03_transition/phase_diagram.npz, phase_diagram.png
         figures/sec03_transition/attractors.npz, attractors.png
         figures/sec03_transition/basins.npz, basins.png
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec03_transition")

PHASE_NPZ = FIG_DIR / "phase_diagram.npz"
PHASE_PNG = FIG_DIR / "phase_diagram.png"
ATTR_NPZ = FIG_DIR / "attractors.npz"
ATTR_PNG = FIG_DIR / "attractors.png"
BASIN_NPZ = FIG_DIR / "basins.npz"
BASIN_PNG = FIG_DIR / "basins.png"
ANIM_NPZ = FIG_DIR / "attractors_animation.npz"
ANIM_GIF = FIG_DIR / "attractors_animation.gif"


def _safe_load(path):
    """Load .npz safely (no deserialization of arbitrary objects)."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------

def coupled_logistic(state, A, D):
    """One iteration of the coupled logistic map."""
    x, y = state
    x_new = 1.0 - A * x * x + D * (y - x)
    y_new = 1.0 - A * y * y + D * (x - y)
    return np.array([x_new, y_new])


def coupled_logistic_jac(state, A, D):
    """Jacobian of the coupled logistic map."""
    x, y = state
    return np.array([
        [-2.0 * A * x - D, D],
        [D, -2.0 * A * y - D]
    ])


# ---------------------------------------------------------------------------
# Phase diagram computation
# ---------------------------------------------------------------------------

def compute_phase_diagram():
    """Compute phase diagram in (A, D) space.

    Vectorized: for each D, all A values are iterated simultaneously as
    NumPy arrays.  Uses temporal variance of x as the phase diagnostic
    (fast proxy for Lyapunov-based classification).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_A, n_D = 500, 200
    n_transient = 5000
    n_sample = 2000
    A_values = np.linspace(0.5, 1.65, n_A)
    D_values = np.linspace(0.0, 0.3, n_D)

    var_grid = np.empty((n_D, n_A))

    for j, D in enumerate(D_values):
        # All A values in parallel
        x = np.full(n_A, 0.1)
        y = np.full(n_A, 0.2)

        # Transient
        for _ in range(n_transient):
            x_new = 1.0 - A_values * x * x + D * (y - x)
            y_new = 1.0 - A_values * y * y + D * (x - y)
            x, y = x_new, y_new
            # Clamp divergent orbits
            mask = (np.abs(x) > 1e10) | (np.abs(y) > 1e10)
            x = np.where(mask, np.nan, x)
            y = np.where(mask, np.nan, y)

        # Sample: accumulate mean and variance of x
        sum_x = np.zeros(n_A)
        sum_x2 = np.zeros(n_A)
        for _ in range(n_sample):
            x_new = 1.0 - A_values * x * x + D * (y - x)
            y_new = 1.0 - A_values * y * y + D * (x - y)
            x, y = x_new, y_new
            mask = (np.abs(x) > 1e10) | (np.abs(y) > 1e10)
            x = np.where(mask, np.nan, x)
            y = np.where(mask, np.nan, y)
            sum_x += np.where(np.isnan(x), 0.0, x)
            sum_x2 += np.where(np.isnan(x), 0.0, x * x)

        mean_x = sum_x / n_sample
        var_grid[j] = sum_x2 / n_sample - mean_x * mean_x

        if (j + 1) % 50 == 0:
            print(f"  Phase diagram: row {j + 1}/{n_D}")
            np.savez_compressed(PHASE_NPZ, A=A_values, D=D_values,
                                var=var_grid)

    np.savez_compressed(PHASE_NPZ, A=A_values, D=D_values, var=var_grid)
    print(f"Saved {PHASE_NPZ}")


# ---------------------------------------------------------------------------
# Attractor portraits at D=0.1
# ---------------------------------------------------------------------------

def compute_attractors():
    """Compute attractor portraits at D=0.1 for several A values."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    D = 0.1
    A_values = [1.10, 1.25, 1.35, 1.3525, 1.355, 1.373]
    labels = ["2T", "4T", "8T", "8T (zoom)", "8C", "4C"]

    results = {}
    for A, label in zip(A_values, labels):
        print(f"  A={A} ({label})")
        state = np.array([0.1, 0.2])
        for _ in range(50_000):
            state = coupled_logistic(state, A, D)
        traj = np.empty((100_000, 2))
        for i in range(100_000):
            state = coupled_logistic(state, A, D)
            traj[i] = state
        results[f"A_{A}_x"] = traj[:, 0]
        results[f"A_{A}_y"] = traj[:, 1]

    results["A_values"] = np.array(A_values)
    results["D"] = np.array([D])
    np.savez_compressed(ATTR_NPZ, **results)
    print(f"Saved {ATTR_NPZ}")


# ---------------------------------------------------------------------------
# Basin of attraction (self-similar stripe structure)
# ---------------------------------------------------------------------------

def _find_reference_orbit(A, D, x0, y0, n_transient=500_000, period=32):
    """Find a reference periodic orbit by iterating from (x0, y0)."""
    x, y = x0, y0
    for _ in range(n_transient):
        x, y = 1.0 - A * x * x + D * (y - x), 1.0 - A * y * y + D * (x - y)
    ref = np.empty((period, 2))
    for i in range(period):
        x, y = 1.0 - A * x * x + D * (y - x), 1.0 - A * y * y + D * (x - y)
        ref[i] = [x, y]
    return ref


def compute_basins():
    """Compute basin of attraction showing stripe structure.

    At A=1.35344, D=0.1, two coexisting asymmetric period-32 cycles
    exist (Kaneko 1983, Fig. 8).  They are mirror images about y=x.
    The basin boundary between them forms self-similar stripes near
    the y=x line.

    Classification: after a long transient, the final state is compared
    to both reference orbits; the closer one determines the basin.

    Vectorized: each row of the grid processes all x values in parallel.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 1.35344
    D = 0.1
    n_grid = 800
    n_transient = 50_000

    # Pre-compute the two reference orbits (mirror images about y=x)
    print("  Computing reference orbits...")
    ref_A = _find_reference_orbit(A, D, 0.1, 0.6)  # 32 points on orbit A
    # Orbit B is the mirror: (x, y) -> (y, x)
    ref_B = ref_A[:, ::-1].copy()

    x_range = np.linspace(-1.0, 1.0, n_grid)
    y_range = np.linspace(-1.0, 1.0, n_grid)
    basin = np.zeros((n_grid, n_grid), dtype=np.int8)

    for j, y0 in enumerate(y_range):
        # Vectorize across all x values for this row
        x = x_range.copy()
        y = np.full(n_grid, y0)

        # Transient
        for _ in range(n_transient):
            x_new = 1.0 - A * x * x + D * (y - x)
            y_new = 1.0 - A * y * y + D * (x - y)
            x, y = x_new, y_new
            diverged = (np.abs(x) > 100) | (np.abs(y) > 100)
            x = np.where(diverged, np.nan, x)
            y = np.where(diverged, np.nan, y)

        # Classify by minimum distance to either reference orbit.
        # For each grid point, compute distance to all 32 points on
        # each reference orbit and take the minimum.
        dist_A = np.full(n_grid, np.inf)
        dist_B = np.full(n_grid, np.inf)
        for k in range(32):
            d_a = (x - ref_A[k, 0])**2 + (y - ref_A[k, 1])**2
            d_b = (x - ref_B[k, 0])**2 + (y - ref_B[k, 1])**2
            dist_A = np.minimum(dist_A, d_a)
            dist_B = np.minimum(dist_B, d_b)

        basin[j] = np.where(np.isnan(x), -1,
                   np.where(dist_A < dist_B, 1,
                   np.where(dist_B < dist_A, 2, 0))).astype(np.int8)

        if (j + 1) % 200 == 0:
            print(f"  Basins: row {j + 1}/{n_grid}")

    np.savez_compressed(BASIN_NPZ, x=x_range, y=y_range, basin=basin,
                        A=np.array([A]), D=np.array([D]))
    print(f"Saved {BASIN_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_diagram(data):
    """Plot phase diagram in (A, D) space."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        apply_axes_polish,
        figure_spec,
        setup,
    )
    setup()

    A = data["A"]
    D = data["D"]
    var = data["var"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    # Log-scale variance for better contrast
    log_var = np.log10(np.where(var > 1e-12, var, 1e-12))
    im = ax.pcolormesh(A, D, log_var, cmap=CMAP_SEQUENTIAL,
                       rasterized=True)
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title("Coupled logistic map: $\\log_{10}$(temporal variance)", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\,\sigma^2_x$")
    cb.ax.tick_params(labelsize=spec.tick_size)

    fig.savefig(PHASE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PHASE_PNG}")


def plot_attractors(data):
    """Plot attractor portraits."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    A_values = data["A_values"]
    labels = ["2T", "4T", "8T", "8T (zoom)", "8C", "4C"]
    panel_labels = list("abcdef")

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 3, figsize=(spec.figsize[0], spec.figsize[1] - 0.4))
    fig.subplots_adjust(wspace=0.32, hspace=0.4)
    axes_flat = axes.flatten()

    for idx, A in enumerate(A_values):
        ax = axes_flat[idx]
        x = data[f"A_{A}_x"]
        y = data[f"A_{A}_y"]
        ax.scatter(x, y, s=0.01, c=COLORS["black"], alpha=0.3, rasterized=True)
        ax.set_title(f"({panel_labels[idx]}) $a = {A}$ ({labels[idx]})", loc="left", usetex=False)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        apply_axes_polish(ax, kind="grid", title_loc="left")

    fig.suptitle(
        f"Coupled logistic map, $\\varepsilon = {data['D'][0]}$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )
    fig.savefig(ATTR_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ATTR_PNG}")


def plot_basins(data):
    """Plot basin of attraction with zoom panel near y=x."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    x = data["x"]
    y = data["y"]
    basin = data["basin"]
    A_val = data["A"][0]

    # -1=diverged(white), 0=undetermined(grey), 1=orbit A(blue), 2=orbit B(red)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
        COLORS["offwhite"],
        COLORS["grey"],
        COLORS["blue"],
        COLORS["red"],
    ])

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=spec.figsize)

    # (a) Full view
    ax1.pcolormesh(x, y, basin, cmap=cmap, vmin=-1, vmax=2, rasterized=True)
    ax1.plot(x, x, color=COLORS["black"], linestyle="--", lw=0.3, alpha=0.4)
    ax1.set_xlabel("$x_0$")
    ax1.set_ylabel("$y_0$")
    ax1.set_title(f"(a) Full basin, $a={A_val:.5f}$", loc="left")
    ax1.set_aspect("equal")
    apply_axes_polish(ax1, kind="double", title_loc="left")

    # Draw zoom box
    zx0, zx1, zy0, zy1 = -0.16, 0.04, -0.11, 0.09
    rect = plt.Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        lw=0.8,
        ec=COLORS["black"],
        fc="none",
    )
    ax1.add_patch(rect)

    # (b) Zoomed view near y=x showing stripe structure
    ax2.pcolormesh(x, y, basin, cmap=cmap, vmin=-1, vmax=2, rasterized=True)
    ax2.plot(
        [-0.2, 0.1],
        [-0.2, 0.1],
        color=COLORS["black"],
        linestyle="--",
        lw=0.3,
        alpha=0.4,
    )
    ax2.set_xlim(zx0, zx1)
    ax2.set_ylim(zy0, zy1)
    ax2.set_xlabel("$x_0$")
    ax2.set_ylabel("$y_0$")
    ax2.set_title("(b) Zoom: stripe structure near $y = x$", loc="left")
    ax2.set_aspect("equal")
    apply_axes_polish(ax2, kind="double", title_loc="left")

    # Legend for basin colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["blue"], label="Orbit A"),
        Patch(facecolor=COLORS["red"], label="Orbit B"),
        Patch(facecolor=COLORS["offwhite"], edgecolor=COLORS["grey"], label="Diverged"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=spec.legend_size, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(BASIN_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {BASIN_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def compute_animation_data():
    """Sweep A from 0.9 to 1.45 at D=0.1 for animation."""
    from dynachaos.utils.animation import compute_animation_sweep

    D = 0.1
    A_sweep = np.linspace(0.9, 1.45, 200)

    def iterate_fn(A):
        state = np.array([0.1, 0.2])
        for _ in range(20_000):
            state = coupled_logistic(state, A, D)
        traj = np.empty((5_000, 2))
        for i in range(5_000):
            state = coupled_logistic(state, A, D)
            traj[i] = state
        return traj

    compute_animation_sweep(iterate_fn, A_sweep, ANIM_NPZ, n_plot=5_000)


def make_animation_gif(data):
    """Create GIF of attractor evolution across A."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"], data["all_x"], data["all_y"], ANIM_GIF,
        title_template=r"Coupled logistic map, $\varepsilon = 0.1$, $a = {param_value}$",
        param_name="a", param_fmt=".3f",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Phase diagram
    try:
        phase_data = _safe_load(PHASE_NPZ)
        print(f"Loaded {PHASE_NPZ}")
    except FileNotFoundError:
        print("Computing phase diagram...")
        compute_phase_diagram()
        phase_data = _safe_load(PHASE_NPZ)
    plot_phase_diagram(phase_data)

    # Attractors
    try:
        attr_data = _safe_load(ATTR_NPZ)
        print(f"Loaded {ATTR_NPZ}")
    except FileNotFoundError:
        print("Computing attractors...")
        compute_attractors()
        attr_data = _safe_load(ATTR_NPZ)
    plot_attractors(attr_data)

    # Basins
    try:
        basin_data = _safe_load(BASIN_NPZ)
        print(f"Loaded {BASIN_NPZ}")
    except FileNotFoundError:
        print("Computing basins...")
        compute_basins()
        basin_data = _safe_load(BASIN_NPZ)
    plot_basins(basin_data)

    # Animation
    try:
        anim_data = _safe_load(ANIM_NPZ)
        print(f"Loaded {ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing animation data...")
        compute_animation_data()
        anim_data = _safe_load(ANIM_NPZ)
    make_animation_gif(anim_data)


if __name__ == "__main__":
    main()
