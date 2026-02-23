#!/usr/bin/env python3
"""
compare_all: Apply modern diagnostics to Kaneko's maps (Section 11).

Applies all four modern diagnostic tools to representative Kaneko maps:
  1. 0-1 test for chaos
  2. SALI (Smaller Alignment Index)
  3. Permutation entropy and complexity-entropy plane
  4. Recurrence Quantification Analysis

Maps tested:
  - Logistic map (1D): periodic vs chaotic regimes
  - Delayed logistic (2D): torus -> fractalization -> chaos

OUTPUTS: figures/sec11_diagnostics/*.npz, *.png
USAGE:   python src/dynachaos/diagnostics/compare_all.py
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec11_diagnostics")

TEST01_NPZ = FIG_DIR / "test01_sweep.npz"
TEST01_PNG = FIG_DIR / "test01_sweep.png"
SALI_NPZ = FIG_DIR / "sali_comparison.npz"
SALI_PNG = FIG_DIR / "sali_comparison.png"
PE_NPZ = FIG_DIR / "permutation_entropy.npz"
PE_PNG = FIG_DIR / "permutation_entropy.png"
CH_NPZ = FIG_DIR / "complexity_entropy_plane.npz"
CH_PNG = FIG_DIR / "complexity_entropy_plane.png"
RQA_NPZ = FIG_DIR / "rqa_measures.npz"
RQA_PNG = FIG_DIR / "rqa_measures.png"


def _safe_load(path):
    """Load .npz safely."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _logistic_series(a, n_transient=5000, n_record=10_000):
    """Scalar time series from the logistic map f(x) = 1 - a x^2."""
    x = 0.1
    for _ in range(n_transient):
        x = 1.0 - a * x * x
    series = np.empty(n_record)
    for i in range(n_record):
        x = 1.0 - a * x * x
        series[i] = x
    return series


def _delayed_logistic_series(D, A=0.3, n_transient=10_000, n_record=10_000):
    """Scalar time series (x component) from the delayed logistic map."""
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
    x, y = fp + 0.01, fp - 0.01
    for _ in range(n_transient):
        x_new = A * x + (1.0 - A) * (1.0 - D * y * y)
        y = x
        x = x_new
    series = np.empty(n_record)
    for i in range(n_record):
        x_new = A * x + (1.0 - A) * (1.0 - D * y * y)
        y = x
        x = x_new
        series[i] = x
    return series


def _delayed_logistic_map(state, A, D):
    """Delayed logistic map for SALI computation."""
    x, y = state
    return np.array([A * x + (1.0 - A) * (1.0 - D * y * y), x])


def _delayed_logistic_jac(state, A, D):
    """Jacobian of the delayed logistic map."""
    x, y = state
    return np.array([[A, -2.0 * (1.0 - A) * D * y], [1.0, 0.0]])


def _coupled_delayed_map(state, A, D1, D2, eps):
    """4D coupled delayed logistic map for SALI (enough dimensions for
    SALI to distinguish torus from chaos)."""
    x, y, z, w = state
    h1 = z - w
    h2 = y - x
    x_new = A * x + D1 * y * (1.0 - y) + eps * h1
    z_new = A * z + D2 * w * (1.0 - w) + eps * h2
    return np.array([x_new, x, z_new, z])


def _coupled_delayed_jac(state, A, D1, D2, eps):
    """Jacobian of the 4D coupled delayed logistic map."""
    x, y, z, w = state
    return np.array([
        [A, D1 * (1.0 - 2.0 * y), eps, -eps],
        [1.0, 0.0, 0.0, 0.0],
        [-eps, eps, A, D2 * (1.0 - 2.0 * w)],
        [0.0, 0.0, 1.0, 0.0]
    ])


# ---------------------------------------------------------------------------
# 0-1 test sweep
# ---------------------------------------------------------------------------

def compute_01_test():
    """0-1 test for chaos across the logistic map parameter range."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.zero_one_test import zero_one_statistic

    n_a = 500
    a_values = np.linspace(1.0, 2.0, n_a)
    K_values = np.empty(n_a)

    for i, a in enumerate(a_values):
        series = _logistic_series(a, n_transient=5000, n_record=5000)
        K_values[i] = zero_one_statistic(series, n_c=50)
        if (i + 1) % 100 == 0:
            print(f"  0-1 test: {i + 1}/{n_a}")
            np.savez_compressed(TEST01_NPZ, a=a_values[:i+1], K=K_values[:i+1])

    np.savez_compressed(TEST01_NPZ, a=a_values, K=K_values)
    print(f"Saved {TEST01_NPZ}")


# ---------------------------------------------------------------------------
# SALI comparison
# ---------------------------------------------------------------------------

def compute_sali():
    """SALI time series for coupled delayed logistic (4D) at various regimes.

    Uses the 4D coupled delayed logistic map so that SALI has enough
    dimensions to distinguish torus from chaos — in 2D, both deviation
    vectors must align regardless of dynamics (only the decay rate differs).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.sali_gali import sali

    A = 0.4
    eps = 5e-3
    cases = [
        (2.35, "3-torus"),
        (2.37, "locking"),
        (2.47, "onset of chaos"),
        (2.55, "developed chaos"),
    ]

    results = {}
    for D2, label in cases:
        D1 = D2 + 0.1
        print(f"  SALI: D2={D2} ({label})")
        x0 = np.array([0.5, 0.5, 0.3, 0.3])
        f = lambda s, _D1=D1, _D2=D2: _coupled_delayed_map(s, A, _D1, _D2, eps)
        jac = lambda s, _D1=D1, _D2=D2: _coupled_delayed_jac(s, A, _D1, _D2, eps)
        s = sali(f, jac, x0, n_iter=10_000, n_transient=5000)
        results[f"D2_{D2}_sali"] = s

    results["D2_values"] = np.array([c[0] for c in cases])
    np.savez_compressed(SALI_NPZ, **results)
    print(f"Saved {SALI_NPZ}")


# ---------------------------------------------------------------------------
# Permutation entropy sweep
# ---------------------------------------------------------------------------

def compute_permutation_entropy():
    """Permutation entropy across logistic map and delayed logistic."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.permutation import permutation_entropy

    # Logistic map sweep
    n_a = 500
    a_values = np.linspace(1.0, 2.0, n_a)
    H_logistic = np.empty(n_a)
    for i, a in enumerate(a_values):
        series = _logistic_series(a, n_transient=5000, n_record=5000)
        H_logistic[i] = permutation_entropy(series, d=5)
        if (i + 1) % 100 == 0:
            print(f"  PE logistic: {i + 1}/{n_a}")

    # Delayed logistic sweep
    n_D = 300
    D_values = np.linspace(1.5, 2.2, n_D)
    H_delayed = np.empty(n_D)
    for i, D in enumerate(D_values):
        series = _delayed_logistic_series(D, n_transient=5000, n_record=5000)
        H_delayed[i] = permutation_entropy(series, d=5)
        if (i + 1) % 100 == 0:
            print(f"  PE delayed: {i + 1}/{n_D}")

    np.savez_compressed(PE_NPZ, a=a_values, H_logistic=H_logistic,
                        D=D_values, H_delayed=H_delayed)
    print(f"Saved {PE_NPZ}")


# ---------------------------------------------------------------------------
# Complexity-entropy plane
# ---------------------------------------------------------------------------

def compute_complexity_entropy():
    """Map different dynamical regimes onto the C-H plane."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.permutation import complexity_entropy

    # Sample many parameter values for the logistic map
    a_values = np.linspace(1.0, 2.0, 200)
    H_vals = np.empty(len(a_values))
    C_vals = np.empty(len(a_values))

    for i, a in enumerate(a_values):
        series = _logistic_series(a, n_transient=5000, n_record=5000)
        H_vals[i], C_vals[i] = complexity_entropy(series, d=5)
        if (i + 1) % 50 == 0:
            print(f"  C-H logistic: {i + 1}/{len(a_values)}")

    # Delayed logistic
    D_values = np.linspace(1.5, 2.2, 200)
    H_del = np.empty(len(D_values))
    C_del = np.empty(len(D_values))

    for i, D in enumerate(D_values):
        series = _delayed_logistic_series(D, n_transient=5000, n_record=5000)
        H_del[i], C_del[i] = complexity_entropy(series, d=5)
        if (i + 1) % 50 == 0:
            print(f"  C-H delayed: {i + 1}/{len(D_values)}")

    np.savez_compressed(CH_NPZ, a=a_values, H_logistic=H_vals,
                        C_logistic=C_vals, D=D_values,
                        H_delayed=H_del, C_delayed=C_del)
    print(f"Saved {CH_NPZ}")


# ---------------------------------------------------------------------------
# RQA measures vs parameter
# ---------------------------------------------------------------------------

def compute_rqa():
    """RQA measures along the delayed logistic transition."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa

    A = 0.3
    n_D = 80
    D_values = np.linspace(1.5, 2.2, n_D)

    RR = np.empty(n_D)
    DET = np.empty(n_D)
    LAM = np.empty(n_D)
    ENTR = np.empty(n_D)

    for i, D in enumerate(D_values):
        fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
        x, y = fp + 0.01, fp - 0.01
        for _ in range(10_000):
            x_new = A * x + (1.0 - A) * (1.0 - D * y * y)
            y = x
            x = x_new

        traj = np.empty((2000, 2))
        for t in range(2000):
            x_new = A * x + (1.0 - A) * (1.0 - D * y * y)
            y = x
            x = x_new
            traj[t] = [x, y]

        R, _ = recurrence_matrix(traj, percentile=5)
        stats = rqa(R, l_min=2, v_min=2)
        RR[i] = stats["RR"]
        DET[i] = stats["DET"]
        LAM[i] = stats["LAM"]
        ENTR[i] = stats["ENTR"]

        if (i + 1) % 20 == 0:
            print(f"  RQA: {i + 1}/{n_D}")
            np.savez_compressed(RQA_NPZ, D=D_values[:i+1],
                                RR=RR[:i+1], DET=DET[:i+1],
                                LAM=LAM[:i+1], ENTR=ENTR[:i+1])

    np.savez_compressed(RQA_NPZ, D=D_values, RR=RR, DET=DET,
                        LAM=LAM, ENTR=ENTR)
    print(f"Saved {RQA_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_01_test(data):
    """Plot 0-1 test K vs a for the logistic map."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.plot(data["a"], data["K"], ".", ms=0.5, rasterized=True, color=COLORS["black"])
    ax.axhline(0.5, color=COLORS["red"], lw=0.7, ls="--", alpha=0.55)
    ax.set_xlabel(r"Nonlinearity $a$")
    ax.set_ylabel(r"$K_{01}$ (0-1 test)")
    ax.set_title(r"0-1 test for chaos: logistic map $f(x) = 1 - ax^2$", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left")
    ax.set_ylim(-0.1, 1.1)
    ax.text(
        1.5,
        0.6,
        "$K_{01} \\approx 1$: chaos",
        fontsize=spec.tick_size,
        color=COLORS["red"],
    )
    ax.text(
        1.1,
        0.15,
        "$K_{01} \\approx 0$: regular",
        fontsize=spec.tick_size,
        color=COLORS["blue"],
    )

    fig.savefig(TEST01_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {TEST01_PNG}")


def plot_sali(data):
    """Plot SALI time series for different dynamical regimes."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        apply_axes_polish,
        COLORS,
        color_for,
        figure_spec,
        finalize_legend,
        marker_for,
        setup,
    )
    setup()

    D2_values = data["D2_values"]
    labels = ["3-torus", "locking",
              "onset of chaos", "developed chaos"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    for idx, D2 in enumerate(D2_values):
        s = data[f"D2_{D2}_sali"]
        # Avoid log(0)
        s_safe = np.where(s > 1e-16, s, 1e-16)
        ax.semilogy(
            np.arange(len(s)),
            s_safe,
            lw=1.1,
            color=color_for(idx),
            marker=marker_for(idx),
            markersize=3.0,
            markevery=max(len(s) // 14, 1),
            markerfacecolor=COLORS["offwhite"],
            markeredgewidth=0.8,
            label=f"$D_2={D2}$ ({labels[idx]})",
        )

    ax.set_xlabel("Iteration $n$")
    ax.set_ylabel("SALI")
    ax.set_title(
        r"SALI: coupled delayed logistic map, $\alpha = 0.4$, "
        r"$\varepsilon = 5 \times 10^{-3}$", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left")
    ax.set_ylim(1e-16, 10)
    finalize_legend(ax, kind="double", loc="lower left")

    fig.savefig(SALI_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SALI_PNG}")


def plot_permutation_entropy(data):
    """Plot permutation entropy sweeps."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=spec.figsize)

    ax1.plot(data["a"], data["H_logistic"], ".", ms=0.5, rasterized=True, color=COLORS["black"])
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel(r"$H_\mathrm{PE}$")
    ax1.set_title(r"Logistic map $1-ax^2$", loc="left")
    ax1.set_ylim(-0.05, 1.05)
    apply_axes_polish(ax1, kind="double", title_loc="left")

    ax2.plot(data["D"], data["H_delayed"], ".", ms=0.5, rasterized=True, color=COLORS["black"])
    ax2.set_xlabel(r"$D$")
    ax2.set_ylabel(r"$H_\mathrm{PE}$")
    ax2.set_title("Delayed logistic, $\\alpha=0.3$", loc="left")
    ax2.set_ylim(-0.05, 1.05)
    apply_axes_polish(ax2, kind="double", title_loc="left")

    fig.suptitle("Permutation entropy ($d=5$)", x=0.01, ha="left", y=1.02, fontsize=spec.title_size)
    fig.savefig(PE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PE_PNG}")


def plot_complexity_entropy(data):
    """Plot the complexity-entropy plane."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
    )
    setup()

    spec = figure_spec("single")
    fig, ax = plt.subplots(figsize=(spec.figsize[0] * 1.25, spec.figsize[1] + 0.35))

    ax.scatter(data["H_logistic"], data["C_logistic"], s=3, alpha=0.6,
               c=COLORS["blue"], label="Logistic map", rasterized=True)
    ax.scatter(data["H_delayed"], data["C_delayed"], s=3, alpha=0.6,
               c=COLORS["red"], label="Delayed logistic", rasterized=True)

    ax.set_xlabel(r"Normalised permutation entropy $H$")
    ax.set_ylabel(r"Statistical complexity $C$")
    ax.set_title("Complexity-entropy plane ($d=5$)", loc="left")
    apply_axes_polish(ax, kind="single", title_loc="left")
    ax.set_xlim(-0.05, 1.05)
    finalize_legend(ax, kind="single", markerscale=2.6, loc="upper left")

    fig.savefig(CH_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CH_PNG}")


def plot_rqa(data):
    """Plot RQA measures vs D for the delayed logistic."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup
    setup()

    D = data["D"]

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=spec.figsize, sharex=True)

    axes[0, 0].plot(D, data["RR"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[0, 0].set_ylabel("RR")
    axes[0, 0].set_title("Recurrence rate", loc="left")
    apply_axes_polish(axes[0, 0], kind="grid", title_loc="left")

    axes[0, 1].plot(D, data["DET"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[0, 1].set_ylabel("DET")
    axes[0, 1].set_title("Determinism", loc="left")
    apply_axes_polish(axes[0, 1], kind="grid", title_loc="left")

    axes[1, 0].plot(D, data["LAM"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[1, 0].set_ylabel("LAM")
    axes[1, 0].set_xlabel(r"$D$")
    axes[1, 0].set_title("Laminarity", loc="left")
    apply_axes_polish(axes[1, 0], kind="grid", title_loc="left")

    axes[1, 1].plot(D, data["ENTR"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[1, 1].set_ylabel("ENTR")
    axes[1, 1].set_xlabel(r"$D$")
    axes[1, 1].set_title("Entropy of diag. lines", loc="left")
    apply_axes_polish(axes[1, 1], kind="grid", title_loc="left")

    fig.subplots_adjust(wspace=0.24, hspace=0.34)
    fig.suptitle(
        "RQA: delayed logistic map, $\\alpha = 0.3$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )
    fig.savefig(RQA_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RQA_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sections = [
        ("0-1 test", TEST01_NPZ, compute_01_test, plot_01_test),
        ("SALI", SALI_NPZ, compute_sali, plot_sali),
        ("Permutation entropy", PE_NPZ, compute_permutation_entropy,
         plot_permutation_entropy),
        ("C-H plane", CH_NPZ, compute_complexity_entropy,
         plot_complexity_entropy),
        ("RQA", RQA_NPZ, compute_rqa, plot_rqa),
    ]

    for name, npz_path, compute_fn, plot_fn in sections:
        try:
            data = _safe_load(npz_path)
            print(f"Loaded {npz_path}")
        except FileNotFoundError:
            print(f"Computing {name}...")
            compute_fn()
            data = _safe_load(npz_path)
        plot_fn(data)


if __name__ == "__main__":
    main()
