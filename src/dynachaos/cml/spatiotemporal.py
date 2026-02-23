#!/usr/bin/env python3
"""
spatiotemporal: Spatiotemporal intermittency in coupled map lattices.

Reproduces Kaneko (1985) "Spatiotemporal Intermittency in Coupled Map Lattices",
PTP 74(5), 1033-1044.

CML model:
    x_{n+1}(i) = f(x_n(i)) + eps/2 [g(x_n(i+1)) + g(x_n(i-1)) - 2 g(x_n(i))]

Three models studied:
  (A) Piecewise map: f(x) = x + x^2 + a for x < c; -3(x-c)+1+a for x > c
      with c = (sqrt(5)-1)/2, g(x) = f(x)
  (B) Coupled circle: f(x) = x + A sin(2 pi x) + C (mod 1), g(x) = sin(2 pi x)
      A=0.2, C=0.55
  (C) Coupled logistic: f(x) = 1 - A x^2, g(x) = f(x)
      A = 1.752

Figures:
  - Spacetime diagrams for each model

OUTPUTS: figures/sec08_sti/spacetime_diagrams.npz, .png
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec08_sti")

STI_NPZ = FIG_DIR / "spacetime_diagrams.npz"
STI_PNG = FIG_DIR / "spacetime_diagrams.png"


def _safe_load(path):
    """Load .npz safely (no deserialization of arbitrary objects)."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def model_A_f(x, a=0.02):
    """Piecewise map for model (A).

    For a > 0 the laminar branch (x < c) has no stable fixed point, so
    the local dynamics is intermittent (Pomeau--Manneville type I).
    Kaneko (1985) uses a slightly positive a to generate spatiotemporal
    intermittency.
    """
    c = (np.sqrt(5) - 1.0) / 2.0
    return np.where(x < c, x + x * x + a, -3.0 * (x - c) + 1.0 + a)


def model_B_f(x, A=0.2, C=0.55):
    """Circle map for model (B)."""
    return (x + A * np.sin(2 * np.pi * x) + C) % 1.0


def model_B_g(x):
    """Coupling function for model (B)."""
    return np.sin(2 * np.pi * x)


def model_C_f(x, A=1.752):
    """Logistic map for model (C)."""
    return 1.0 - A * x * x


# ---------------------------------------------------------------------------
# CML iteration
# ---------------------------------------------------------------------------

def cml_step(x, f, g, eps):
    """One CML step with periodic boundary conditions."""
    fx = f(x)
    gx = g(x)
    gx_left = np.roll(gx, -1)
    gx_right = np.roll(gx, 1)
    coupling = eps / 2.0 * (gx_left + gx_right - 2.0 * gx)
    return fx + coupling


def simulate_cml(f, g, eps, N=200, n_transient=2000, n_record=500,
                 x0=None):
    """Run CML and record spacetime diagram."""
    if x0 is None:
        rng = np.random.default_rng(42)
        x0 = rng.uniform(0, 1, N)

    x = x0.copy()
    for _ in range(n_transient):
        x = cml_step(x, f, g, eps)

    spacetime = np.empty((n_record, N))
    for t in range(n_record):
        x = cml_step(x, f, g, eps)
        spacetime[t] = x

    return spacetime


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute():
    """Generate spacetime diagrams for all three models."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    N = 200
    n_record = 500

    results = {}

    print("  Model (A)...")
    for eps in [0.06, 0.07, 0.08]:
        st = simulate_cml(model_A_f, model_A_f, eps, N=N, n_record=n_record)
        results[f"A_eps_{eps}"] = st

    print("  Model (B)...")
    for eps in [0.02, 0.024, 0.03]:
        st = simulate_cml(model_B_f, model_B_g, eps, N=N, n_record=n_record)
        results[f"B_eps_{eps}"] = st

    print("  Model (C)...")
    for eps in [0.16, 0.20, 0.30]:
        st = simulate_cml(model_C_f, model_C_f, eps, N=N, n_record=n_record)
        results[f"C_eps_{eps}"] = st

    np.savez_compressed(STI_NPZ, **results)
    print(f"Saved {STI_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data):
    """Plot spacetime diagrams for all three models."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import setup, DOUBLE_COL, CMAP_SPACETIME
    setup()

    configs = [
        ("A", [0.06, 0.07, 0.08], "Model (A): piecewise"),
        ("B", [0.02, 0.024, 0.03], "Model (B): circle"),
        ("C", [0.16, 0.20, 0.30], "Model (C): logistic"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 7.0))

    for row, (model, eps_vals, title) in enumerate(configs):
        for col, eps in enumerate(eps_vals):
            ax = axes[row, col]
            key = f"{model}_eps_{eps}"
            if key in data:
                st = data[key]
                # Per-panel normalization to handle different value ranges
                vmin, vmax = np.nanmin(st), np.nanmax(st)
                if vmax - vmin < 1e-12:
                    vmin, vmax = vmin - 0.5, vmax + 0.5
                ax.imshow(st, aspect="auto", cmap=CMAP_SPACETIME,
                          origin="lower", interpolation="nearest",
                          vmin=vmin, vmax=vmax)
            ax.set_title(rf"$\varepsilon = {eps}$", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{title}\nTime $n$", fontsize=7)
            else:
                ax.set_ylabel("$n$", fontsize=7)
            ax.set_xlabel("Site $i$", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Spatiotemporal intermittency in CML", fontsize=11, y=1.01)
    fig.savefig(STI_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {STI_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = _safe_load(STI_NPZ)
        print(f"Loaded {STI_NPZ}")
    except FileNotFoundError:
        print("Computing spacetime diagrams...")
        compute()
        data = _safe_load(STI_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
