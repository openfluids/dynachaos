"""Shared analysis and plotting pipeline for benchmark examples.

Provides three functions:
- ``load_jsonc``: Load a JSONC file (JSON with // comments)
- ``run_embedding_analysis``: AMI -> Cao -> FNN -> embedding -> D2
- ``plot_benchmark``: 2x3 multi-panel figure with attractor, diagnostics, summary

Every benchmark script calls these with system-specific data and reference values.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynachaos.diagnostics import (
    average_mutual_information,
    cao_method,
    correlation_dimension,
    false_nearest_neighbors,
)
from dynachaos.diagnostics.recurrence import embed_time_delay
from dynachaos.utils.style import (
    COLORS,
    apply_axes_polish,
    figure_spec,
    finalize_legend,
    setup,
)


def load_jsonc(path):
    """Load a JSONC file (JSON with // comments)."""
    text = Path(path).read_text()
    text = re.sub(r'//.*', '', text)
    return json.loads(text)


def run_embedding_analysis(x, tau_max=100, d_max=15, n_r=40, verbose=False):
    """Run AMI -> Cao -> FNN -> embed -> D2 pipeline.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Scalar time series.
    tau_max : int
        Maximum delay for AMI scan.
    d_max : int
        Maximum embedding dimension for Cao/FNN.
    n_r : int
        Number of r values for correlation dimension.
    verbose : bool
        If True, print timing and memory info during D2 computation.

    Returns
    -------
    dict with keys:
        tau_opt, d_opt, taus, mi_values, E1, E2,
        f1, f2, f3, r_values, C_values, D2, embedded
    """
    # Ensure contiguous array (e.g. traj[:, 0] is a non-contiguous view)
    x = np.ascontiguousarray(x, dtype=np.float64)

    # AMI: find optimal delay
    taus, mi_values = average_mutual_information(x, tau_max=tau_max)

    # Find first local minimum of AMI
    tau_opt = 1
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            tau_opt = int(taus[i])
            break

    # Cao's method: find optimal dimension
    # cao_method takes d_max as int, returns E1/E2 of shape (d_max-1,)
    E1, E2 = cao_method(x, tau_opt, d_max=d_max)
    dims = np.arange(1, d_max)  # E1[i] = E(d=i+1)/E(d=i), so dims 1..d_max-1

    # FNN fractions: d_max as int, returns arrays of shape (d_max,)
    f1, f2, f3 = false_nearest_neighbors(x, tau_opt, d_max=d_max)
    fnn_dims = np.arange(1, d_max + 1)

    # Find d_opt: first d where E1 > 0.95
    d_opt = 2
    for i in range(len(E1)):
        if E1[i] > 0.95:
            d_opt = int(dims[i]) + 1
            break
    d_opt = max(d_opt, 2)

    # Embed and compute correlation dimension
    embedded = embed_time_delay(x, d_opt, tau_opt)
    D2, r_values, C_values, local_slopes, scaling_mask = correlation_dimension(
        embedded, n_r=n_r, theiler_window=tau_opt, verbose=verbose,
    )

    return {
        "tau_opt": tau_opt,
        "d_opt": d_opt,
        "taus": taus,
        "mi_values": mi_values,
        "dims": dims,
        "E1": E1,
        "E2": E2,
        "fnn_dims": fnn_dims,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "r_values": r_values,
        "C_values": C_values,
        "D2": D2,
        "local_slopes": local_slopes,
        "scaling_mask": scaling_mask,
        "embedded": embedded,
    }


def plot_benchmark(results, attractor_xy, output_png, system_name,
                   ref_D2=None, ref_lambda1=None, computed_lambda1=None,
                   computed_spectrum=None, ref_spectrum=None,
                   attractor_xlabel="$x$", attractor_ylabel="$y$",
                   attractor_scatter_kw=None):
    """Produce 2x3 multi-panel benchmark figure.

    Panels:
        (a) Attractor phase portrait
        (b) AMI I(tau) with tau_opt marked
        (c) Cao E1(d)/E2(d) with d_opt marked
        (d) FNN fractions f1, f2, f3 vs d
        (e) log C(r) vs log r with D2 slope
        (f) Summary text

    Parameters
    ----------
    results : dict
        Output of ``run_embedding_analysis``.
    attractor_xy : tuple of (ndarray, ndarray)
        (x, y) data for attractor scatter plot.
    output_png : Path
        Output filename.
    system_name : str
        System name for titles.
    ref_D2 : float or None
        Literature D2 value.
    ref_lambda1 : float or None
        Literature lambda_1 value.
    computed_lambda1 : float or None
        Computed lambda_1 value.
    computed_spectrum : ndarray or None
        Full computed Lyapunov spectrum.
    ref_spectrum : ndarray or None
        Literature Lyapunov spectrum.
    attractor_xlabel, attractor_ylabel : str
        Axis labels for attractor panel.
    attractor_scatter_kw : dict or None
        Extra kwargs for attractor scatter.
    """
    setup()
    spec = figure_spec("grid")

    fig, axes = plt.subplots(2, 3, figsize=(spec.figsize[0], spec.figsize[1] + 1.0))
    fig.subplots_adjust(hspace=0.50, wspace=0.38)

    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    # ── (a) Attractor ──
    scatter_kw = {"s": 0.05, "c": COLORS["black"], "alpha": 0.3, "rasterized": True}
    if attractor_scatter_kw:
        scatter_kw.update(attractor_scatter_kw)
    ax_a.scatter(*attractor_xy, **scatter_kw)
    ax_a.set_xlabel(attractor_xlabel)
    ax_a.set_ylabel(attractor_ylabel)
    ax_a.set_title("(a) Attractor", loc="left")
    ax_a.grid(False)
    apply_axes_polish(ax_a, kind="grid", title_loc="left")

    # ── (b) AMI ──
    ax_b.plot(results["taus"], results["mi_values"], color=COLORS["black"], lw=1.2)
    ax_b.axvline(results["tau_opt"], color=COLORS["red"], ls="--", lw=0.8,
                 label=rf"$\tau = {results['tau_opt']}$")
    ax_b.set_xlabel(r"$\tau$")
    ax_b.set_ylabel(r"$I(\tau)$")
    ax_b.set_title("(b) Average Mutual Information", loc="left")
    apply_axes_polish(ax_b, kind="grid", title_loc="left")
    finalize_legend(ax_b, kind="grid")

    # ── (c) Cao ──
    ax_c.plot(results["dims"], results["E1"], color=COLORS["black"], lw=1.2,
              marker="o", ms=3, label="$E_1(d)$")
    ax_c.plot(results["dims"], results["E2"], color=COLORS["blue"], lw=1.2,
              marker="s", ms=3, label="$E_2(d)$")
    ax_c.axhline(0.95, color=COLORS["grey"], ls=":", lw=0.6)
    ax_c.axvline(results["d_opt"], color=COLORS["red"], ls="--", lw=0.8,
                 label=rf"$d = {results['d_opt']}$")
    ax_c.set_xlabel("$d$")
    ax_c.set_ylabel("$E_1, E_2$")
    ax_c.set_title("(c) Cao's method", loc="left")
    apply_axes_polish(ax_c, kind="grid", title_loc="left")
    finalize_legend(ax_c, kind="grid", loc="center right")

    # ── (d) FNN ──
    ax_d.plot(results["fnn_dims"], results["f1"], color=COLORS["black"], lw=1.2,
              marker="o", ms=3, label="$f_1$")
    ax_d.plot(results["fnn_dims"], results["f2"], color=COLORS["blue"], lw=1.2,
              marker="s", ms=3, label="$f_2$")
    ax_d.plot(results["fnn_dims"], results["f3"], color=COLORS["red"], lw=1.2,
              marker="^", ms=3, label="$f_3$")
    ax_d.set_xlabel("$d$")
    ax_d.set_ylabel("FNN fraction")
    ax_d.set_title("(d) False nearest neighbors", loc="left")
    apply_axes_polish(ax_d, kind="grid", title_loc="left")
    finalize_legend(ax_d, kind="grid")

    # ── (e) Correlation dimension ──
    r_vals = results["r_values"]
    C_vals = results["C_values"]
    scaling = results.get("scaling_mask", np.zeros(len(r_vals), dtype=bool))
    mask = C_vals > 0
    if np.any(mask):
        # Plot all valid points in grey
        ax_e.plot(np.log(r_vals[mask]), np.log(C_vals[mask]),
                  color=COLORS["grey"], lw=0.8, marker="o", ms=2, alpha=0.5)
        # Highlight scaling region points
        scaling_valid = mask & scaling
        if np.any(scaling_valid):
            ax_e.plot(np.log(r_vals[scaling_valid]), np.log(C_vals[scaling_valid]),
                      color=COLORS["black"], lw=1.2, marker="o", ms=3)
            # Fit line over scaling region
            lr = np.log(r_vals[scaling_valid])
            lc = np.log(C_vals[scaling_valid])
            coeffs = np.polyfit(lr, lc, 1)
            ax_e.plot(lr, np.polyval(coeffs, lr), color=COLORS["red"],
                      ls="--", lw=0.8, label=rf"$D_2 = {results['D2']:.2f}$")
    ax_e.set_xlabel(r"$\ln\, r$")
    ax_e.set_ylabel(r"$\ln\, C(r)$")
    ax_e.set_title("(e) Correlation integral", loc="left")
    apply_axes_polish(ax_e, kind="grid", title_loc="left")
    finalize_legend(ax_e, kind="grid")

    # ── (f) Summary ──
    ax_f.axis("off")
    lines = [f"System: {system_name}", ""]

    # Embedding parameters
    lines.append(f"Optimal delay: tau = {results['tau_opt']}")
    lines.append(f"Embedding dim: d = {results['d_opt']}")
    lines.append("")

    # Correlation dimension
    lines.append(f"D2 (computed): {results['D2']:.3f}")
    if ref_D2 is not None:
        lines.append(f"D2 (literature): {ref_D2}")
    lines.append("")

    # Lyapunov
    if computed_lambda1 is not None:
        lines.append(f"lambda_1 (computed): {computed_lambda1:.4f}")
    if ref_lambda1 is not None:
        lines.append(f"lambda_1 (literature): {ref_lambda1}")

    if computed_spectrum is not None:
        spec_str = ", ".join(f"{v:.4f}" for v in computed_spectrum)
        lines.append(f"Spectrum: [{spec_str}]")
    if ref_spectrum is not None:
        ref_str = ", ".join(f"{v}" for v in ref_spectrum)
        lines.append(f"Ref spectrum: [{ref_str}]")

    summary_text = "\n".join(lines)
    ax_f.text(0.05, 0.95, summary_text, transform=ax_f.transAxes,
              fontsize=spec.tick_size, verticalalignment="top",
              fontfamily="monospace")
    ax_f.set_title("(f) Summary", loc="left")
    apply_axes_polish(ax_f, kind="grid", title_loc="left")

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
