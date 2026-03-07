#!/usr/bin/env python3
"""
correlation_figure: Spatial correlation decay and Lyapunov density convergence.

Two-panel figure for Section 8 (spatiotemporal intermittency):
  (a) Spatial autocorrelation |C(r)|/C(0) vs separation r (semilogy)
  (b) Maximal Lyapunov exponent density lambda_max/L vs subsystem size L

CML model: logistic f(x) = 1 - a*x^2, g = f, periodic BC, N=200, eps=0.3.
Four regimes:  a=1.5 (frozen random), a=1.7 (pattern selection),
               a=1.85 (defect turbulence), a=1.95 (fully-developed turbulence).

OUTPUTS: figures/sec08_sti/correlation_decay.npz, correlation_decay.png
USAGE:   python src/dynachaos/cml/correlation_figure.py
"""

from dynachaos.io.paths import safe_load, section_dir
import numpy as np

from dynachaos.cml.primitives import (
    cml_jacobian_subblock_logistic as _cml_jacobian_subblock,
    cml_step_logistic as cml_step,
)

FIG_DIR = section_dir("sec08_sti")
CORR_NPZ = FIG_DIR / "correlation_decay.npz"
CORR_PNG = FIG_DIR / "correlation_decay.png"


# ---------------------------------------------------------------------------
# Panel 1: Spatial correlations
# ---------------------------------------------------------------------------

def compute_correlations():
    """Compute spatial autocorrelation C(r) for multiple a values.

    C(r) = <x(i) x(i+r)> - <x(i)> <x(i+r)>, averaged over sites and time.
    Normalized so C(0) = 1.
    """
    N = 200
    eps = 0.3
    a_values = np.array([1.5, 1.7, 1.85, 1.95])
    n_transient = 5000
    n_sample = 1000
    sample_interval = 10
    rng = np.random.default_rng(42)

    r_max = N // 2
    all_corr = np.empty((len(a_values), r_max + 1))

    for ia, a in enumerate(a_values):
        print(f"  Correlation: a={a}")
        x = rng.uniform(-0.5, 0.5, N)

        # Transient
        for _ in range(n_transient):
            x = cml_step(x, a, eps)

        # Collect decorrelated snapshots
        snapshots = np.empty((n_sample, N))
        for s in range(n_sample):
            for _ in range(sample_interval):
                x = cml_step(x, a, eps)
            snapshots[s] = x

        # Compute spatial autocorrelation via FFT (much faster than direct)
        # C(r) = <x(i) x(i+r)> - <x>^2, averaged over time
        mean_x = snapshots.mean()
        fluct = snapshots - mean_x
        corr_sum = np.zeros(r_max + 1)

        for s in range(n_sample):
            # Full circular correlation via FFT
            fft_f = np.fft.rfft(fluct[s])
            power = np.real(fft_f * np.conj(fft_f))
            full_corr = np.fft.irfft(power, n=N) / N
            corr_sum += full_corr[:r_max + 1]

        corr_sum /= n_sample
        # Normalize: C(r)/C(0) so C(0) = 1
        if abs(corr_sum[0]) > 1e-15:
            all_corr[ia] = corr_sum / corr_sum[0]
        else:
            all_corr[ia] = corr_sum

    return a_values, np.arange(r_max + 1), all_corr


def compute_lyapunov_density():
    """Compute subsystem maximal Lyapunov exponent density for multiple a, L.

    For each a value and subsystem size L, the full N=200 CML is evolved, but
    a single perturbation vector of length L (for sites 0..L-1) is propagated
    using the local Jacobian.  The resulting maximal exponent, divided by L,
    gives the Lyapunov density.
    """
    N = 200
    eps = 0.3
    a_values = np.array([1.5, 1.7, 1.85, 1.95])
    L_values = np.array([10, 20, 40, 60, 80, 100, 150, 200])
    n_transient = 5000
    n_iter = 20000
    rng = np.random.default_rng(42)

    density = np.empty((len(a_values), len(L_values)))

    for ia, a in enumerate(a_values):
        print(f"  Lyapunov density: a={a}")
        x = rng.uniform(-0.5, 0.5, N)

        # Transient on full lattice
        for _ in range(n_transient):
            x = cml_step(x, a, eps)

        for iL, L in enumerate(L_values):
            # Reset trajectory from the post-transient state
            x_run = x.copy()

            # Random unit tangent vector of length L
            v = rng.standard_normal(L)
            v /= np.linalg.norm(v)

            log_sum = 0.0
            for _ in range(n_iter):
                # Build subsystem Jacobian and propagate
                J_sub = _cml_jacobian_subblock(x_run, a, eps, L)
                v = J_sub @ v
                norm_v = np.linalg.norm(v)
                if norm_v > 0:
                    log_sum += np.log(norm_v)
                    v /= norm_v
                else:
                    log_sum += -100.0
                    v = rng.standard_normal(L)
                    v /= np.linalg.norm(v)
                # Evolve full lattice
                x_run = cml_step(x_run, a, eps)

            lam_max = log_sum / n_iter
            density[ia, iL] = lam_max / L

    return a_values, L_values, density


# ---------------------------------------------------------------------------
# Exponential fit for correlation length
# ---------------------------------------------------------------------------

def _fit_correlation_length(r, corr_normalized):
    """Fit |C(r)| ~ exp(-r/xi) to estimate correlation length xi.

    Uses a linear fit to log|C(r)| for r > 0 where |C(r)| > threshold.
    """
    abs_corr = np.abs(corr_normalized)
    # Use r >= 1 and |C| above noise floor
    mask = (r >= 1) & (abs_corr > 1e-6)
    if mask.sum() < 3:
        return np.inf

    r_fit = r[mask].astype(float)
    log_c = np.log(abs_corr[mask])

    # Linear regression: log|C| = -r/xi + const
    coeffs = np.polyfit(r_fit, log_c, 1)
    slope = coeffs[0]
    if slope >= 0:
        return np.inf
    return -1.0 / slope


# ---------------------------------------------------------------------------
# Combined computation
# ---------------------------------------------------------------------------

def compute():
    """Run both computations and save to a single .npz."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing spatial correlations...")
    a_corr, r_vals, all_corr = compute_correlations()

    print("Computing Lyapunov density convergence...")
    a_lyap, L_vals, density = compute_lyapunov_density()

    # Fit correlation lengths
    xi_values = np.array([
        _fit_correlation_length(r_vals, all_corr[ia])
        for ia in range(len(a_corr))
    ])

    np.savez_compressed(
        CORR_NPZ,
        a_corr=a_corr,
        r_vals=r_vals,
        all_corr=all_corr,
        xi_values=xi_values,
        a_lyap=a_lyap,
        L_vals=L_vals,
        density=density,
    )
    print(f"Saved {CORR_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data):
    """Two-panel figure: correlation decay + Lyapunov density."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        apply_axes_polish,
        figure_spec,
        series_style,
        setup,
    )
    setup()

    a_corr = data["a_corr"]
    r_vals = data["r_vals"]
    all_corr = data["all_corr"]
    xi_values = data["xi_values"]
    a_lyap = data["a_lyap"]
    L_vals = data["L_vals"]
    density = data["density"]

    labels = [
        r"$a=1.50$",
        r"$a=1.70$",
        r"$a=1.85$",
        r"$a=1.95$",
    ]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(spec.figsize[0], spec.figsize[1] + 0.2),
        gridspec_kw={"width_ratios": (1.25, 0.95)},
    )

    # --- Panel (a): Spatial correlation decay ---
    for ia in range(len(a_corr)):
        sty = series_style(ia)
        sty.pop("marker", None)
        sty.pop("markersize", None)
        sty.pop("markerfacecolor", None)
        sty.pop("markeredgewidth", None)
        abs_corr = np.abs(all_corr[ia])
        mask = abs_corr > 0
        r_plot = r_vals[mask]
        c_plot = abs_corr[mask]
        xi = xi_values[ia]
        label = labels[ia]
        if np.isfinite(xi):
            label += rf", $\xi \approx {xi:.1f}$"
        ax1.semilogy(r_plot, c_plot, label=label, **sty)

    ax1.set_xlabel(r"Separation $r$")
    ax1.set_ylabel(r"$|C(r)|/C(0)$")
    ax1.set_title(r"(a) Spatial correlation decay", loc="left")
    apply_axes_polish(ax1, kind="double", title_loc="left", grid=False)
    handles, labels_out = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=spec.legend_size,
        bbox_to_anchor=(0.5, 1.01),
    )

    # --- Panel (b): Lyapunov density convergence ---
    for ia in range(len(a_lyap)):
        sty = series_style(ia)
        ax2.plot(L_vals, density[ia], markevery=1, **sty)

    ax2.set_xlabel(r"Subsystem size $L$")
    ax2.set_ylabel(r"$\lambda_{\max}/L$")
    ax2.set_title(r"(b) Finite-size proxy $\lambda_{\max}/L$", loc="left")
    apply_axes_polish(ax2, kind="double", title_loc="left", grid=False)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(CORR_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CORR_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(CORR_NPZ)
        print(f"Loaded {CORR_NPZ}")
    except FileNotFoundError:
        print("Computing correlation and Lyapunov density data...")
        compute()
        data = safe_load(CORR_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
