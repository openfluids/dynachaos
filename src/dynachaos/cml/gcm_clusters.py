#!/usr/bin/env python3
"""
gcm_clusters: GCM cluster states and collective Lyapunov exponent.

Reproduces cluster formation in globally coupled maps (Kaneko, 1990)
and measures the collective Lyapunov exponent of the mean field.

GCM model:
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + (eps/N) sum_j f(x_n(j))

with f(x) = 1 - a x^2.

Figure 1 (cluster spacetime): Spacetime diagram colored by cluster membership,
    showing coherent, ordered, and partially ordered phases.

Figure 2 (collective Lyapunov): lambda_c vs a, identifying collective chaos
    (lambda_c > 0) regions in parameter space.

OUTPUTS: figures/sec10_gcm/gcm_clusters.npz,
         figures/sec10_gcm/gcm_clusters.png,
         figures/sec10_gcm/collective_lyapunov.npz,
         figures/sec10_gcm/collective_lyapunov.png
USAGE:   python src/dynachaos/cml/gcm_clusters.py
"""

from dynachaos.io.paths import section_dir
import numpy as np

FIG_DIR = section_dir("sec10_gcm")

CLUSTER_NPZ = FIG_DIR / "gcm_clusters.npz"
CLUSTER_PNG = FIG_DIR / "gcm_clusters.png"
COLL_NPZ = FIG_DIR / "collective_lyapunov.npz"
COLL_PNG = FIG_DIR / "collective_lyapunov.png"


def _safe_load(path):
    """Load .npz safely (no deserialization of arbitrary objects)."""
    return np.load(path, allow_pickle = False)


# ---------------------------------------------------------------------------
# GCM model
# ---------------------------------------------------------------------------

def gcm_step(x, a, eps):
    """One GCM step."""
    fx = 1.0 - a * x * x
    mean_field = np.mean(fx)
    return (1.0 - eps) * fx + eps * mean_field


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def detect_clusters(x, tol=1e-6):
    """Assign cluster labels to sites based on proximity.

    Two sites i, j belong to the same cluster if |x(i) - x(j)| < tol.
    Returns an integer array of cluster labels (0, 1, 2, ...).
    """
    N = len(x)
    labels = -np.ones(N, dtype=int)
    cluster_id = 0

    idx_sorted = np.argsort(x)
    x_sorted = x[idx_sorted]

    labels[idx_sorted[0]] = cluster_id
    for k in range(1, N):
        if x_sorted[k] - x_sorted[k - 1] > tol:
            cluster_id += 1
        labels[idx_sorted[k]] = cluster_id

    return labels


# ---------------------------------------------------------------------------
# Cluster computation
# ---------------------------------------------------------------------------

def compute_clusters():
    """Compute GCM cluster spacetime diagram."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    a = 1.55
    eps = 0.1
    N = 100
    n_transient = 20_000
    n_record = 500

    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, N)

    # Transient
    print("  Running transient...")
    for _ in range(n_transient):
        x = gcm_step(x, a, eps)

    # Record
    print("  Recording cluster states...")
    cluster_labels = np.empty((n_record, N), dtype=int)
    x_record = np.empty((n_record, N))

    for t in range(n_record):
        x = gcm_step(x, a, eps)
        cluster_labels[t] = detect_clusters(x)
        x_record[t] = x

    np.savez_compressed(
        CLUSTER_NPZ,
        cluster_labels=cluster_labels,
        x_record=x_record,
        a=np.array([a]),
        eps=np.array([eps]),
        N=np.array([N]),
        n_transient=np.array([n_transient]),
        n_record=np.array([n_record]),
    )
    print(f"Saved {CLUSTER_NPZ}")


# ---------------------------------------------------------------------------
# Collective Lyapunov computation
# ---------------------------------------------------------------------------

def compute_collective():
    """Compute collective Lyapunov exponent vs a.

    Uses the Benettin algorithm adapted to the mean field:
    run two copies of the GCM from nearly identical ICs, track
    mean-field divergence, and periodically renormalize.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    eps = 0.1
    N = 500
    a_values = np.linspace(1.4, 2.0, 100)

    n_transient = 5_000
    n_measure = 50_000
    renorm_interval = 10
    h_delta = 1e-8  # mean-field perturbation scale

    rng = np.random.default_rng(42)
    lyap_c = np.empty(len(a_values))

    for ia, a in enumerate(a_values):
        # Fresh ICs for each a
        x1 = rng.uniform(-1, 1, N)
        x2 = x1.copy()
        # Uniform perturbation: mean(x2 - x1) = h_delta
        x2 += h_delta

        # Transient (both copies)
        for _ in range(n_transient):
            x1 = gcm_step(x1, a, eps)
            x2 = gcm_step(x2, a, eps)

        # Measure growth rate of mean-field difference.
        # Use the same metric (mean-field magnitude) for both
        # measurement and renormalization to avoid size-dependent bias.
        log_sum = 0.0
        n_renorm = 0

        for t in range(n_measure):
            x1 = gcm_step(x1, a, eps)
            x2 = gcm_step(x2, a, eps)

            # Guard: if x2 diverged between renormalization steps, reset
            if not np.all(np.isfinite(x2)):
                x2 = x1.copy()
                x2 += h_delta

            if (t + 1) % renorm_interval == 0:
                diff = x2 - x1
                h_diff = np.abs(np.mean(diff))

                if h_diff > 0:
                    log_sum += np.log(h_diff / h_delta)
                    n_renorm += 1
                    # Renormalize: rescale so |mean(diff)| = h_delta
                    with np.errstate(over="ignore", invalid="ignore"):
                        x2 = x1 + diff * (h_delta / h_diff)
                    if not np.all(np.isfinite(x2)):
                        x2 = x1.copy()
                        x2 += h_delta
                else:
                    x2 = x1.copy()
                    x2 += h_delta

        if n_renorm > 0:
            lyap_c[ia] = log_sum / (n_renorm * renorm_interval)
        else:
            lyap_c[ia] = 0.0

        if (ia + 1) % 10 == 0:
            print(f"  Collective Lyapunov: {ia + 1}/{len(a_values)}")
            np.savez_compressed(
                COLL_NPZ,
                a_values=a_values[:ia + 1],
                lyap_c=lyap_c[:ia + 1],
                eps=np.array([eps]),
                N=np.array([N]),
            )

    np.savez_compressed(
        COLL_NPZ,
        a_values=a_values,
        lyap_c=lyap_c,
        eps=np.array([eps]),
        N=np.array([N]),
    )
    print(f"Saved {COLL_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_clusters(data):
    """Plot spacetime diagram of cluster membership."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from dynachaos.utils.style import (
        apply_axes_polish,
        figure_spec,
        setup,
    )
    setup()

    cluster_labels = data["cluster_labels"]
    a = data["a"][0]
    eps = data["eps"][0]
    N = data["N"][0]

    # Build a discrete colormap with enough distinct colors
    n_clusters_max = int(cluster_labels.max()) + 1
    base_cmap = plt.colormaps.get_cmap("tab20").resampled(min(n_clusters_max, 20))
    colors = [base_cmap(i % 20) for i in range(n_clusters_max)]
    discrete_cmap = ListedColormap(colors)

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    im = ax.imshow(
        cluster_labels,
        aspect="auto",
        origin="lower",
        cmap=discrete_cmap,
        interpolation="nearest",
        vmin=-0.5,
        vmax=n_clusters_max - 0.5,
    )

    ax.set_xlabel("Site index $i$")
    ax.set_ylabel("Time step $n$")
    ax.set_title(
        rf"GCM cluster states, $a = {a}$, $\varepsilon = {eps}$, $N = {int(N)}$",
        loc="left",
    )
    apply_axes_polish(ax, kind="double", title_loc="left")
    ax.grid(False)

    # Colorbar for cluster identity
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Cluster ID", fontsize=spec.label_size)
    cbar.ax.tick_params(labelsize=spec.tick_size)

    fig.savefig(CLUSTER_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CLUSTER_PNG}")


def plot_collective(data):
    """Plot collective Lyapunov exponent vs a."""
    import matplotlib.pyplot as plt
    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        series_style,
        setup,
    )
    setup()

    a_values = data["a_values"]
    lyap_c = data["lyap_c"]
    eps = data["eps"][0]
    N = int(data["N"][0])

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    # Shade region where lambda_c > 0 (collective chaos)
    positive = lyap_c > 0
    if np.any(positive):
        ax.fill_between(
            a_values,
            ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.5,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5,
            where=positive,
            alpha=0.12,
            color=COLORS["red"],
            label="Collective chaos",
            transform=ax.get_xaxis_transform(),
        )

    sty = series_style(0)
    ax.plot(
        a_values,
        lyap_c,
        color=sty["color"],
        linewidth=1.1,
        label=r"$\lambda_c$",
    )

    ax.axhline(0, color=COLORS["red"], lw=0.7, ls="--")

    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$\lambda_c$")
    ax.set_title(
        rf"Collective Lyapunov exponent, $\varepsilon = {eps}$, $N = {N}$",
        loc="left",
    )
    apply_axes_polish(ax, kind="double", title_loc="left")
    finalize_legend(ax, kind="double", loc="upper left")

    fig.savefig(COLL_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {COLL_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Cluster spacetime diagram
    try:
        cluster_data = _safe_load(CLUSTER_NPZ)
        print(f"Loaded {CLUSTER_NPZ}")
    except FileNotFoundError:
        print("Computing GCM cluster states...")
        compute_clusters()
        cluster_data = _safe_load(CLUSTER_NPZ)
    plot_clusters(cluster_data)

    # Collective Lyapunov exponent
    try:
        coll_data = _safe_load(COLL_NPZ)
        print(f"Loaded {COLL_NPZ}")
    except FileNotFoundError:
        print("Computing collective Lyapunov exponent...")
        compute_collective()
        coll_data = _safe_load(COLL_NPZ)
    plot_collective(coll_data)


if __name__ == "__main__":
    main()
