"""Figure pipeline for the intermittency diagnostics toolkit.

This module is an original dynachaos contribution: it composes the package's
intermittency building blocks on canonical Type-I, on-off, and Lorenz-166.2
signals without returning a mechanism label.
"""

from pathlib import Path

import numpy as np

from dynachaos.diagnostics.intermittency import (
    burst_amplitude_distribution,
    compare_powerlaw_exponential,
    detect_laminar_phases,
    laminar_burst_symmetry,
    return_map_reconstruction,
)
from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps.intermittency import (
    logistic_type_i_oracle,
    lorenz_1662_oracle,
    on_off_oracle,
)

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "intermittency_diagnostics.npz"
OUTPUT_PNG = FIG_DIR / "intermittency_diagnostics.png"
_THIS_FILE = Path(__file__).resolve()

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "type_i_series",
    "type_i_laminar_mask",
    "type_i_laminar_lengths",
    "type_i_return_points",
    "type_i_channel_points",
    "type_i_tail_alpha",
    "type_i_vuong_z",
    "type_i_channel_slope",
    "on_off_series",
    "on_off_laminar_mask",
    "on_off_laminar_lengths",
    "on_off_burst_lengths",
    "on_off_amplitudes",
    "on_off_burst_alpha",
    "on_off_symmetry_p",
    "lorenz_section_points",
    "lorenz_return_points",
    "lorenz_channel_slope",
)


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260601,
    n_type_i=3000,
    n_on_off=20_000,
):
    """Compute deterministic intermittency diagnostics and optionally cache them."""
    rng = np.random.default_rng(seed)
    on_off_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

    type_i = logistic_type_i_oracle(n_type_i, x0=0.2)
    type_i_mask, type_i_lengths = detect_laminar_phases(
        type_i,
        method="period",
        percentile=10.0,
    )
    type_i_comparison = compare_powerlaw_exponential(type_i_lengths)
    type_i_return = return_map_reconstruction(type_i, fs=1.0)

    on_off = on_off_oracle(
        n_on_off,
        x0=1e-4,
        transverse_lyapunov=0.0,
        noise_scale=0.8,
        seed=on_off_seed,
    )
    on_off_threshold = np.percentile(np.abs(on_off), 50.0)
    on_off_mask = np.abs(on_off) <= on_off_threshold
    on_off_symmetry = laminar_burst_symmetry(on_off, on_off_mask)
    on_off_burst = burst_amplitude_distribution(on_off, on_off_mask)

    lorenz = lorenz_1662_oracle(t_span=(0.0, 5.0), dt=0.01, t_transient=0.0)
    lorenz_return = return_map_reconstruction(lorenz[:, 0], fs=100.0)

    payload = {
        "schema_version": np.array([1], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "type_i_series": type_i.astype(np.float64),
        "type_i_laminar_mask": type_i_mask.astype(np.bool_),
        "type_i_laminar_lengths": type_i_lengths.astype(np.int64),
        "type_i_return_points": type_i_return.extrema.points.astype(np.float64),
        "type_i_channel_points": type_i_return.tangent_channel.points.astype(np.float64),
        "type_i_tail_alpha": np.array(
            [type_i_comparison.power_law.alpha],
            dtype=np.float64,
        ),
        "type_i_vuong_z": np.array([type_i_comparison.z], dtype=np.float64),
        "type_i_channel_slope": np.array(
            [type_i_return.tangent_channel.slope],
            dtype=np.float64,
        ),
        "on_off_series": on_off.astype(np.float64),
        "on_off_laminar_mask": on_off_mask.astype(np.bool_),
        "on_off_laminar_lengths": on_off_symmetry.laminar_lengths.astype(np.int64),
        "on_off_burst_lengths": on_off_symmetry.burst_lengths.astype(np.int64),
        "on_off_amplitudes": on_off_burst.amplitudes.astype(np.float64),
        "on_off_burst_alpha": np.array(
            [on_off_burst.power_law.alpha],
            dtype=np.float64,
        ),
        "on_off_symmetry_p": np.array([on_off_symmetry.p_value], dtype=np.float64),
        "lorenz_section_points": lorenz_return.poincare["section_points"].astype(np.float64),
        "lorenz_return_points": lorenz_return.extrema.points.astype(np.float64),
        "lorenz_channel_slope": np.array(
            [lorenz_return.tangent_channel.slope],
            dtype=np.float64,
        ),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the cached diagnostics summary figure."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        color_for,
        figure_spec,
        finalize_legend,
        setup,
    )

    setup()

    type_i_points = np.asarray(data["type_i_return_points"], dtype=np.float64)
    type_i_channel = np.asarray(data["type_i_channel_points"], dtype=np.float64)
    on_off_laminar = np.asarray(data["on_off_laminar_lengths"], dtype=np.float64)
    on_off_burst = np.asarray(data["on_off_burst_lengths"], dtype=np.float64)
    lorenz_section = np.asarray(data["lorenz_section_points"], dtype=np.float64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("double")
    fig, axes = plt.subplots(1, 3, figsize=spec.figsize, constrained_layout=True)

    axes[0].scatter(
        type_i_points[:, 0],
        type_i_points[:, 1],
        s=8,
        color=color_for(0),
        alpha=0.35,
        linewidths=0,
    )
    axes[0].scatter(
        type_i_channel[:, 0],
        type_i_channel[:, 1],
        s=12,
        color=COLORS["black"],
        linewidths=0,
    )
    axes[0].set_title("Type-I return map")
    axes[0].set_xlabel("$x_n$")
    axes[0].set_ylabel("$x_{n+1}$")

    bins = np.histogram_bin_edges(
        np.r_[on_off_laminar, on_off_burst],
        bins="fd",
    )
    axes[1].hist(
        on_off_laminar,
        bins=bins,
        density=True,
        alpha=0.55,
        color=color_for(1),
        label="laminar",
    )
    axes[1].hist(
        on_off_burst,
        bins=bins,
        density=True,
        alpha=0.45,
        color=color_for(2),
        label="burst",
    )
    axes[1].set_title("On-off durations")
    axes[1].set_xlabel("run length")
    axes[1].set_ylabel("density")
    finalize_legend(axes[1], kind="double")

    axes[2].scatter(
        lorenz_section[:, 0],
        lorenz_section[:, 1],
        s=14,
        color=color_for(3),
        alpha=0.8,
        linewidths=0,
    )
    axes[2].set_title("Lorenz $\\rho=166.2$")
    axes[2].set_xlabel("$x(t)$")
    axes[2].set_ylabel("$x(t+\\tau)$")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        apply_axes_polish(ax, kind="double")

    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """Load or compute the intermittency cache, then render the figure."""
    try:
        data = safe_load(OUTPUT_NPZ)
        missing = tuple(key for key in REQUIRED_KEYS if key not in data.files)
        if missing:
            data.close()
            missing_text = ", ".join(missing)
            print(f"Cache {OUTPUT_NPZ} missing keys ({missing_text}); recomputing...")
            data = compute()
    except FileNotFoundError:
        data = compute()

    plot(data)


def _source_file_label():
    try:
        return str(_THIS_FILE.relative_to(_THIS_FILE.parents[3]))
    except ValueError:
        return _THIS_FILE.name


if __name__ == "__main__":
    main()
