"""Export decimated, web-ready JSON payloads from cached figure .npz arrays.

This is the data layer for an interactive gallery: it reads the arrays already
cached under ``figures/<section>/<name>.npz`` by the reproduction pipelines and
writes self-describing JSON payloads under ``site/data/<section>/<name>.json``
that a browser can render as charts. No HTML/CSS/JS is produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# Emit floats with 6 significant figures to keep payloads small.
_SIG_FIGS = 6


def _round_sig(value: float) -> float:
    """Round a single float to 6 significant figures."""
    if not np.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return float(f"{value:.{_SIG_FIGS}g}")


def _round_list(values: np.ndarray) -> list[float]:
    """Round a 1-D array to 6 significant figures, as a plain list."""
    return [_round_sig(float(v)) for v in values]


def _round_grid(values: np.ndarray) -> list[list[float]]:
    """Round a 2-D array to 6 significant figures, as nested lists."""
    return [_round_list(row) for row in values]


def _require(npz: np.lib.npyio.NpzFile, key: str, source: str) -> np.ndarray:
    """Fetch ``key`` from ``npz``, failing loudly if it is missing."""
    if key not in npz.files:
        raise KeyError(f"{source}: expected key {key!r}, found {list(npz.files)}")
    return npz[key]


def _load(section: str, name: str) -> np.lib.npyio.NpzFile:
    path = FIGURES_DIR / section / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    return np.load(path, allow_pickle=False)


def _no_decimation(n: int) -> dict[str, Any]:
    return {"method": "none", "from": n, "to": n}


def _stride_decimation(n_from: int, n_to: int) -> dict[str, Any]:
    return {"method": "stride", "from": n_from, "to": n_to}


def export_rqa_measures() -> dict[str, Any]:
    """sec11_diagnostics/rqa_measures: four RQA measures vs embedding delay D."""
    source = "sec11_diagnostics/rqa_measures"
    npz = _load("sec11_diagnostics", "rqa_measures")
    d = _require(npz, "D", source)
    n = d.shape[0]
    traces = []
    for key, name in (
        ("RR", "Recurrence rate"),
        ("DET", "Determinism"),
        ("LAM", "Laminarity"),
        ("ENTR", "Entropy of diagonal lines"),
    ):
        y = _require(npz, key, source)
        traces.append({"name": name, "x": _round_list(d), "y": _round_list(y)})
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "RQA measure"}},
        "panels": [{"title": "Recurrence quantification measures", "traces": traces}],
    }


def export_test01_sweep() -> dict[str, Any]:
    """sec11_diagnostics/test01_sweep: 0-1 test statistic vs logistic nonlinearity a."""
    source = "sec11_diagnostics/test01_sweep"
    npz = _load("sec11_diagnostics", "test01_sweep")
    a = _require(npz, "a", source)
    k = _require(npz, "K", source)
    n = a.shape[0]
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "Nonlinearity a"}, "y": {"label": "K_01 (0-1 test)"}},
        "panels": [
            {
                "title": "0-1 test for chaos: logistic map",
                "traces": [{"name": "K_01", "x": _round_list(a), "y": _round_list(k)}],
            }
        ],
    }


def export_permutation_entropy() -> dict[str, Any]:
    """sec11_diagnostics/permutation_entropy: PE vs a (logistic) and vs D (delayed logistic).

    Keys are ``a``, ``H_logistic``, ``D``, ``H_delayed`` (matches the brief).
    """
    source = "sec11_diagnostics/permutation_entropy"
    npz = _load("sec11_diagnostics", "permutation_entropy")
    a = _require(npz, "a", source)
    h_logistic = _require(npz, "H_logistic", source)
    d = _require(npz, "D", source)
    h_delayed = _require(npz, "H_delayed", source)
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(a.shape[0] + d.shape[0]),
        "axes": {"x": {"label": "a or D"}, "y": {"label": "H_PE (permutation entropy)"}},
        "panels": [
            {
                "title": "Logistic map",
                "traces": [{"name": "H_PE", "x": _round_list(a), "y": _round_list(h_logistic)}],
            },
            {
                "title": "Delayed logistic, alpha = 0.3",
                "traces": [{"name": "H_PE", "x": _round_list(d), "y": _round_list(h_delayed)}],
            },
        ],
    }


def export_complexity_entropy_plane() -> dict[str, Any]:
    """sec11_diagnostics/complexity_entropy_plane: (H, C) scatter for two maps.

    Keys are ``a``, ``H_logistic``, ``C_logistic``, ``D``, ``H_delayed``, ``C_delayed``.
    """
    source = "sec11_diagnostics/complexity_entropy_plane"
    npz = _load("sec11_diagnostics", "complexity_entropy_plane")
    h_logistic = _require(npz, "H_logistic", source)
    c_logistic = _require(npz, "C_logistic", source)
    h_delayed = _require(npz, "H_delayed", source)
    c_delayed = _require(npz, "C_delayed", source)
    n = h_logistic.shape[0] + h_delayed.shape[0]
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _no_decimation(n),
        "axes": {
            "x": {"label": "Normalised permutation entropy H"},
            "y": {"label": "Statistical complexity C"},
        },
        "panels": [
            {
                "title": "Complexity-entropy plane (d=5)",
                "traces": [
                    {
                        "name": "Logistic map",
                        "x": _round_list(h_logistic),
                        "y": _round_list(c_logistic),
                    },
                    {
                        "name": "Delayed logistic map",
                        "x": _round_list(h_delayed),
                        "y": _round_list(c_delayed),
                    },
                ],
            }
        ],
    }


def export_correlation_dimension() -> dict[str, Any]:
    """sec07_fractalization/correlation_dimension: D2 vs D with an error band."""
    source = "sec07_fractalization/correlation_dimension"
    npz = _load("sec07_fractalization", "correlation_dimension")
    d = _require(npz, "D", source)
    d2 = _require(npz, "D2", source)
    d2_err = _require(npz, "D2_err", source)
    n = d.shape[0]
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "Correlation dimension D2"}},
        "panels": [
            {
                "title": "Delayed logistic map, alpha = 0.3",
                "traces": [
                    {
                        "name": "D2",
                        "x": _round_list(d),
                        "y": _round_list(d2),
                        "yerr": _round_list(d2_err),
                    }
                ],
            }
        ],
    }


def export_lyapunov_vs_d() -> dict[str, Any]:
    """sec05_oscillation/lyapunov_vs_D: two-component Lyapunov spectrum vs D."""
    source = "sec05_oscillation/lyapunov_vs_D"
    npz = _load("sec05_oscillation", "lyapunov_vs_D")
    d = _require(npz, "D", source)
    spectra = _require(npz, "spectra", source)
    spectra_err = _require(npz, "spectra_err", source)
    n = d.shape[0]
    traces = []
    for col in range(spectra.shape[1]):
        traces.append(
            {
                "name": f"lambda_{col + 1}",
                "x": _round_list(d),
                "y": _round_list(spectra[:, col]),
                "yerr": _round_list(spectra_err[:, col]),
            }
        )
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "Lyapunov exponent"}},
        "panels": [{"title": "Delayed logistic map, alpha = 0.3", "traces": traces}],
    }


def export_lyapunov_vs_db() -> dict[str, Any]:
    """sec06_three_torus/lyapunov_vs_DB: Lyapunov spectrum vs D_B for three epsilon.

    Keys are ``DB``, ``eps_values`` (the three epsilon values, in the order the
    ``eps_<value>_spectra`` keys are named), and one ``eps_<value>_spectra``
    array of shape (500, 4) per epsilon. Only the first 3 columns are used
    (per the brief).
    """
    source = "sec06_three_torus/lyapunov_vs_DB"
    npz = _load("sec06_three_torus", "lyapunov_vs_DB")
    db = _require(npz, "DB", source)
    eps_values = _require(npz, "eps_values", source)
    n = db.shape[0]
    panels = []
    for eps in eps_values:
        key = f"eps_{eps:g}_spectra"
        spectra = _require(npz, key, source)
        traces = [
            {
                "name": f"lambda_{col + 1}",
                "x": _round_list(db),
                "y": _round_list(spectra[:, col]),
            }
            for col in range(3)
        ]
        panels.append({"title": f"epsilon = {eps:g}", "traces": traces})
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D_B"}, "y": {"label": "Lyapunov exponent"}},
        "panels": panels,
    }


def export_map_i_attractors(max_points: int = 20000) -> dict[str, Any]:
    """sec04_doubling/map_I_attractors: (X, Y) projections for three D values.

    Keys are ``D_2.11_traj``, ``D_2.16_traj``, ``D_2.19_traj`` (each (100000, 3))
    and ``D_values`` giving the order (matches labels torus / 2x torus / chaos
    used by the reproduction script).
    """
    source = "sec04_doubling/map_I_attractors"
    npz = _load("sec04_doubling", "map_I_attractors")
    d_values = _require(npz, "D_values", source)
    labels = ["torus", "2x torus", "chaos"]
    panels = []
    n_from = None
    n_to = None
    for d_value, label in zip(d_values, labels, strict=True):
        key = f"D_{d_value:g}_traj"
        traj = _require(npz, key, source)
        n_from = traj.shape[0]
        idx = np.arange(0, n_from, max(1, n_from // max_points))[:max_points]
        n_to = idx.shape[0]
        panels.append(
            {
                "title": f"D={d_value:g} ({label})",
                "traces": [
                    {
                        "name": label,
                        "x": _round_list(traj[idx, 0]),
                        "y": _round_list(traj[idx, 1]),
                    }
                ],
            }
        )
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _stride_decimation(n_from, n_to),
        "axes": {"x": {"label": "X"}, "y": {"label": "Y"}},
        "panels": panels,
    }


def export_arnold_tongues(max_axis: int = 600) -> dict[str, Any]:
    """sec02_circle_map/arnold_tongues: rotation number heatmap over (Omega, K)."""
    source = "sec02_circle_map/arnold_tongues"
    npz = _load("sec02_circle_map", "arnold_tongues")
    omega = _require(npz, "Omega", source)
    k = _require(npz, "K", source)
    rho = _require(npz, "rho", source)
    n_k, n_omega = rho.shape

    k_step = max(1, -(-n_k // max_axis))
    omega_step = max(1, -(-n_omega // max_axis))

    k_idx = np.arange(0, n_k, k_step)
    omega_idx = np.arange(0, n_omega, omega_step)
    rho_ds = rho[np.ix_(k_idx, omega_idx)]

    return {
        "figure": source,
        "kind": "heatmap",
        "decimation": _stride_decimation(n_k * n_omega, rho_ds.shape[0] * rho_ds.shape[1]),
        "axes": {"x": {"label": "Bare frequency Omega"}, "y": {"label": "Nonlinearity K"}},
        "panels": [
            {
                "title": "Rotation number over the circle-map parameter plane",
                "x": _round_list(omega[omega_idx]),
                "y": _round_list(k[k_idx]),
                "z": _round_grid(rho_ds),
            }
        ],
    }


def export_phase_diagram() -> dict[str, Any]:
    """sec09_pattern/phase_diagram: activity maps over (a, epsilon), full resolution.

    Keys are ``a``, ``eps``, ``lam``, ``spatial_activity`` (plus a
    ``schema_version`` int that is not chart data and is not emitted).
    """
    source = "sec09_pattern/phase_diagram"
    npz = _load("sec09_pattern", "phase_diagram")
    a = _require(npz, "a", source)
    eps = _require(npz, "eps", source)
    lam = _require(npz, "lam", source)
    spatial_activity = _require(npz, "spatial_activity", source)
    n = lam.shape[0] * lam.shape[1]
    return {
        "figure": source,
        "kind": "heatmap",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "Nonlinearity a"}, "y": {"label": "Coupling epsilon"}},
        "panels": [
            {
                "title": "Largest Lyapunov exponent",
                "x": _round_list(a),
                "y": _round_list(eps),
                "z": _round_grid(lam),
            },
            {
                "title": "Spatial activity map",
                "x": _round_list(a),
                "y": _round_list(eps),
                "z": _round_grid(spatial_activity),
            },
        ],
    }


EXPORTERS: tuple[tuple[str, str, Any], ...] = (
    ("sec11_diagnostics", "rqa_measures", export_rqa_measures),
    ("sec11_diagnostics", "test01_sweep", export_test01_sweep),
    ("sec11_diagnostics", "permutation_entropy", export_permutation_entropy),
    ("sec11_diagnostics", "complexity_entropy_plane", export_complexity_entropy_plane),
    ("sec07_fractalization", "correlation_dimension", export_correlation_dimension),
    ("sec05_oscillation", "lyapunov_vs_D", export_lyapunov_vs_d),
    ("sec06_three_torus", "lyapunov_vs_DB", export_lyapunov_vs_db),
    ("sec04_doubling", "map_I_attractors", export_map_i_attractors),
    ("sec02_circle_map", "arnold_tongues", export_arnold_tongues),
    ("sec09_pattern", "phase_diagram", export_phase_diagram),
)


def main() -> int:
    """Export all ten figure-data JSON payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "site" / "data",
        help="Output root directory (default: site/data)",
    )
    args = parser.parse_args()

    rows: list[tuple[str, str, int]] = []
    for section, name, exporter in EXPORTERS:
        payload = exporter()
        out_dir = args.outdir / section
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.json"
        text = json.dumps(payload, separators=(",", ":"))
        out_path.write_text(text)
        rows.append((f"{section}/{name}", payload["kind"], len(text.encode("utf-8"))))

    header = f"{'figure':<45} {'kind':<8} {'bytes':>10}"
    print(header)
    print("-" * len(header))
    for figure, kind, nbytes in rows:
        print(f"{figure:<45} {kind:<8} {nbytes:>10,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
