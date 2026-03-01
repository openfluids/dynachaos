#!/usr/bin/env python3
"""
benchmark_logistic: Validate diagnostics against the logistic map x -> 1 - ax^2.

At a=2.0 (fully developed chaos):
- lambda_1 = ln(2) = 0.6931 (exact)
- D2 = 1.0 (exact, full interval; analytically known)

The fully chaotic logistic map at a=2 produces a distribution conjugate to
x = sin^2(pi * theta) with uniform theta, giving exact D2 = 1.0.

References
----------
Ott, E. (2002). "Chaos in Dynamical Systems." 2nd ed., Cambridge University Press.
Strogatz, S.H. (2015). "Nonlinear Dynamics and Chaos." 2nd ed., Westview Press.

OUTPUTS: benchmark_logistic.npz, benchmark_logistic.png
USAGE:   python examples/benchmark_logistic.py                        # .npz exists -> plot only
         rm examples/benchmark_logistic.npz && python examples/benchmark_logistic.py  # recompute
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")

from _pipeline import load_jsonc  # noqa: E402

CONFIG = load_jsonc(SCRIPT.with_suffix(".jsonc"))
REF_LAMBDA1 = CONFIG["ref_lambda1"]
REF_D2 = CONFIG["ref_D2"]


def compute():
    from dynachaos.diagnostics.lyapunov import lyapunov_exponent_1d
    from dynachaos.maps.base import LogisticMap

    a = CONFIG["a"]
    lm = LogisticMap(a=a)

    # Generate trajectory
    N = CONFIG["N"]
    n_transient = CONFIG["n_transient"]
    traj = lm.trajectory(x0=CONFIG["x0"], n_iter=N, n_transient=n_transient)

    # Lyapunov exponent (1D, exact derivative)
    lambda1 = lyapunov_exponent_1d(lm.f, lm.df, x0=CONFIG["x0"],
                                   n_iter=CONFIG["lyap_n_iter"],
                                   n_transient=CONFIG["lyap_n_transient"])

    # Embedding analysis
    from _pipeline import run_embedding_analysis
    results = run_embedding_analysis(
        traj, tau_max=CONFIG["tau_max"], d_max=CONFIG["d_max"],
        n_r=CONFIG["n_r"], verbose=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        traj=traj, a=a, lambda1=lambda1,
        tau_opt=results["tau_opt"], d_opt=results["d_opt"],
        taus=results["taus"], mi_values=results["mi_values"],
        dims=results["dims"], E1=results["E1"], E2=results["E2"],
        fnn_dims=results["fnn_dims"],
        f1=results["f1"], f2=results["f2"], f3=results["f3"],
        r_values=results["r_values"], C_values=results["C_values"],
        D2=results["D2"],
        local_slopes=results["local_slopes"],
        scaling_mask=results["scaling_mask"],
    )
    print(f"Saved {OUTPUT_NPZ}")
    print(f"  lambda_1 = {lambda1:.6f} (ref: {REF_LAMBDA1:.6f})")
    print(f"  D2       = {results['D2']:.3f} (ref: {REF_D2})")


def plot(data):
    from _pipeline import plot_benchmark

    traj = data["traj"]
    results = {
        "tau_opt": int(data["tau_opt"]),
        "d_opt": int(data["d_opt"]),
        "taus": data["taus"],
        "mi_values": data["mi_values"],
        "dims": data["dims"],
        "E1": data["E1"],
        "E2": data["E2"],
        "fnn_dims": data["fnn_dims"],
        "f1": data["f1"],
        "f2": data["f2"],
        "f3": data["f3"],
        "r_values": data["r_values"],
        "C_values": data["C_values"],
        "D2": float(data["D2"]),
        "local_slopes": data["local_slopes"],
        "scaling_mask": data["scaling_mask"].astype(bool),
    }

    # Return map: x_n vs x_{n+1}
    attractor_xy = (traj[:-1], traj[1:])

    plot_benchmark(
        results, attractor_xy, OUTPUT_PNG,
        system_name=f"Logistic map, a={float(data['a']):.1f}",
        ref_D2=REF_D2,
        ref_lambda1=REF_LAMBDA1,
        computed_lambda1=float(data["lambda1"]),
        attractor_xlabel=r"$x_n$",
        attractor_ylabel=r"$x_{n+1}$",
    )


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing logistic benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
