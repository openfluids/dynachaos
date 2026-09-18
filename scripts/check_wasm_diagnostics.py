"""Check that the diagnostics WebAssembly exports match the native build.

Four exports are compared on one fixed input: ``correlation_counts``,
``apen_counts``, ``fuzzy_entropy_sum`` and ``ordinal_distribution``. The
native side is the pyo3 extension, which runs the same ``rust/core`` code;
the browser side is the ``site/wasm`` bundle driven through node. Both sides
receive the same embedded trajectory, built once by
``dynachaos.diagnostics.recurrence.embed_time_delay``.

The tolerance is set per export by what the kernel computes:

* Counts — correlation pairs, ApEn template matches, ordinal patterns — are
  integers produced by comparisons and additions. They must match exactly.
* ``fuzzy_entropy_sum`` accumulates ``exp(-(d / r)^n)`` over the pairs. The
  wasm module carries its own ``exp`` and sums sequentially, while the
  native build folds per-thread partial sums, so the last bits can differ.
  Measured 1.1e-15 relative on the fixed input below; the limit leaves room
  for another platform ``exp`` without letting a defect through.

Each export also gets a self-check: one value of the native result is
perturbed and the comparison must fail, so a script that compares nothing
cannot pass.

Run it with::

    uv run python scripts/check_wasm_diagnostics.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import isfinite
from pathlib import Path

import numpy as np

from dynachaos._rust import (
    apen_counts,
    correlation_counts,
    fuzzy_entropy_sum,
    ordinal_distribution,
)
from dynachaos.diagnostics.recurrence import embed_time_delay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"

# The fixed input every comparison below runs on: a deterministic chaotic
# series (logistic map at r = 4) embedded once and shared by both sides.
N_SAMPLES = 2000
EMBED_DIM = 3
EMBED_TAU = 2
THEILER = 0
R_VALUES = [0.02, 0.05, 0.1, 0.2, 0.4]
APEN_R = 0.2
FUZZY_R = 0.2
FUZZY_N = 2
ORDINAL_D = 3
ORDINAL_TAU = 2

# Justified above: measured 1.1e-15 relative on this input.
FUZZY_REL_TOLERANCE = 1e-12

# One node process evaluates all four exports on the spec it reads from
# stdin and prints the flat results as JSON. JSON round-trips f64 exactly,
# so an integer count cannot drift through the encoding.
NODE_DRIVER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
const dir = process.argv[1];
const spec = JSON.parse(readFileSync(0, "utf8"));
const mod = await import(pathToFileURL(join(dir, "dynachaos_wasm.js")).href);
await mod.default({ module_or_path: readFileSync(join(dir, "dynachaos_wasm_bg.wasm")) });
const traj = new Float64Array(spec.traj);
const x = new Float64Array(spec.x);
process.stdout.write(JSON.stringify({
  correlation_counts: Array.from(mod.correlation_counts(
    traj, spec.dim, new Float64Array(spec.r_values), spec.theiler, true)),
  apen_counts: Array.from(mod.apen_counts(traj, spec.dim, spec.apen_r)),
  fuzzy_entropy_sum: Array.from(mod.fuzzy_entropy_sum(
    traj, spec.dim, spec.fuzzy_r, spec.fuzzy_n, spec.theiler)),
  ordinal_distribution: Array.from(mod.ordinal_distribution(
    x, spec.ordinal_d, spec.ordinal_tau)),
}));
"""


def logistic_series(n: int) -> np.ndarray:
    """Return n samples of the logistic map at r = 4."""
    x = np.empty(n)
    value = 0.123456789
    for i in range(n):
        value = 4.0 * value * (1.0 - value)
        x[i] = value
    return x


def wasm_results(spec: dict[str, object]) -> dict[str, list[float]]:
    """Return the four flat results from the WebAssembly build."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", NODE_DRIVER, str(SITE_WASM)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def compare(
    name: str,
    native_values: list[float],
    wasm_values: list[float],
    rel_tolerance: float,
) -> tuple[bool, float]:
    """Compare element by element. Return (failed, worst difference)."""
    if len(native_values) != len(wasm_values):
        return True, float("inf")
    worst = 0.0
    for index, (native, wasm) in enumerate(zip(native_values, wasm_values, strict=True)):
        if not (isfinite(native) and isfinite(wasm)):
            print(f"FAIL: {name} value {index} is not finite ({native!r} vs {wasm!r})")
            return True, float("inf")
        difference = abs(native - wasm)
        worst = max(worst, difference)
        scale = max(abs(native), abs(wasm), 1.0)
        if difference > rel_tolerance * scale:
            return True, worst
    return False, worst


def check_header(name: str, header: list[float], expected: list[float]) -> bool:
    """Return True when the wasm header reports the values the script sent."""
    if header != expected:
        print(f"FAIL: {name} header {header} does not match the request {expected}")
        return False
    return True


def self_check(
    name: str,
    native_values: list[float],
    wasm_values: list[float],
    rel_tolerance: float,
    perturbation: float,
) -> bool:
    """Return True when perturbing one native value trips the comparison."""
    perturbed = list(native_values)
    perturbed[0] += perturbation
    failed, _ = compare(name, perturbed, wasm_values, rel_tolerance)
    return failed


def main() -> int:
    """Compare the four exports and report the worst difference of each."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1

    x = logistic_series(N_SAMPLES)
    traj = embed_time_delay(x, EMBED_DIM, EMBED_TAU)
    n_templates = len(traj)

    spec = {
        "traj": traj.ravel().tolist(),
        "x": x.tolist(),
        "dim": EMBED_DIM,
        "r_values": R_VALUES,
        "theiler": THEILER,
        "apen_r": APEN_R,
        "fuzzy_r": FUZZY_R,
        "fuzzy_n": FUZZY_N,
        "ordinal_d": ORDINAL_D,
        "ordinal_tau": ORDINAL_TAU,
    }
    wasm = wasm_results(spec)

    failed = False

    # correlation_counts: [n_pts, dim, n_r, theiler, chebyshev, radii..., counts...]
    out = wasm["correlation_counts"]
    n_r = len(R_VALUES)
    if not check_header(
        "correlation_counts",
        out[:5],
        [float(n_templates), float(EMBED_DIM), float(n_r), float(THEILER), 1.0],
    ):
        failed = True
    elif out[5 : 5 + n_r] != sorted(R_VALUES):
        print(f"FAIL: correlation_counts radii {out[5 : 5 + n_r]} != {sorted(R_VALUES)}")
        failed = True
    else:
        native = [float(v) for v in correlation_counts(traj, np.array(R_VALUES), THEILER, True)]
        wasm_counts = out[5 + n_r :]
        bad, worst = compare("correlation_counts", native, wasm_counts, 0.0)
        print(f"correlation_counts: {n_r} radii over {n_templates} rows, worst {worst:.3e}")
        if bad:
            print("FAIL: correlation_counts counts differ")
            failed = True
        elif not self_check("correlation_counts", native, wasm_counts, 0.0, 1.0):
            print("FAIL: correlation_counts self-check did not trip")
            failed = True

    # apen_counts: [n_pts, dim, r, counts...]
    out = wasm["apen_counts"]
    if not check_header("apen_counts", out[:3], [float(n_templates), float(EMBED_DIM), APEN_R]):
        failed = True
    else:
        native = [float(v) for v in apen_counts(traj, APEN_R)]
        wasm_counts = out[3:]
        bad, worst = compare("apen_counts", native, wasm_counts, 0.0)
        print(f"apen_counts: {n_templates} rows, worst {worst:.3e}")
        if bad:
            print("FAIL: apen_counts counts differ")
            failed = True
        elif not self_check("apen_counts", native, wasm_counts, 0.0, 1.0):
            print("FAIL: apen_counts self-check did not trip")
            failed = True

    # fuzzy_entropy_sum: [n_pts, dim, r, n, theiler, sum]
    out = wasm["fuzzy_entropy_sum"]
    if not check_header(
        "fuzzy_entropy_sum",
        out[:5],
        [float(n_templates), float(EMBED_DIM), FUZZY_R, float(FUZZY_N), float(THEILER)],
    ):
        failed = True
    else:
        native = [float(fuzzy_entropy_sum(traj, FUZZY_R, FUZZY_N, THEILER))]
        wasm_sum = out[5:]
        bad, worst = compare("fuzzy_entropy_sum", native, wasm_sum, FUZZY_REL_TOLERANCE)
        rel = worst / max(abs(native[0]), 1.0)
        print(
            f"fuzzy_entropy_sum: worst {worst:.3e} (rel {rel:.3e}, limit {FUZZY_REL_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: fuzzy_entropy_sum differs beyond the stated tolerance")
            failed = True
        elif not self_check(
            "fuzzy_entropy_sum", native, wasm_sum, FUZZY_REL_TOLERANCE, abs(native[0]) * 1e-6
        ):
            print("FAIL: fuzzy_entropy_sum self-check did not trip")
            failed = True

    # ordinal_distribution: [n, d, tau, n_windows, counts...]
    out = wasm["ordinal_distribution"]
    native_counts, native_windows = ordinal_distribution(x, ORDINAL_D, ORDINAL_TAU)
    if not check_header(
        "ordinal_distribution",
        out[:4],
        [float(N_SAMPLES), float(ORDINAL_D), float(ORDINAL_TAU), float(native_windows)],
    ):
        failed = True
    else:
        native = [float(v) for v in native_counts]
        wasm_counts = out[4:]
        bad, worst = compare("ordinal_distribution", native, wasm_counts, 0.0)
        print(f"ordinal_distribution: {len(native)} patterns, worst {worst:.3e}")
        if bad:
            print("FAIL: ordinal_distribution counts differ")
            failed = True
        elif not self_check("ordinal_distribution", native, wasm_counts, 0.0, 1.0):
            print("FAIL: ordinal_distribution self-check did not trip")
            failed = True

    if failed:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
