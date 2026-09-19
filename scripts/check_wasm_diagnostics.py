"""Check that the diagnostics WebAssembly exports match the native build.

Nine exports are compared on fixed inputs: ``correlation_counts``,
``apen_counts``, ``fuzzy_entropy_sum``, ``ordinal_distribution``,
``diagonal_lines``, ``vertical_lines``, ``multifractal_moments``,
``ami_histogram`` and ``select_dimension_cao``, plus ``zero_one_k``. The
native side is the pyo3 extension, which runs the same ``rust/core`` code;
the browser side is the ``site/wasm`` bundle driven through node. Both sides
receive the same inputs, built once below.

The tolerance is set per export by what the kernel computes:

* Counts — correlation pairs, ApEn template matches, ordinal patterns,
  recurrence line lengths — are integers produced by comparisons and
  additions. They must match exactly. The Cao dimension is an integer and
  must match exactly too.
* ``fuzzy_entropy_sum`` accumulates ``exp(-(d / r)^n)`` over the pairs. The
  wasm module carries its own ``exp`` and sums sequentially, while the
  native build folds per-thread partial sums, so the last bits can differ.
  Measured 1.1e-15 relative on the fixed input below; the limit leaves room
  for another platform ``exp`` without letting a defect through.
* ``ami_histogram`` and ``multifractal_moments`` evaluate ``ln`` (and
  ``powf`` for the moments) inside the kernel, so the same last-bit argument
  applies. They are held to the same 1e-12 relative tolerance.
* ``zero_one_k`` forms the mean-square displacement from an FFT
  autocovariance and correlates it with the lag. The wasm module's own
  ``sin``/``cos`` and FFT differ in the last bits from the native ones, so
  the per-c K values are held to 1e-9 absolute — loose enough for a
  transcendental last-bit difference, tight enough that an all-zero kernel
  (which returns exact zeros) fails the median check below.

Each export also gets a self-check: one value of the native result is
perturbed and the comparison must fail, so a script that compares nothing
cannot pass. ``zero_one_k`` gets a stronger guard instead: the median K on
the logistic map at r = 4 must exceed 0.9, so an all-zero wasm kernel fails
even though every element compares equal.

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
    ami_histogram,
    apen_counts,
    correlation_counts,
    diagonal_lines,
    fuzzy_entropy_sum,
    multifractal_moments,
    ordinal_distribution,
    select_dimension_cao,
    vertical_lines,
    zero_one_k,  # type: ignore[attr-defined]
)
from dynachaos.diagnostics.recurrence import embed_time_delay, recurrence_matrix

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

# Recurrence lines: a small matrix keeps the node message small. The mask is
# sent as 0/1 bytes, row-major, with its side length.
RQA_SIDE = 60
RQA_PERCENTILE = 10
L_MIN = 2
V_MIN = 2

# Multifractal: a small nonnegative field, box sizes that divide it, and a
# few q values. Sizes that divide the field keep every scale valid so no
# NaN crosses the JSON boundary.
MF_NY = 16
MF_NX = 16
MF_BOXES = [2, 4, 8]
MF_Q = [0.0, 1.0, 2.0]

# AMI: a modest lag and bin count.
AMI_TAU_MAX = 20
AMI_BINS = 16

# Cao: an E1 curve that plateaus at 1 from dimension 3 on.
CAO_E1 = [0.35, 0.62, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0]

# zero_one_k: a handful of frequencies over the same series.
ZERO_ONE_C = [0.7, 1.1, 1.9, 2.6, 3.3]
ZERO_ONE_N_CUT = 100

# Justified above: measured 1.1e-15 relative on this input.
FUZZY_REL_TOLERANCE = 1e-12
# ln / powf differ in the last bits between the browser and native math
# libraries; the same 1e-12 relative limit covers them.
LOG_REL_TOLERANCE = 1e-12
# Per-c K carries an FFT and sin/cos; held to a loose absolute bound, with
# the median guard below doing the real work.
ZERO_ONE_ABS_TOLERANCE = 1e-9
ZERO_ONE_MEDIAN_MIN = 0.9

# One node process evaluates all the exports on the spec it reads from
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
const mask = new Uint8Array(spec.mask);
const field = new Float64Array(spec.field);
const e1 = new Float64Array(spec.e1);
process.stdout.write(JSON.stringify({
  correlation_counts: Array.from(mod.correlation_counts(
    traj, spec.dim, new Float64Array(spec.r_values), spec.theiler, true)),
  correlation_counts_euclidean: Array.from(mod.correlation_counts(
    traj, spec.dim, new Float64Array(spec.r_values), spec.theiler_nz, false)),
  apen_counts: Array.from(mod.apen_counts(traj, spec.dim, spec.apen_r)),
  fuzzy_entropy_sum: Array.from(mod.fuzzy_entropy_sum(
    traj, spec.dim, spec.fuzzy_r, spec.fuzzy_n, spec.theiler)),
  ordinal_distribution: Array.from(mod.ordinal_distribution(
    x, spec.ordinal_d, spec.ordinal_tau)),
  diagonal_lines: Array.from(mod.diagonal_lines(mask, spec.side, spec.l_min)),
  vertical_lines: Array.from(mod.vertical_lines(mask, spec.side, spec.v_min)),
  multifractal_moments: Array.from(mod.multifractal_moments(
    field, spec.mf_ny, spec.mf_nx,
    new Float64Array(spec.mf_boxes), new Float64Array(spec.mf_q))),
  ami_histogram: Array.from(mod.ami_histogram(x, spec.ami_tau, spec.ami_bins)),
  select_dimension_cao: Array.from(mod.select_dimension_cao(
    e1, spec.cao_lo, spec.cao_hi, spec.cao_tol,
    spec.cao_span, spec.cao_smooth, spec.cao_min, spec.cao_max)),
  zero_one_k: Array.from(mod.zero_one_k(
    x, new Float64Array(spec.zero_one_c), spec.zero_one_n_cut)),
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
    """Return the flat results from the WebAssembly build."""
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
    """Compare the exports and report the worst difference of each."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1

    x = logistic_series(N_SAMPLES)
    traj = embed_time_delay(x, EMBED_DIM, EMBED_TAU)
    n_templates = len(traj)

    # A small recurrence matrix for the line kernels: threshold the embedded
    # trajectory's pairwise distances at a fixed percentile, as rqa does.
    r_traj = embed_time_delay(x[:RQA_SIDE], EMBED_DIM, EMBED_TAU)
    R, _eps = recurrence_matrix(r_traj, percentile=RQA_PERCENTILE)
    side = int(R.shape[0])
    mask_u8 = R.astype(np.uint8).ravel().tolist()

    # A small nonnegative multifractal field from the same series.
    field = (x[: MF_NY * MF_NX] + 0.5).tolist()

    spec: dict[str, object] = {
        "traj": traj.ravel().tolist(),
        "x": x.tolist(),
        "dim": EMBED_DIM,
        "r_values": R_VALUES,
        "theiler": THEILER,
        "theiler_nz": 5,
        "apen_r": APEN_R,
        "fuzzy_r": FUZZY_R,
        "fuzzy_n": FUZZY_N,
        "ordinal_d": ORDINAL_D,
        "ordinal_tau": ORDINAL_TAU,
        "mask": mask_u8,
        "side": side,
        "l_min": L_MIN,
        "v_min": V_MIN,
        "field": field,
        "mf_ny": MF_NY,
        "mf_nx": MF_NX,
        "mf_boxes": MF_BOXES,
        "mf_q": MF_Q,
        "ami_tau": AMI_TAU_MAX,
        "ami_bins": AMI_BINS,
        "e1": CAO_E1,
        "cao_lo": 0.95,
        "cao_hi": 1.05,
        "cao_tol": 0.02,
        "cao_span": 3,
        "cao_smooth": 1,
        "cao_min": 2,
        "cao_max": 0.0,
        "zero_one_c": ZERO_ONE_C,
        "zero_one_n_cut": ZERO_ONE_N_CUT,
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

    # correlation_counts, Euclidean norm and a nonzero Theiler window.
    out = wasm["correlation_counts_euclidean"]
    theiler_nz = 5
    if not check_header(
        "correlation_counts_euclidean",
        out[:5],
        [float(n_templates), float(EMBED_DIM), float(n_r), float(theiler_nz), 0.0],
    ):
        failed = True
    else:
        native = [float(v) for v in correlation_counts(traj, np.array(R_VALUES), theiler_nz, False)]
        wasm_counts = out[5 + n_r :]
        bad, worst = compare("correlation_counts_euclidean", native, wasm_counts, 0.0)
        print(
            f"correlation_counts_euclidean: {n_r} radii over {n_templates} rows "
            f"(theiler {theiler_nz}), worst {worst:.3e}"
        )
        if bad:
            print("FAIL: correlation_counts_euclidean counts differ")
            failed = True
        elif not self_check("correlation_counts_euclidean", native, wasm_counts, 0.0, 1.0):
            print("FAIL: correlation_counts_euclidean self-check did not trip")
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

    # diagonal_lines: [side, l_min, n_lines, lengths...]
    out = wasm["diagonal_lines"]
    native = [float(v) for v in diagonal_lines(R, L_MIN)]
    if not check_header("diagonal_lines", out[:3], [float(side), float(L_MIN), float(len(native))]):
        failed = True
    else:
        wasm_lengths = out[3:]
        bad, worst = compare("diagonal_lines", native, wasm_lengths, 0.0)
        print(f"diagonal_lines: {len(native)} lines over side {side}, worst {worst:.3e}")
        if bad:
            print("FAIL: diagonal_lines lengths differ")
            failed = True
        elif not self_check("diagonal_lines", native, wasm_lengths, 0.0, 1.0):
            print("FAIL: diagonal_lines self-check did not trip")
            failed = True

    # vertical_lines: [side, v_min, n_lines, lengths...]
    out = wasm["vertical_lines"]
    native = [float(v) for v in vertical_lines(R, V_MIN)]
    if not check_header("vertical_lines", out[:3], [float(side), float(V_MIN), float(len(native))]):
        failed = True
    else:
        wasm_lengths = out[3:]
        bad, worst = compare("vertical_lines", native, wasm_lengths, 0.0)
        print(f"vertical_lines: {len(native)} lines over side {side}, worst {worst:.3e}")
        if bad:
            print("FAIL: vertical_lines lengths differ")
            failed = True
        elif not self_check("vertical_lines", native, wasm_lengths, 0.0, 1.0):
            print("FAIL: vertical_lines self-check did not trip")
            failed = True

    # multifractal_moments: [ny, nx, n_scales, n_q, log_z, alpha_num, f_num, ln_scales]
    out = wasm["multifractal_moments"]
    n_scales = len(MF_BOXES)
    n_q = len(MF_Q)
    if not check_header(
        "multifractal_moments",
        out[:4],
        [float(MF_NY), float(MF_NX), float(n_scales), float(n_q)],
    ):
        failed = True
    else:
        lz, an, fn, ln = multifractal_moments(
            np.array(field).reshape(MF_NY, MF_NX),
            np.array(MF_BOXES, dtype=np.int64),
            np.array(MF_Q),
        )
        native = (
            [float(v) for v in np.asarray(lz).ravel()]
            + [float(v) for v in np.asarray(an).ravel()]
            + [float(v) for v in np.asarray(fn).ravel()]
            + [float(v) for v in np.asarray(ln).ravel()]
        )
        wasm_values = out[4:]
        bad, worst = compare("multifractal_moments", native, wasm_values, LOG_REL_TOLERANCE)
        rel = worst / max(max(abs(v) for v in native), 1.0)
        print(
            f"multifractal_moments: {n_scales} scales x {n_q} q, worst {worst:.3e} "
            f"(rel {rel:.3e}, limit {LOG_REL_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: multifractal_moments differs beyond the stated tolerance")
            failed = True
        elif not self_check("multifractal_moments", native, wasm_values, LOG_REL_TOLERANCE, 1e-6):
            print("FAIL: multifractal_moments self-check did not trip")
            failed = True

    # ami_histogram: [n, tau_max, n_bins, mi...]
    out = wasm["ami_histogram"]
    native = [float(v) for v in ami_histogram(x, AMI_TAU_MAX, AMI_BINS)]
    if not check_header(
        "ami_histogram",
        out[:3],
        [float(N_SAMPLES), float(AMI_TAU_MAX), float(AMI_BINS)],
    ):
        failed = True
    else:
        wasm_mi = out[3:]
        bad, worst = compare("ami_histogram", native, wasm_mi, LOG_REL_TOLERANCE)
        rel = worst / max(max(abs(v) for v in native), 1.0)
        print(
            f"ami_histogram: {AMI_TAU_MAX} lags x {AMI_BINS} bins, worst {worst:.3e} "
            f"(rel {rel:.3e}, limit {LOG_REL_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: ami_histogram differs beyond the stated tolerance")
            failed = True
        elif not self_check("ami_histogram", native, wasm_mi, LOG_REL_TOLERANCE, 1e-6):
            print("FAIL: ami_histogram self-check did not trip")
            failed = True

    # select_dimension_cao: [n, min_dim, max_dim, dim]
    out = wasm["select_dimension_cao"]
    native_dim = float(select_dimension_cao(np.array(CAO_E1), 0.95, 1.05, 0.02, 3, 1, 2, None))
    if not check_header(
        "select_dimension_cao", out[:4], [float(len(CAO_E1)), 2.0, 0.0, native_dim]
    ):
        failed = True
    else:
        print(f"select_dimension_cao: dim {int(out[3])} over {len(CAO_E1)} E1 values")
        if not self_check("select_dimension_cao", [native_dim], [out[3]], 0.0, 1.0):
            print("FAIL: select_dimension_cao self-check did not trip")
            failed = True

    # zero_one_k: [n, n_c, n_cut, k...] — per-c K against pyo3, plus a median
    # guard so an all-zero wasm kernel fails even though zeros compare equal.
    out = wasm["zero_one_k"]
    native = [float(v) for v in zero_one_k(x, np.array(ZERO_ONE_C), ZERO_ONE_N_CUT)]
    if not check_header(
        "zero_one_k",
        out[:3],
        [float(N_SAMPLES), float(len(ZERO_ONE_C)), float(ZERO_ONE_N_CUT)],
    ):
        failed = True
    else:
        wasm_k = out[3:]
        bad, worst = compare("zero_one_k", native, wasm_k, ZERO_ONE_ABS_TOLERANCE)
        median_k = float(np.median(wasm_k))
        print(
            f"zero_one_k: {len(ZERO_ONE_C)} c values, worst {worst:.3e}, "
            f"median K {median_k:.3f} (min {ZERO_ONE_MEDIAN_MIN})"
        )
        if bad:
            print("FAIL: zero_one_k differs beyond the stated tolerance")
            failed = True
        elif median_k <= ZERO_ONE_MEDIAN_MIN:
            print(f"FAIL: zero_one_k median {median_k:.3f} <= {ZERO_ONE_MEDIAN_MIN}")
            failed = True
        elif not self_check("zero_one_k", native, wasm_k, ZERO_ONE_ABS_TOLERANCE, 0.5):
            print("FAIL: zero_one_k self-check did not trip")
            failed = True

    if failed:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
