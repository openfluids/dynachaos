"""Check that the WebAssembly build computes what the native build computes.

Both sides evaluate the same fixed tile of the sine circle map and print the
raw bit pattern of every value, so a one-bit disagreement cannot hide behind a
decimal formatter.

The two builds cannot be bit-identical everywhere, and the reason is physical
rather than a defect. Measured on 2026-09-08 over a 64 x 64 tile:

* Native Rust and NumPy agree on all 4096 cells, exactly. Both call the
  platform sine.
* The WebAssembly build calls the sine that Rust compiles into the module.
  It differs from the platform one by about one unit in the last place.
* Below the critical line ``K = 1 / (2 * pi)``, about 0.159, that stays one
  unit in the last place: 2 cells differed, by at most 7.8e-16.
* Above it the map is no longer invertible and the orbit is chaotic, so the
  same one-bit difference grows over 700 iterations: 15 of 1920 cells differed,
  13 of them by more than the ``1e-12`` floor below, by up to 2.9e-2. That 13
  is the number this script prints.

This check therefore asserts what is true in each region. Below the critical
line the rotation number is well conditioned and the two builds must agree to a
tight bound, so a port defect shows up at once. Above it the value of a single
cell is not reproducible across sine implementations, and demanding that it
were would only invite someone to widen the bound until the check meant
nothing. What is asserted there instead is that divergence stays rare.

Do not loosen the subcritical bound to make this pass. A difference there means
the browser computes something the paper did not.

The native side is built WITHOUT ``-C target-cpu=native``. The repository sets
that flag for local builds, and it lets the compiler fuse a multiply and an add
into a single instruction that ``wasm32`` has no equivalent for. Comparing a
natively tuned build against the browser build would measure the compiler
rather than the port.

Run it with::

    uv run python scripts/check_wasm_parity.py
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from math import isfinite, pi
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WASM_CRATE = PROJECT_ROOT / "rust" / "wasm"
SITE_WASM = PROJECT_ROOT / "site" / "wasm"

# The tile both emitters compute. The grid size is not repeated here: the
# kernel reports it in the header, and this script reads it back, so the row a
# cell belongs to cannot drift out of step with the emitters. The K range is
# not in the header, so it does stay here; keep it in step with
# rust/wasm/examples/tile_reference.rs and rust/wasm/parity_tile.mjs.
#
# If the two emitters ever disagree with each other, this check says so: the
# header comparison below catches a different grid, and a different K range or
# starting angle breaks the subcritical bound everywhere at once.
HEADER = 4
K_MIN = 0.0
K_MAX = 0.3

# The sine circle map stops being invertible here. Orbits above this line are
# chaotic, so a one-bit difference grows instead of staying put.
K_CRITICAL = 1.0 / (2.0 * pi)

# Below the critical line the two builds agreed to 7.8e-16 when measured. This
# limit leaves room for another platform sine without letting a defect through.
SUBCRITICAL_LIMIT = 1e-14

# Above it, a cell counts as diverged past this. Anything closer is the same
# orbit seen through a slightly different sine.
CHAOTIC_NOISE_FLOOR = 1e-12

# Sensitivity explained 0.7% of the chaotic cells when measured. Many more than
# that is a defect, not chaos.
CHAOTIC_SHARE_LIMIT = 0.05


def native_tile() -> list[int]:
    """Return the tile from the native build, as f64 bit patterns."""
    # An empty RUSTFLAGS replaces the repository's target-cpu=native setting.
    env = {**os.environ, "RUSTFLAGS": ""}
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(WASM_CRATE / "Cargo.toml"),
            "--release",
            "--example",
            "tile_reference",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return [int(line, 16) for line in result.stdout.split()]


def wasm_tile() -> list[int]:
    """Return the tile from the WebAssembly build, as f64 bit patterns."""
    result = subprocess.run(
        ["node", str(WASM_CRATE / "parity_tile.mjs"), str(SITE_WASM)],
        capture_output=True,
        text=True,
        check=True,
    )
    return [int(line, 16) for line in result.stdout.split()]


def as_float(bits: int) -> float:
    """Return the f64 that a bit pattern encodes."""
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def k_of_row(row: int, n_k: int) -> float:
    """Return the K value of a tile row."""
    return K_MIN + row * (K_MAX - K_MIN) / (n_k - 1)


def main() -> int:
    """Compare the two tiles region by region and report."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1

    native = native_tile()
    wasm = wasm_tile()

    if len(native) != len(wasm):
        print(f"FAIL: native returned {len(native)} values, wasm returned {len(wasm)}")
        return 1
    if native[:HEADER] != wasm[:HEADER]:
        print("FAIL: the headers disagree, so the two builds ran different grids")
        return 1

    # The kernel reports the grid it actually used in the first two header
    # elements, after its own clamping. Read it back rather than assuming it.
    n_omega = int(as_float(native[0]))
    n_k = int(as_float(native[1]))

    native_values = [as_float(bits) for bits in native[HEADER:]]
    wasm_values = [as_float(bits) for bits in wasm[HEADER:]]

    if len(native_values) != n_omega * n_k:
        print(
            f"FAIL: the header says {n_k} x {n_omega} cells but "
            f"{len(native_values)} values followed it"
        )
        return 1

    subcritical_worst = 0.0
    chaotic_cells = 0
    chaotic_diverged = 0
    chaotic_worst = 0.0

    for index, (left, right) in enumerate(zip(native_values, wasm_values, strict=True)):
        if not (isfinite(left) and isfinite(right)):
            print(f"FAIL: cell {index} is not finite (native {left!r}, wasm {right!r})")
            return 1
        difference = abs(left - right)
        if k_of_row(index // n_omega, n_k) <= K_CRITICAL:
            subcritical_worst = max(subcritical_worst, difference)
        else:
            chaotic_cells += 1
            chaotic_worst = max(chaotic_worst, difference)
            if difference > CHAOTIC_NOISE_FLOOR:
                chaotic_diverged += 1

    share = chaotic_diverged / max(chaotic_cells, 1)
    print(f"compared {len(native_values)} cells ({n_k} x {n_omega})")
    print(f"  below K = {K_CRITICAL:.4f}: worst difference {subcritical_worst:.3e}")
    print(
        f"  above K = {K_CRITICAL:.4f}: {chaotic_diverged} of {chaotic_cells} cells diverged "
        f"({share:.1%}), worst {chaotic_worst:.3e}"
    )

    failed = False
    if subcritical_worst > SUBCRITICAL_LIMIT:
        print(
            f"FAIL: below the critical line the builds must agree to {SUBCRITICAL_LIMIT:.1e}, "
            f"got {subcritical_worst:.3e}. That region is well conditioned, so this is a defect."
        )
        failed = True
    if share > CHAOTIC_SHARE_LIMIT:
        print(
            f"FAIL: {share:.1%} of the chaotic cells diverged, above the "
            f"{CHAOTIC_SHARE_LIMIT:.0%} guide. Sensitivity explains a few, not this many."
        )
        failed = True

    if failed:
        print("\nInvestigate before changing this check. Read the module docstring first.")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
