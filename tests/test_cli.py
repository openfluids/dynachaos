import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    return env


def _write_sec02_artifacts(output_root: Path, *, include_outputs: bool = True) -> None:
    section_dir = output_root / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    if include_outputs:
        for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
            (section_dir / png_name).write_bytes(b"png")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", *args],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
        timeout=30,
    )


def test_cli_list_sections():
    proc = _run_cli("list")
    assert proc.returncode == 0, proc.stderr
    assert "sec02_circle_map" in proc.stdout
    assert "sec11_diagnostics" in proc.stdout


def test_cli_version():
    proc = _run_cli("--version")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("dynachaos ")


def test_cli_run_smoke_on_existing_figure_cache(tmp_path):
    """Smoke profile runs against a real committed cache, copied out of the repo.

    This used to pass --output-root figures, pointing the run at the repository's
    own committed artifacts. The smoke profile validates the cache and then runs
    the section's modules, so the run overwrote figures/sec02_circle_map/*.png in
    place: executing the suite left three modified PNGs in the working tree, and
    a careless `git add` would have committed regenerated binaries.

    The point of the test is that smoke succeeds against a genuine cache rather
    than a synthetic one, so the cache is copied to tmp_path instead of being
    mutated where it lives.
    """
    committed = Path(__file__).resolve().parents[1] / "figures" / "sec02_circle_map"
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    for artifact in committed.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, section_dir / artifact.name)

    before = {p.name: p.stat().st_mtime_ns for p in committed.iterdir() if p.is_file()}

    proc = _run_cli(
        "run",
        "sec02_circle_map",
        "--profile",
        "smoke",
        "--output-root",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "[sec02_circle_map]" in proc.stdout

    after = {p.name: p.stat().st_mtime_ns for p in committed.iterdir() if p.is_file()}
    assert before == after, (
        f"the run modified committed artifacts under figures/sec02_circle_map: "
        f"{sorted(k for k in before if before[k] != after.get(k))}"
    )


def test_cli_style_list():
    proc = _run_cli("style", "list")
    assert proc.returncode == 0, proc.stderr
    assert "editorial-grid" in proc.stdout
    assert "zurich-transit" in proc.stdout


def test_cli_style_preview(tmp_path):
    outdir = tmp_path / "previews"
    proc = _run_cli(
        "style",
        "preview",
        "--theme",
        "editorial-grid",
        "--output-dir",
        str(outdir),
    )
    assert proc.returncode == 0, proc.stderr
    out_file = outdir / "editorial-grid.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_cli_verify_caches_on_temp_output_root(tmp_path):
    _write_sec02_artifacts(tmp_path, include_outputs=False)

    proc = _run_cli(
        "verify",
        "caches",
        "sec02_circle_map",
        "--output-root",
        str(tmp_path),
    )

    assert proc.returncode == 0, proc.stderr
    assert "[sec02_circle_map] 3 caches ok" in proc.stdout


def test_cli_verify_outputs_reports_missing_without_recomputing(tmp_path):
    _write_sec02_artifacts(tmp_path, include_outputs=False)

    proc = _run_cli(
        "verify",
        "outputs",
        "sec02_circle_map",
        "--output-root",
        str(tmp_path),
    )

    assert proc.returncode == 1
    assert "missing expected artifact" in proc.stderr
    assert "devils_staircase.png" in proc.stderr


def test_cli_inspect_section_reports_expected_artifacts(tmp_path):
    _write_sec02_artifacts(tmp_path, include_outputs=False)

    proc = _run_cli(
        "inspect",
        "section",
        "sec02_circle_map",
        "--output-root",
        str(tmp_path),
    )

    assert proc.returncode == 0, proc.stderr
    assert "section\tsec02_circle_map" in proc.stdout
    assert "cache\tok\t" in proc.stdout
    assert "output\tmissing\t" in proc.stdout
    assert "devils_staircase.png" in proc.stdout
    assert "A,rho,lam" in proc.stdout
