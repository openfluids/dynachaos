import os
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


def test_cli_run_smoke_on_existing_figure_cache():
    proc = _run_cli(
        "run",
        "sec02_circle_map",
        "--profile",
        "smoke",
        "--output-root",
        "figures",
    )
    assert proc.returncode == 0, proc.stderr
    assert "[sec02_circle_map]" in proc.stdout


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
