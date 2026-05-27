import os
import subprocess
import sys
from pathlib import Path


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def test_cli_list_sections():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    proc = subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", "list"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sec02_circle_map" in proc.stdout
    assert "sec11_diagnostics" in proc.stdout


def test_cli_version():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    proc = subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("dynachaos ")


def test_cli_run_smoke_on_existing_figure_cache():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynachaos.cli",
            "run",
            "sec02_circle_map",
            "--profile",
            "smoke",
            "--output-root",
            "figures",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[sec02_circle_map]" in proc.stdout


def test_cli_style_list():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    proc = subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", "style", "list"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "editorial-grid" in proc.stdout
    assert "zurich-transit" in proc.stdout


def test_cli_style_preview(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    outdir = tmp_path / "previews"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynachaos.cli",
            "style",
            "preview",
            "--theme",
            "editorial-grid",
            "--output-dir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    out_file = outdir / "editorial-grid.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0
