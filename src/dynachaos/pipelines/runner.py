"""Pipeline runner for paper section generation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dynachaos.pipelines.registry import get_section, list_sections

_ALLOWED_OUTPUT_SUFFIXES = {".npz", ".png"}


def _repo_src_dir() -> Path | None:
    """Return local repo src/ path when running from source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "dynachaos"
        if candidate.is_dir():
            return candidate.parent
    return None


def _runner_env(output_root: Path, profile: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DYNACHAOS_OUTPUT_ROOT"] = str(output_root)
    env["DYNACHAOS_PROFILE"] = profile

    src_dir = _repo_src_dir()
    if src_dir is not None:
        current = env.get("PYTHONPATH")
        if current:
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{current}"
        else:
            env["PYTHONPATH"] = str(src_dir)

    return env


def _run_module(module_name: str, output_root: Path, profile: str) -> None:
    env = _runner_env(output_root, profile)
    cmd = [sys.executable, "-m", module_name]
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Module run failed: {' '.join(cmd)} (exit {proc.returncode})")


def _validate_npz(path: Path, *, required_keys: tuple[str, ...], section_id: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = tuple(key for key in required_keys if key not in data.files)
    except Exception as exc:
        raise RuntimeError(f"Section {section_id} has malformed NPZ artifact: {path}") from exc

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Section {section_id} artifact {path} is missing required NPZ keys: {missing_text}"
        )


def _validate_artifact(path: Path, *, required_keys: tuple[str, ...], section_id: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Section {section_id} is missing expected artifact: {path}")

    suffix = path.suffix.lower()
    if suffix not in _ALLOWED_OUTPUT_SUFFIXES:
        raise RuntimeError(
            f"Section {section_id} artifact {path} has unsupported extension: {path.suffix}"
        )

    if suffix == ".npz":
        _validate_npz(path, required_keys=required_keys, section_id=section_id)
        return

    if path.stat().st_size == 0:
        raise RuntimeError(f"Section {section_id} artifact {path} is empty")


def run_section(
    section_id: str,
    *,
    output_root: str | Path | None = None,
    profile: str = "paper",
    recompute: bool = False,
) -> list[Path]:
    """Run one section pipeline and return expected output paths."""
    spec = get_section(section_id)
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else (Path.cwd() / "figures").resolve()
    )

    cache_paths = [root / section_id / name for name in spec.cache_files]
    output_paths = [root / section_id / name for name in spec.output_files]

    if profile not in {"paper", "smoke"}:
        raise ValueError("profile must be one of: paper, smoke")

    if recompute:
        for path in output_paths:
            if path.exists():
                path.unlink()

    if profile == "smoke":
        for path in cache_paths:
            _validate_artifact(
                path,
                required_keys=spec.required_npz_keys(path.name),
                section_id=section_id,
            )

    for module_name in spec.modules:
        _run_module(module_name, root, profile)

    for path in output_paths:
        _validate_artifact(
            path,
            required_keys=spec.required_npz_keys(path.name),
            section_id=section_id,
        )

    return output_paths


def run_all(
    *,
    output_root: str | Path | None = None,
    profile: str = "paper",
    recompute: bool = False,
) -> dict[str, list[Path]]:
    """Run all section pipelines in paper order."""
    results: dict[str, list[Path]] = {}
    for section_id in list_sections():
        results[section_id] = run_section(
            section_id,
            output_root=output_root,
            profile=profile,
            recompute=recompute,
        )
    return results
