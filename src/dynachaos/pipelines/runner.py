"""Pipeline runner for paper section generation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dynachaos.pipelines.registry import get_section, list_sections


def _repo_src_dir() -> Path | None:
    """Return local repo src/ path when running from source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "dynachaos"
        if candidate.is_dir():
            return candidate.parent
    return None


def _runner_env(output_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DYNACHAOS_OUTPUT_ROOT"] = str(output_root)

    src_dir = _repo_src_dir()
    if src_dir is not None:
        current = env.get("PYTHONPATH")
        if current:
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{current}"
        else:
            env["PYTHONPATH"] = str(src_dir)

    return env


def _run_module(module_name: str, output_root: Path) -> None:
    env = _runner_env(output_root)
    cmd = [sys.executable, "-m", module_name]
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Module run failed: {' '.join(cmd)} (exit {proc.returncode})")


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
        missing_cache = [p for p in cache_paths if not p.exists()]
        if missing_cache:
            missing = "\n".join(str(p) for p in missing_cache)
            raise RuntimeError(
                "Smoke profile requires precomputed cache files. "
                "Run paper profile first or provide an output root with existing *.npz files.\n"
                f"Missing:\n{missing}"
            )

    for module_name in spec.modules:
        _run_module(module_name, root)

    missing_outputs = [p for p in output_paths if not p.exists()]
    if missing_outputs:
        missing = "\n".join(str(p) for p in missing_outputs)
        raise RuntimeError(f"Section {section_id} completed with missing outputs:\n{missing}")

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
