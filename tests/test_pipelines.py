import re
from pathlib import Path

import pytest

from dynachaos.pipelines.registry import get_section, list_sections
from dynachaos.pipelines.runner import run_section


def test_section_registry_contains_all_expected_sections():
    sections = list_sections()
    assert sections == (
        "sec02_circle_map",
        "sec03_transition",
        "sec04_doubling",
        "sec05_oscillation",
        "sec06_three_torus",
        "sec07_fractalization",
        "sec08_sti",
        "sec09_pattern",
        "sec10_gcm",
        "sec11_diagnostics",
    )


def test_registry_covers_all_includegraphics_targets():
    paper_tex = Path("paper/main.tex").read_text(encoding="utf-8")
    refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", paper_tex)
    include_targets = set(refs)

    declared_pngs = set()
    for section_id in list_sections():
        spec = get_section(section_id)
        for rel in spec.output_files:
            if rel.endswith(".png"):
                declared_pngs.add(f"{section_id}/{rel}")

    missing = include_targets - declared_pngs
    assert not missing, f"Missing includegraphics coverage for: {sorted(missing)}"


def test_smoke_profile_requires_precomputed_cache(tmp_path):
    with pytest.raises(RuntimeError, match="Smoke profile requires precomputed cache files"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")
