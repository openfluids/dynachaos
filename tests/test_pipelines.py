import json
import re
from pathlib import Path

import numpy as np
import pytest

from dynachaos.pipelines import runner
from dynachaos.pipelines.registry import get_section, list_sections
from dynachaos.pipelines.runner import run_section, validate_section_cache, validate_section_outputs


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
    with pytest.raises(RuntimeError, match="Section sec02_circle_map is missing expected artifact"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")


def test_smoke_profile_rejects_malformed_cache_before_running_module(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    (section_dir / "devils_staircase.npz").write_text("not an npz", encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("smoke profile should validate caches before running modules")

    monkeypatch.setattr(runner, "_run_module", fail_run)

    with pytest.raises(RuntimeError, match="Section sec02_circle_map has malformed NPZ artifact"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")


def test_smoke_profile_rejects_missing_required_cache_keys_before_running_module(
    tmp_path, monkeypatch
):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0])

    def fail_run(*args, **kwargs):
        raise AssertionError("smoke profile should validate caches before running modules")

    monkeypatch.setattr(runner, "_run_module", fail_run)

    with pytest.raises(RuntimeError, match="missing required NPZ keys: lam"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")


def test_smoke_profile_accepts_valid_cached_outputs_without_recomputation(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    calls = []

    def record_run(module_name, output_root, profile):
        calls.append((module_name, output_root, profile))

    monkeypatch.setattr(runner, "_run_module", record_run)

    outputs = run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")

    assert [path.name for path in outputs] == list(get_section("sec02_circle_map").output_files)
    assert calls == [
        ("dynachaos.maps.circle_map", tmp_path.resolve(), "smoke"),
        ("dynachaos.maps.arnold_tongues", tmp_path.resolve(), "smoke"),
    ]


def test_run_section_writes_opt_in_timing_ledger(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    clock = iter([10.0, 10.5, 20.0, 21.25])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner, "get_rss_mb", lambda: 123.4567)
    monkeypatch.setattr(runner, "_run_module", lambda module_name, output_root, profile: None)

    ledger_path = tmp_path / "perf" / "sections.jsonl"
    run_section(
        "sec02_circle_map",
        output_root=tmp_path,
        profile="smoke",
        timing_ledger=ledger_path,
    )

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        {
            "cache_state": "validated",
            "module": "dynachaos.maps.circle_map",
            "peak_rss_mb": 123.457,
            "profile": "smoke",
            "section_id": "sec02_circle_map",
            "wall_time_s": 0.5,
        },
        {
            "cache_state": "validated",
            "module": "dynachaos.maps.arnold_tongues",
            "peak_rss_mb": 123.457,
            "profile": "smoke",
            "section_id": "sec02_circle_map",
            "wall_time_s": 1.25,
        },
    ]


def test_run_section_timing_ledger_can_come_from_env(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    clock = iter([1.0, 1.25, 2.0, 2.75])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner, "get_rss_mb", lambda: 50.0)
    monkeypatch.setattr(runner, "_run_module", lambda module_name, output_root, profile: None)

    ledger_path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("DYNACHAOS_TIMING_LEDGER", str(ledger_path))
    run_section("sec02_circle_map", output_root=tmp_path, profile="paper")

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [event["cache_state"] for event in events] == ["not_checked", "not_checked"]
    assert [event["wall_time_s"] for event in events] == [0.25, 0.75]


def test_run_section_without_timing_ledger_does_not_touch_timer(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    def fail_timer():
        raise AssertionError("default runs should not collect timing")

    monkeypatch.delenv("DYNACHAOS_TIMING_LEDGER", raising=False)
    monkeypatch.setattr(runner.time, "perf_counter", fail_timer)
    monkeypatch.setattr(runner, "_run_module", lambda module_name, output_root, profile: None)

    run_section("sec02_circle_map", output_root=tmp_path, profile="paper")


def test_section_validators_can_check_cache_and_outputs_without_running_modules(tmp_path):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    cache_paths = validate_section_cache("sec02_circle_map", output_root=tmp_path)
    output_paths = validate_section_outputs("sec02_circle_map", output_root=tmp_path)

    assert [path.name for path in cache_paths] == list(get_section("sec02_circle_map").cache_files)
    assert [path.name for path in output_paths] == list(
        get_section("sec02_circle_map").output_files
    )


def test_section_output_validator_rejects_empty_figure(tmp_path):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    (section_dir / "devils_staircase.png").write_bytes(b"")
    (section_dir / "arnold_tongues.png").write_bytes(b"png")
    (section_dir / "staircase_zoom.png").write_bytes(b"png")

    with pytest.raises(RuntimeError, match="devils_staircase.png.*empty"):
        validate_section_outputs("sec02_circle_map", output_root=tmp_path)
