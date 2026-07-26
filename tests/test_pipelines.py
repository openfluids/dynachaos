import json
import re
import sys
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
        "sec12_intermittency",
    )


def test_registry_covers_all_includegraphics_targets():
    paper_tex_path = Path("paper/main.tex")
    if not paper_tex_path.exists():
        pytest.skip("paper sources are not part of the package repository")
    paper_tex = paper_tex_path.read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------
# T2: schema_version contract for attractors.npz
# ---------------------------------------------------------------------------


def test_smoke_gate_rejects_attractors_missing_schema_version(tmp_path):
    """Regression: attractors.npz without schema_version must fail smoke gate (T2).

    Before the fix, schema_version was absent from the NpzContract, so an old-format
    cache (no schema_version key) would pass validate_section_cache even though
    main() would reject and recompute it.
    """
    section_dir = tmp_path / "sec03_transition"
    section_dir.mkdir()
    np.savez_compressed(
        section_dir / "phase_diagram.npz",
        A=[1.0],
        D=[0.1],
        asym=[0.0],
        lyap=[0.1],
        schema_version=[4],
    )
    # attractors.npz missing the schema_version key (old pre-v4 cache format)
    np.savez_compressed(
        section_dir / "attractors.npz",
        A_values=[1.0],
        labels=["sync"],
        initial_states=[[0.5, 0.5]],
        x_limits=[0.0, 1.0],
        y_limits=[0.0, 1.0],
        D=[0.1],
    )
    np.savez_compressed(
        section_dir / "basins.npz",
        x=[0.5],
        y=[0.5],
        basin=[0],
        A=[1.0],
        D=[0.1],
    )
    with pytest.raises(RuntimeError, match="missing required NPZ keys.*schema_version"):
        validate_section_cache("sec03_transition", output_root=tmp_path)


# ---------------------------------------------------------------------------
# T1: child RSS measurement
# ---------------------------------------------------------------------------


def test_run_module_returns_child_peak_rss_mb(monkeypatch, tmp_path):
    """_run_module returns peak RSS of the child process via os.wait4 (T1 unit test).

    Uses a monkeypatched Popen and wait4 to inject a known rusage and verify
    the KB-to-MB conversion matches get_rss_mb's convention.
    """
    import os as _os

    from dynachaos.pipelines import runner as _runner

    if not hasattr(_os, "wait4"):
        pytest.skip("os.wait4 not available on this platform")

    # ru_maxrss units are platform-dependent: bytes on macOS, KiB on Linux.
    # Build the fake value in the units the running platform actually reports,
    # so this asserts the conversion rather than the developer's platform.
    _MIB = 1024 * 1024
    _RSS_RAW_256_MIB = 256 * _MIB if sys.platform == "darwin" else 256 * 1024

    class FakeRusage:
        ru_maxrss = _RSS_RAW_256_MIB

    class FakePopen:
        pid = 9999
        returncode = None

        def __init__(self, cmd, env=None):
            pass

    def fake_wait4(pid, options):
        return (pid, 0, FakeRusage())  # exit_status=0 → clean exit

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(_runner.os, "wait4", fake_wait4)
    if hasattr(_os, "waitstatus_to_exitcode"):
        monkeypatch.setattr(_runner.os, "waitstatus_to_exitcode", lambda x: 0)

    rss = _runner._run_module("dummy.module", tmp_path, "paper")

    assert rss == pytest.approx(256.0, abs=0.01)


@pytest.mark.skipif(not hasattr(__import__("os"), "wait4"), reason="requires POSIX wait4")
def test_ledger_records_child_rss_not_orchestrator(tmp_path, monkeypatch):
    """Ledger peak_rss_mb reflects child allocation, not orchestrator RSS (T1 integration).

    A real subprocess that allocates 100 MB is spawned via a monkeypatched _run_module;
    the ledgered value must clearly exceed what the allocation-free orchestrator
    would report.

    The child writes to every page. ``np.zeros`` is backed by ``calloc``, which
    returns lazily-mapped pages that are never resident until touched, so a
    zeros-only child does not reliably raise peak RSS on any platform.
    """
    import os as _os

    from dynachaos.pipelines import runner as _runner

    section_dir = tmp_path / "sec03_transition"
    section_dir.mkdir()
    np.savez_compressed(
        section_dir / "phase_diagram.npz",
        A=[1.0],
        D=[0.1],
        asym=[0.0],
        lyap=[0.1],
        schema_version=[4],
    )
    np.savez_compressed(
        section_dir / "attractors.npz",
        A_values=[1.0],
        labels=["sync"],
        initial_states=[[0.5, 0.5]],
        x_limits=[0.0, 1.0],
        y_limits=[0.0, 1.0],
        D=[0.1],
        schema_version=[4],
    )
    np.savez_compressed(
        section_dir / "basins.npz",
        x=[0.5],
        y=[0.5],
        basin=[0],
        A=[1.0],
        D=[0.1],
    )
    for png in ("phase_diagram.png", "attractors.png", "basins.png"):
        (section_dir / png).write_bytes(b"png")

    def make_allocating_module(alloc_mib):
        """Build a fake _run_module whose child allocates ``alloc_mib`` and reports its RSS."""

        def allocating_module(module_name, output_root, profile):
            import subprocess as _sp
            import sys as _sys

            proc = _sp.Popen(
                [
                    _sys.executable,
                    "-c",
                    # ones() writes every element, faulting the pages in; zeros()
                    # would be lazily mapped and never become resident.
                    f"import numpy as np; x = np.ones({alloc_mib} * 1024 * 1024 // 8); "
                    "x[::4096] += 1",
                ],
            )
            pid, status, rusage = _os.wait4(proc.pid, 0)
            divisor = 1024 * 1024 if _sys.platform == "darwin" else 1024
            return rusage.ru_maxrss / divisor

        return allocating_module

    def ledger_rss(alloc_mib, ledger_name):
        monkeypatch.setattr(_runner, "_run_module", make_allocating_module(alloc_mib))
        ledger_path = tmp_path / ledger_name
        run_section(
            "sec03_transition", output_root=tmp_path, profile="paper", timing_ledger=ledger_path
        )
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
        assert len(events) == 1
        return events[0]["peak_rss_mb"]

    # Measure the ledger against two different child allocations. Comparing a
    # single measurement against the orchestrator's own RSS -- the obvious
    # approach, and what this test used to do -- cannot work: getrusage(
    # RUSAGE_SELF).ru_maxrss is a high-water mark over the whole process
    # lifetime, so inside a pytest session it accumulates whatever the heaviest
    # earlier test allocated. It climbs from ~72 MB to ~130 MB on Linux and to
    # ~465 MB on macOS purely as a function of test ordering, and at ~130 MB it
    # coincides with the child's own footprint. Any threshold against a moving
    # reference is really a threshold against test-suite history.
    #
    # The difference between two child allocations has no such dependence. If
    # the ledger were recording the orchestrator, both runs would report the
    # same number and the delta would collapse to zero.
    small_mib, large_mib = 100, 300
    rss_small = ledger_rss(small_mib, "timing_small.jsonl")
    rss_large = ledger_rss(large_mib, "timing_large.jsonl")

    # Each child faults in its full array, so its peak RSS must clear that
    # allocation. A value below this means nothing resembling the child was
    # measured.
    assert rss_small >= 50.0, (
        f"Expected child RSS >= 50 MB (numpy + {small_mib} MB alloc), got {rss_small:.1f} MB"
    )
    # The ledger must follow the child's allocation, not a constant. Allow a
    # generous band: the delta is dominated by the 200 MB difference in touched
    # pages, but allocator behaviour and the interpreter baseline add slop.
    observed_delta = rss_large - rss_small
    expected_delta = large_mib - small_mib
    assert 0.5 * expected_delta < observed_delta < 1.5 * expected_delta, (
        f"Ledger reported {rss_small:.1f} MB for a {small_mib} MB child and "
        f"{rss_large:.1f} MB for a {large_mib} MB one — a delta of "
        f"{observed_delta:.1f} MB where ~{expected_delta} MB was expected. The "
        f"ledger does not track the child's allocation; it may be recording the "
        f"orchestrator's own rusage."
    )
