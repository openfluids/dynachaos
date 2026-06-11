import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from dynachaos.workflow import WorkflowError, run_workflow

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


def _repo_src() -> Path:
    return ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    return env


def _write_config(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _minimal_config(input_name: str, diagnostic: str) -> str:
    return (
        f'{{"input": {{"path": "{input_name}"}}, "output": {{"dir": "out"}}, '
        f'"diagnostics": [{{"name": "{diagnostic}"}}]}}'
    )


def test_workflow_fixture_writes_stable_outputs(tmp_path):
    shutil.copy(DATA / "workflow_signal.npy", tmp_path / "workflow_signal.npy")
    cfg = tmp_path / "workflow_fixture.jsonc"
    cfg.write_text((DATA / "workflow_fixture.jsonc").read_text(encoding="utf-8"), encoding="utf-8")

    paths = run_workflow(cfg)

    assert paths["output_dir"] == tmp_path / "workflow_output"
    assert paths["results"].exists()
    assert paths["metadata"].exists()
    assert paths["summary"].exists()
    results = json.loads(paths["results"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert set(results["diagnostics"]) == {"permutation_entropy", "rqa_streaming"}
    assert "value" in results["diagnostics"]["permutation_entropy"]
    assert "RR" in results["diagnostics"]["rqa_streaming"]["stats"]
    assert metadata["scale_cost"]["signal_length_N"] == 64
    assert "reliability" in metadata
    assert "rqa_streaming" in metadata["reliability"]
    assert "Results JSON: `results.json`" in paths["summary"].read_text(encoding="utf-8")


def test_workflow_generated_benchmark_signal(tmp_path):
    cfg = _write_config(
        tmp_path / "generated.jsonc",
        """
        {
          "input": {"generated": {"name": "logistic", "n": 32, "seed": 7, "a": 1.9}},
          "output": {"dir": "out"},
          "diagnostics": [{"name": "permutation_entropy", "d": 3}]
        }
        """,
    )

    paths = run_workflow(cfg)

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["input"]["kind"] == "generated"
    assert metadata["scale_cost"]["signal_length_N"] == 32


def test_workflow_missing_input_is_explicit(tmp_path):
    cfg = _write_config(
        tmp_path / "missing.jsonc",
        _minimal_config("missing.npy", "permutation_entropy"),
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input file does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing input did not fail")


def test_workflow_invalid_shape_is_explicit(tmp_path):
    shutil.copy(DATA / "workflow_bad_shape.npy", tmp_path / "bad.npy")
    cfg = _write_config(
        tmp_path / "bad_shape.jsonc",
        _minimal_config("bad.npy", "permutation_entropy"),
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input signal must be 1D" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid shape did not fail")


def test_workflow_unsupported_diagnostic_is_explicit(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "unsupported.jsonc",
        _minimal_config("x.npy", "not_a_diagnostic"),
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "unsupported diagnostic name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported diagnostic did not fail")


def test_workflow_dense_rqa_scale_envelope_is_explicit(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "too_dense.jsonc",
        """
        {
          "input": {"path": "x.npy"},
          "output": {"dir": "out"},
          "scale_limits": {"dense_rqa_max_bytes": 128},
          "diagnostics": [{"name": "rqa_dense", "embedding": {"d": 1}, "eps": 0.1}]
        }
        """,
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        msg = str(exc)
        assert "8*N^2" in msg
        assert "rqa_streaming" in msg
        assert "allow_dense_rqa_beyond_envelope" in msg
    else:  # pragma: no cover
        raise AssertionError("oversized dense RQA did not fail")


def test_workflow_rejects_non_boolean_envelope_override(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "string_override.jsonc",
        """
        {
          "input": {"path": "x.npy"},
          "output": {"dir": "out"},
          "scale_limits": {"dense_rqa_max_bytes": 128,
                           "allow_dense_rqa_beyond_envelope": "false"},
          "diagnostics": [{"name": "rqa_dense", "embedding": {"d": 1}, "eps": 0.1}]
        }
        """,
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "must be the JSON boolean" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("string envelope override did not fail")


def test_workflow_rejects_both_path_and_generated(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "dual_input.jsonc",
        """
        {
          "input": {"path": "x.npy",
                    "generated": {"name": "logistic", "n": 32}},
          "output": {"dir": "out"},
          "diagnostics": [{"name": "permutation_entropy"}]
        }
        """,
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "not both" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("dual input source did not fail")


def test_cli_analyze_reports_user_error_without_traceback(tmp_path):
    cfg = _write_config(
        tmp_path / "missing.jsonc",
        _minimal_config("missing.npy", "permutation_entropy"),
    )

    proc = subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", "analyze", str(cfg)],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
        timeout=30,
    )

    assert proc.returncode == 2
    assert "input file does not exist" in proc.stderr
    assert "Traceback" not in proc.stderr
