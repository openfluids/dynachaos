from pathlib import Path

import numpy as np

from dynachaos.io.paths import safe_load
from dynachaos.pipelines.registry import get_section
from dynachaos.pipelines.runner import run_section

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_FIGURES = REPO_ROOT / "figures"
DETERMINISM_SECTIONS = ("sec12_intermittency",)


def _assert_npz_matches(generated_path: Path, committed_path: Path):
    with safe_load(generated_path) as generated, safe_load(committed_path) as committed:
        assert set(generated.files) == set(committed.files)
        for key in generated.files:
            actual = generated[key]
            expected = committed[key]
            assert actual.shape == expected.shape, key
            assert actual.dtype == expected.dtype, key
            if np.issubdtype(actual.dtype, np.floating):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(actual, expected)


def test_representative_figure_caches_recompute_deterministically(tmp_path):
    for section_id in DETERMINISM_SECTIONS:
        run_section(section_id, output_root=tmp_path, profile="paper", recompute=True)
        spec = get_section(section_id)
        for cache_file in spec.cache_files:
            _assert_npz_matches(
                tmp_path / section_id / cache_file,
                COMMITTED_FIGURES / section_id / cache_file,
            )
