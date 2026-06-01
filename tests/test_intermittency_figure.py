import numpy as np

from dynachaos.diagnostics import intermittency_figure
from dynachaos.pipelines.registry import get_section


def test_intermittency_figure_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "intermittency_diagnostics.npz"

    payload = intermittency_figure.compute(output_path)

    assert output_path.exists()
    assert tuple(payload) == intermittency_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [1])
    np.testing.assert_array_equal(payload["seed"], [20260601])
    np.testing.assert_equal(
        payload["source_file"][0],
        "src/dynachaos/diagnostics/intermittency_figure.py",
    )
    np.testing.assert_equal(payload["type_i_laminar_lengths"].size, 146)
    np.testing.assert_equal(payload["type_i_return_points"].shape, (1098, 2))
    np.testing.assert_equal(payload["type_i_channel_points"].shape, (110, 2))
    np.testing.assert_equal(payload["on_off_laminar_lengths"].size, 106)
    np.testing.assert_equal(payload["on_off_burst_lengths"].size, 107)
    np.testing.assert_equal(payload["lorenz_section_points"].shape, (5, 2))
    np.testing.assert_allclose(payload["type_i_tail_alpha"], [-3.94100986])
    np.testing.assert_allclose(payload["type_i_vuong_z"], [-5.92558068])
    np.testing.assert_allclose(payload["type_i_channel_slope"], [1.00056036])
    np.testing.assert_allclose(payload["on_off_burst_alpha"], [-3.00368458])
    np.testing.assert_allclose(payload["on_off_symmetry_p"], [0.99482730])
    np.testing.assert_allclose(payload["lorenz_channel_slope"], [0.83403938])


def test_intermittency_figure_plot_writes_png(tmp_path):
    payload = intermittency_figure.compute(None)
    output_path = tmp_path / "intermittency_diagnostics.png"

    result = intermittency_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_intermittency_figure_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("intermittency_diagnostics.npz")

    assert spec.modules == ("dynachaos.diagnostics.intermittency_figure",)
    assert spec.cache_files == ("intermittency_diagnostics.npz",)
    assert spec.output_files == (
        "intermittency_diagnostics.npz",
        "intermittency_diagnostics.png",
    )
    assert set(contract_keys).issubset(intermittency_figure.REQUIRED_KEYS)
