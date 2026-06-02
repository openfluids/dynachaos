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
    np.testing.assert_array_equal(payload["type_i_period"], [3])
    np.testing.assert_allclose(payload["type_i_laminar_percentile"], [70.0])
    np.testing.assert_equal(payload["type_i_laminar_lengths"].size, 6228)
    np.testing.assert_equal(payload["type_i_return_points"].shape, (73531, 2))
    np.testing.assert_equal(payload["type_i_channel_points"].shape, (7354, 2))
    np.testing.assert_allclose(payload["on_off_transverse_lyapunov"], [-0.025])
    np.testing.assert_allclose(payload["on_off_threshold_percentile"], [70.0])
    np.testing.assert_equal(payload["on_off_laminar_lengths"].size, 40)
    np.testing.assert_equal(payload["on_off_burst_lengths"].size, 40)
    np.testing.assert_equal(payload["lorenz_section_points"].shape, (5, 2))
    np.testing.assert_allclose(payload["type_i_tail_alpha"], [-1.50748972])
    np.testing.assert_allclose(payload["type_i_vuong_z"], [1.31575131])
    np.testing.assert_allclose(payload["type_i_channel_slope"], [0.99982474])
    np.testing.assert_allclose(payload["on_off_laminar_alpha"], [-1.51334166])
    np.testing.assert_allclose(payload["on_off_burst_alpha"], [-1.05644466])
    np.testing.assert_allclose(payload["on_off_symmetry_p"], [0.76593145])
    np.testing.assert_allclose(payload["lorenz_channel_slope"], [0.83403938])
    np.testing.assert_array_less(-1.7, payload["type_i_tail_alpha"])
    np.testing.assert_array_less(payload["type_i_tail_alpha"], -1.3)
    np.testing.assert_array_less(0.0, payload["type_i_vuong_z"])
    np.testing.assert_array_less(-1.7, payload["on_off_laminar_alpha"])
    np.testing.assert_array_less(payload["on_off_laminar_alpha"], -1.3)
    np.testing.assert_array_less(-1.2, payload["on_off_burst_alpha"])
    np.testing.assert_array_less(payload["on_off_burst_alpha"], -0.8)


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
