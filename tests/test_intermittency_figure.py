import numpy as np

from dynachaos.diagnostics import intermittency_figure
from dynachaos.pipelines.registry import get_section


def test_intermittency_figure_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "type_i_intermittency.npz"

    payload = intermittency_figure.compute(output_path)

    assert output_path.exists()
    assert tuple(payload) == intermittency_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [5])
    np.testing.assert_array_equal(payload["seed"], [20260601])
    np.testing.assert_equal(
        payload["source_file"][0],
        "src/dynachaos/diagnostics/intermittency_figure.py",
    )
    np.testing.assert_array_less(payload["logistic_tail_r"], payload["logistic_mechanism_r"])
    np.testing.assert_array_equal(payload["logistic_period"], [3])
    np.testing.assert_allclose(payload["logistic_laminar_percentile"], [70.0])
    np.testing.assert_equal(payload["logistic_laminar_lengths"].size, 6228)
    np.testing.assert_equal(payload["logistic_f3_return_points"].shape, (4000, 2))
    np.testing.assert_equal(payload["logistic_f3_channel_points"].shape, (177, 2))
    np.testing.assert_equal(
        _count_x_clusters(payload["logistic_f3_channel_points"]),
        3,
    )
    np.testing.assert_equal(payload["normal_form_eps"].size, 8)
    np.testing.assert_equal(payload["normal_form_mean_lengths"].size, 8)
    np.testing.assert_equal(payload["lorenz_return_points"].shape, (274, 2))
    np.testing.assert_equal(payload["lorenz_channel_points"].shape, (82, 2))
    np.testing.assert_allclose(payload["type_i_tail_alpha"], [-1.50748972])
    np.testing.assert_allclose(payload["type_i_vuong_z"], [1.31575131])
    np.testing.assert_allclose(payload["logistic_f3_channel_slope"], [1.00042829])
    np.testing.assert_allclose(payload["normal_form_beta"], [0.4919801])
    np.testing.assert_allclose(payload["lorenz_channel_slope"], [0.98549932])
    np.testing.assert_array_less(-1.7, payload["type_i_tail_alpha"])
    np.testing.assert_array_less(payload["type_i_tail_alpha"], -1.3)
    np.testing.assert_array_less(0.0, payload["type_i_vuong_z"])
    np.testing.assert_array_less(0.35, payload["normal_form_beta"])
    np.testing.assert_array_less(payload["normal_form_beta"], 0.65)
    np.testing.assert_array_less(0.8, payload["lorenz_channel_slope"])
    np.testing.assert_array_less(payload["lorenz_channel_slope"], 1.2)


def test_intermittency_figure_plot_writes_png(tmp_path):
    payload = intermittency_figure.compute(None)
    output_path = tmp_path / "type_i_intermittency.png"

    result = intermittency_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_intermittency_figure_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("type_i_intermittency.npz")

    assert spec.modules == ("dynachaos.diagnostics.intermittency_figure",)
    assert spec.cache_files == ("type_i_intermittency.npz",)
    assert spec.output_files == (
        "type_i_intermittency.npz",
        "type_i_intermittency.png",
    )
    assert set(contract_keys).issubset(intermittency_figure.REQUIRED_KEYS)


def _count_x_clusters(points):
    x = np.sort(np.asarray(points)[:, 0])
    return 1 + int(np.count_nonzero(np.diff(x) > 0.05))
