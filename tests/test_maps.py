import numpy as np
import pytest

from dynachaos.cml.gcm_clusters import broad_positive_mask, compute_clusters, compute_collective
from dynachaos.cml.primitives import (
    cluster_labels_by_tolerance,
    cml_jacobian_subblock_logistic,
    cml_step,
    cml_step_logistic,
    cml_step_logistic_batch,
    gcm_step,
    sustained_positive_mask,
)
from dynachaos.diagnostics.compare_all_helpers import (
    load_or_compute_npz,
    sweep_pair_metric,
    sweep_scalar_metric,
)
from dynachaos.maps._iter import (
    iterate_unwrapped,
    run_animation_sweep,
    run_transient,
    sample_trajectory,
    trajectory_after_transient,
)
from dynachaos.maps.circle_map import circle_map, circle_map_derivative
from dynachaos.maps.coupled_logistic import (
    ATTRACTOR_CASES,
    PHASE_REQUIRED_KEYS,
    PhaseDiagramPayload,
    coupled_logistic,
    coupled_logistic_jac,
)
from dynachaos.maps.coupled_logistic import (
    compute_attractors as compute_coupled_attractors,
)
from dynachaos.maps.coupled_logistic import (
    compute_basins as compute_coupled_basins,
)
from dynachaos.maps.coupled_logistic import (
    compute_phase_diagram as compute_coupled_phase_diagram,
)
from dynachaos.maps.delayed_logistic import (
    compute_attractors as compute_delayed_attractors,
)
from dynachaos.maps.delayed_logistic import (
    compute_locking_sequence as compute_delayed_locking_sequence,
)
from dynachaos.maps.delayed_logistic import (
    compute_lyapunov_spectrum as compute_delayed_lyapunov_spectrum,
)
from dynachaos.maps.henon import henon, henon_jac
from dynachaos.maps.modulated_circle import longest_plateau_window, modulated_circle
from dynachaos.maps.primitives import (
    delayed_logistic,
    delayed_logistic_jac,
    logistic,
    logistic_derivative,
)
from dynachaos.maps.torus_doubling import (
    compute_map_I,
    compute_map_IV,
    compute_map_IV_lyapunov,
    map_I,
    map_I_jac,
    map_IV,
    map_IV_jac,
)


def test_circle_map_range_and_derivative():
    theta = 0.37
    out = circle_map(theta, A=0.1, D=0.25)
    deriv = circle_map_derivative(theta, A=0.1, D=0.25)

    assert 0.0 <= out < 1.0
    assert np.isfinite(deriv)


def test_coupled_logistic_shapes():
    state = np.array([0.2, -0.1], dtype=np.float64)
    out = coupled_logistic(state, A=1.2, D=0.08)
    jac = coupled_logistic_jac(state, A=1.2, D=0.08)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_coupled_logistic_preserves_diagonal():
    state = np.array([0.37, 0.37], dtype=np.float64)
    out = coupled_logistic(state, A=1.3, D=0.1)

    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], logistic(0.37, 1.3))


def test_coupled_logistic_respects_exchange_symmetry():
    state = np.array([0.2, -0.4], dtype=np.float64)
    swapped = state[::-1].copy()

    out = coupled_logistic(state, A=1.25, D=0.1)
    out_swapped = coupled_logistic(swapped, A=1.25, D=0.1)

    np.testing.assert_allclose(out_swapped, out[::-1])


def test_compute_coupled_phase_diagram_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "phase_diagram.npz"

    payload = compute_coupled_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=2,
        n_sample=3,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["asym"].shape == (2, 3)
    assert payload["lyap"].shape == (2, 3)
    assert int(payload["schema_version"][0]) == 2
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["A"], payload["A"])
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["asym"], payload["asym"])
        np.testing.assert_allclose(saved["lyap"], payload["lyap"])


def test_coupled_phase_payload_round_trip(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    payload = PhaseDiagramPayload(
        A=np.array([0.8, 1.0], dtype=np.float64),
        D=np.array([0.0, 0.1], dtype=np.float64),
        asym=np.zeros((2, 2), dtype=np.float64),
        lyap=np.ones((2, 2), dtype=np.float64),
    )
    np.savez_compressed(path, **payload.to_npz())

    with np.load(path, allow_pickle=False) as saved:
        loaded = PhaseDiagramPayload.from_npz(saved)

    assert set(PHASE_REQUIRED_KEYS) == set(payload.to_npz())
    np.testing.assert_allclose(loaded.A, payload.A)
    np.testing.assert_allclose(loaded.D, payload.D)
    np.testing.assert_allclose(loaded.asym, payload.asym)
    np.testing.assert_allclose(loaded.lyap, payload.lyap)
    assert loaded.schema_version == payload.schema_version


def test_coupled_phase_payload_rejects_stale_schema(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8], dtype=np.float64),
        D=np.array([0.0], dtype=np.float64),
        asym=np.zeros((1, 1), dtype=np.float64),
        lyap=np.zeros((1, 1), dtype=np.float64),
        schema_version=np.array([1], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match="stale phase diagram cache"):
            PhaseDiagramPayload.from_npz(saved)


def test_coupled_phase_payload_rejects_missing_keys(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8], dtype=np.float64),
        D=np.array([0.0], dtype=np.float64),
        asym=np.zeros((1, 1), dtype=np.float64),
        schema_version=np.array([2], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(KeyError, match="missing keys: lyap"):
            PhaseDiagramPayload.from_npz(saved)


def test_coupled_phase_payload_rejects_grid_shape_mismatch(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8, 1.0], dtype=np.float64),
        D=np.array([0.0, 0.1], dtype=np.float64),
        asym=np.zeros((2, 1), dtype=np.float64),
        lyap=np.zeros((2, 2), dtype=np.float64),
        schema_version=np.array([2], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match="grid shape mismatch"):
            PhaseDiagramPayload.from_npz(saved)


def test_compute_coupled_phase_diagram_accepts_scalar_sweeps():
    payload = compute_coupled_phase_diagram(
        A_values=1.0,
        D_values=0.1,
        n_transient=1,
        n_sample=1,
        output_path=None,
    )

    assert payload["A"].shape == (1,)
    assert payload["D"].shape == (1,)
    assert payload["asym"].shape == (1, 1)
    assert payload["lyap"].shape == (1, 1)


def test_compute_coupled_attractors_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "attractors.npz"
    cases = ATTRACTOR_CASES[:2]

    payload = compute_coupled_attractors(
        cases=cases,
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["x_0"].shape == (4,)
    assert payload["y_0"].shape == (4,)
    assert payload["x_1"].shape == (4,)
    assert payload["y_1"].shape == (4,)
    assert int(payload["schema_version"][0]) == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["A_values"], payload["A_values"])
        np.testing.assert_allclose(saved["x_0"], payload["x_0"])
        np.testing.assert_allclose(saved["y_1"], payload["y_1"])


def test_compute_coupled_basins_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "basins.npz"

    payload = compute_coupled_basins(
        n_grid=4,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=output_path,
    )

    assert payload["x"].shape == (4,)
    assert payload["y"].shape == (4,)
    assert payload["basin"].shape == (4, 4)
    assert payload["basin"].dtype == np.int8
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["x"], payload["x"])
        np.testing.assert_array_equal(saved["basin"], payload["basin"])


# ---------------------------------------------------------------------------
# Logistic map primitives
# ---------------------------------------------------------------------------


def test_logistic_known_values():
    """f(0, a) = 1 for all a; f(1, 2) = 1 - 2 = -1."""
    assert logistic(0.0, 1.5) == 1.0
    assert logistic(1.0, 2.0) == -1.0


def test_logistic_vectorized():
    x = np.array([0.0, 0.5, 1.0])
    result = logistic(x, 2.0)
    expected = 1.0 - 2.0 * x * x
    np.testing.assert_allclose(result, expected)


def test_logistic_derivative_values():
    """f'(x) = -2ax."""
    assert logistic_derivative(0.0, 1.5) == 0.0
    assert logistic_derivative(1.0, 2.0) == -4.0


def test_logistic_derivative_vectorized():
    x = np.array([-1.0, 0.0, 0.5])
    result = logistic_derivative(x, 1.5)
    expected = -2.0 * 1.5 * x
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Delayed logistic map
# ---------------------------------------------------------------------------


def test_delayed_logistic_shape_and_finite():
    state = np.array([0.5, 0.3])
    out = delayed_logistic(state, A=0.3, D=1.8)
    jac = delayed_logistic_jac(state, A=0.3, D=1.8)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_delayed_logistic_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    state = np.array([0.5, 0.3])
    A, D = 0.3, 1.8
    jac = delayed_logistic_jac(state, A, D)

    eps = 1e-7
    for j in range(2):
        state_plus = state.copy()
        state_plus[j] += eps
        state_minus = state.copy()
        state_minus[j] -= eps
        fd_col = (delayed_logistic(state_plus, A, D) - delayed_logistic(state_minus, A, D)) / (
            2.0 * eps
        )
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_compute_delayed_attractors_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "attractors.npz"

    payload = compute_delayed_attractors(
        D_values=np.array([1.55, 1.65]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.55_x"].shape == (4,)
    assert payload["D_1.65_y"].shape == (4,)
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.55_x"], payload["D_1.55_x"])


def test_compute_delayed_attractors_rejects_rounded_key_collisions():
    with pytest.raises(ValueError, match="unique"):
        compute_delayed_attractors(
            D_values=np.array([1.554, 1.555]),
            n_transient=1,
            n_plot=1,
            output_path=None,
        )


def test_compute_delayed_lyapunov_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "lyapunov_vs_D.npz"

    payload = compute_delayed_lyapunov_spectrum(
        D_values=np.array([1.5, 1.6]),
        n_iter=4,
        n_transient=2,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["D"].shape == (2,)
    assert payload["spectra"].shape == (2, 2)
    assert np.all(np.isfinite(payload["spectra"]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["spectra"], payload["spectra"])


def test_compute_delayed_locking_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "locking_sequence.npz"

    payload = compute_delayed_locking_sequence(
        D_values=np.array([1.86, 1.88]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.860_x"].shape == (4,)
    assert payload["D_1.880_y"].shape == (4,)
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.860_x"], payload["D_1.860_x"])


def test_compute_delayed_locking_rejects_rounded_key_collisions():
    with pytest.raises(ValueError, match="unique"):
        compute_delayed_locking_sequence(
            D_values=np.array([1.86041, 1.86042]),
            n_transient=1,
            n_plot=1,
            output_path=None,
        )


# ---------------------------------------------------------------------------
# Torus doubling maps
# ---------------------------------------------------------------------------


def test_map_I_shape():
    state = np.array([0.5, 0.3, 0.4])
    out = map_I(state, A=0.4, D=2.0)
    jac = map_I_jac(state, A=0.4, D=2.0)

    assert out.shape == (3,)
    assert jac.shape == (3, 3)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_map_IV_shape():
    state = np.array([0.5, 0.3, 0.4, 0.2])
    out = map_IV(state, A=0.3, D=1.5)
    jac = map_IV_jac(state, A=0.3, D=1.5)

    assert out.shape == (4,)
    assert jac.shape == (4, 4)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_compute_torus_map_I_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_I_attractors.npz"

    payload = compute_map_I(
        D_values=np.array([2.11, 2.16]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_2.11_traj"].shape[1] == 3
    assert payload["D_2.16_traj"].shape[1] == 3
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_2.11_traj"], payload["D_2.11_traj"])


def test_compute_torus_map_IV_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_IV_attractors.npz"

    payload = compute_map_IV(
        D_values=np.array([1.515, 1.5206]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.515_traj"].shape[1] == 4
    assert payload["D_1.5206_traj"].shape[1] == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.515_traj"], payload["D_1.515_traj"])


def test_compute_torus_map_IV_lyapunov_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_IV_lyapunov.npz"

    payload = compute_map_IV_lyapunov(
        D_values=np.array([1.5, 1.51]),
        n_iter=4,
        n_transient=2,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["D"].shape == (2,)
    assert payload["spectra"].shape == (2, 4)
    assert np.all(np.isfinite(payload["spectra"]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["spectra"], payload["spectra"])


def test_run_animation_sweep_returns_and_writes_payload(tmp_path):
    output_path = tmp_path / "animation.npz"

    def iterate_fn(param):
        return np.column_stack(
            (
                np.full(3, param, dtype=np.float64),
                np.arange(3, dtype=np.float64),
            )
        )

    payload = run_animation_sweep(
        iterate_fn,
        np.array([0.1, 0.2]),
        output_path,
        n_plot=3,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["param_values"], np.array([0.1, 0.2]))
    np.testing.assert_allclose(payload["all_x"], np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
    np.testing.assert_allclose(payload["all_y"], np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["all_x"], payload["all_x"])


# ---------------------------------------------------------------------------
# Modulated circle map
# ---------------------------------------------------------------------------


def test_modulated_circle_range():
    state = np.array([0.3, 0.7])
    out = modulated_circle(state, A=0.15, C=0.618, D=0.25, eps=0.05)

    assert out.shape == (2,)
    assert 0.0 <= out[0] < 1.0
    assert 0.0 <= out[1] < 1.0


def test_longest_plateau_window_selects_widest_run():
    D = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rho = np.array([0.00, 0.25, 0.2502, 0.40, 0.25, 0.2501, 0.2502])

    window = longest_plateau_window(D, rho, target=0.25, tol=5e-4)

    assert window == (0.4, 0.6)


def test_broad_positive_mask_rejects_short_spikes():
    values = np.array([-0.1, 0.03, -0.02, 0.04, 0.05, 0.06, 0.07, -0.01])

    mask = broad_positive_mask(values, threshold=0.02, min_run=3)

    np.testing.assert_array_equal(
        mask,
        np.array([False, False, False, True, True, True, True, False]),
    )


def test_sustained_positive_mask_matches_legacy_alias():
    values = np.array([-0.1, 0.03, -0.02, 0.04, 0.05, 0.06, 0.07, -0.01])

    np.testing.assert_array_equal(
        sustained_positive_mask(values, threshold=0.02, min_run=3),
        broad_positive_mask(values, threshold=0.02, min_run=3),
    )


def test_compute_clusters_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "gcm_clusters.npz"

    payload = compute_clusters(
        n_sites=8,
        n_transient=2,
        n_record=4,
        seed=123,
        output_path=output_path,
    )

    assert payload["cluster_labels"].shape == (4, 8)
    assert payload["x_record"].shape == (4, 8)
    assert int(payload["N"][0]) == 8
    assert int(payload["n_transient"][0]) == 2
    assert int(payload["n_record"][0]) == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_array_equal(saved["cluster_labels"], payload["cluster_labels"])
        np.testing.assert_allclose(saved["x_record"], payload["x_record"])


def test_compute_collective_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "collective_lyapunov.npz"
    a_values = np.array([1.4, 1.6])

    payload = compute_collective(
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=4,
        renorm_interval=2,
        seed=123,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["a_values"], a_values)
    assert payload["lyap_c"].shape == (2,)
    assert np.all(np.isfinite(payload["lyap_c"]))
    assert int(payload["N"][0]) == 8
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["a_values"], payload["a_values"])
        np.testing.assert_allclose(saved["lyap_c"], payload["lyap_c"])


def test_cluster_labels_by_tolerance_groups_sorted_runs():
    values = np.array([0.10, 0.1000002, 0.50, 0.5000001, 0.90])

    labels = cluster_labels_by_tolerance(values, tol=1e-5)

    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 2]))


def test_iterate_unwrapped_scalar_matches_manual_accumulation():
    value = 0.1
    for _ in range(5):
        value += 0.25 + 0.1 * value

    helper = iterate_unwrapped(0.1, lambda x: 0.25 + 0.1 * x, 5)

    np.testing.assert_allclose(helper, value)


def test_run_transient_and_sample_helpers_record_post_step_states():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    after = run_transient(state, step_fn, 2)
    np.testing.assert_allclose(after, np.array([2.0, 3.0]))

    samples = sample_trajectory(after, step_fn, 3)
    np.testing.assert_allclose(
        samples,
        np.array([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]),
    )

    combined = trajectory_after_transient(state, step_fn, 2, 3)
    np.testing.assert_allclose(combined, samples)


def test_cml_step_matches_manual_generic_update():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    eps = 0.2

    def f(arr):
        return 2.0 * arr

    def g(arr):
        return arr + 1.0

    out = cml_step(x, f, g, eps)
    manual = f(x) + eps / 2.0 * (np.roll(g(x), -1) + np.roll(g(x), 1) - 2.0 * g(x))

    np.testing.assert_allclose(out, manual)


def test_cml_step_default_preserves_flattened_roll_behavior_for_2d_input():
    x = np.array([[0.1, -0.2], [0.3, -0.4]])
    eps = 0.2

    def f(arr):
        return 2.0 * arr

    def g(arr):
        return arr + 1.0

    out = cml_step(x, f, g, eps)
    gx = g(x)
    manual = f(x) + eps / 2.0 * (np.roll(gx, -1) + np.roll(gx, 1) - 2.0 * gx)

    np.testing.assert_allclose(out, manual)


def test_cml_step_logistic_axis_argument_enables_rowwise_topology():
    x = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.5]])
    eps = 0.2

    explicit = cml_step_logistic(x, 1.6, eps, axis=1)
    rowwise = np.vstack([cml_step_logistic(row, 1.6, eps) for row in x])

    np.testing.assert_allclose(explicit, rowwise)


def test_cml_step_logistic_batch_matches_rowwise_update():
    x = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.5]])
    a_col = np.array([[1.5], [1.8]])
    eps = 0.2

    batch = cml_step_logistic_batch(x, a_col, eps)
    rowwise = np.vstack(
        [cml_step_logistic(x[idx], float(a_col[idx, 0]), eps) for idx in range(len(x))]
    )

    np.testing.assert_allclose(batch, rowwise)


def test_cml_jacobian_subblock_logistic_matches_manual_matrix():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    a = 1.5
    eps = 0.2
    L = 3

    J = cml_jacobian_subblock_logistic(x, a, eps, L)
    dfx = logistic_derivative(x, a)
    expected = np.array(
        [
            [(1.0 - eps) * dfx[0], (eps / 2.0) * dfx[1], 0.0],
            [(eps / 2.0) * dfx[0], (1.0 - eps) * dfx[1], (eps / 2.0) * dfx[2]],
            [0.0, (eps / 2.0) * dfx[1], (1.0 - eps) * dfx[2]],
        ]
    )

    np.testing.assert_allclose(J, expected)


def test_gcm_step_matches_manual_formula():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    a = 1.7
    eps = 0.15

    out = gcm_step(x, a, eps)
    fx = logistic(x, a)
    manual = (1.0 - eps) * fx + eps * np.mean(fx)

    np.testing.assert_allclose(out, manual)


def test_run_transient_returns_none_when_diverged():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    def diverged_fn(s):
        return s[0] > 2.5

    out = run_transient(state, step_fn, 5, diverged_fn=diverged_fn)
    assert out is None


def test_sample_trajectory_allow_partial_returns_prefix():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    def diverged_fn(s):
        return s[0] > 2.5

    samples = sample_trajectory(
        state,
        step_fn,
        5,
        diverged_fn=diverged_fn,
        allow_partial=True,
    )

    np.testing.assert_allclose(samples, np.array([[1.0, 2.0], [2.0, 3.0]]))


def test_iterate_unwrapped_vectorized_updates_elementwise():
    state = np.array([0.0, 1.0, 2.0])
    out = iterate_unwrapped(state, lambda s: np.array([1.0, -1.0, 0.5]), 3)
    np.testing.assert_allclose(out, np.array([3.0, -2.0, 3.5]))


def test_sweep_metric_helpers_return_expected_arrays():
    values = np.array([1.0, 2.0, 3.0])

    def series_fn(v):
        return np.array([v, 2.0 * v])

    def scalar_metric(s):
        return float(np.sum(s))

    def pair_metric(s):
        return float(np.min(s)), float(np.max(s))

    scalar = sweep_scalar_metric(values, series_fn, scalar_metric)
    first, second = sweep_pair_metric(values, series_fn, pair_metric)

    np.testing.assert_allclose(scalar, np.array([3.0, 6.0, 9.0]))
    np.testing.assert_allclose(first, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(second, np.array([2.0, 4.0, 6.0]))


def test_load_or_compute_npz_computes_when_missing(tmp_path):
    path = tmp_path / "sample.npz"

    def compute_fn():
        np.savez_compressed(path, values=np.array([1.0, 2.0]))

    data = load_or_compute_npz(path, "sample", compute_fn)

    np.testing.assert_allclose(data["values"], np.array([1.0, 2.0]))


def test_load_or_compute_npz_recomputes_when_required_keys_missing(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez_compressed(path, stale=np.array([0.0]))
    calls = 0

    def compute_fn():
        nonlocal calls
        calls += 1
        np.savez_compressed(path, values=np.array([3.0, 4.0]))

    data = load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))

    assert calls == 1
    np.testing.assert_allclose(data["values"], np.array([3.0, 4.0]))


def test_load_or_compute_npz_raises_when_compute_leaves_required_keys_missing(tmp_path):
    path = tmp_path / "sample.npz"

    def compute_fn():
        np.savez_compressed(path, other=np.array([1.0]))

    with pytest.raises(KeyError, match="missing required keys"):
        load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))


# ---------------------------------------------------------------------------
# Henon map
# ---------------------------------------------------------------------------


def test_henon_shape_and_finite():
    """Output shapes and finiteness."""
    state = np.array([0.1, 0.1])
    out = henon(state)
    jac = henon_jac(state)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_henon_known_value():
    """At origin: x' = 1, y' = 0."""
    out = henon(np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_henon_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    state = np.array([0.63, -0.19])
    a, b = 1.4, 0.3
    jac = henon_jac(state, a, b)

    eps = 1e-7
    for j in range(2):
        state_plus = state.copy()
        state_plus[j] += eps
        state_minus = state.copy()
        state_minus[j] -= eps
        fd_col = (henon(state_plus, a, b) - henon(state_minus, a, b)) / (2.0 * eps)
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_compute_clusters_seed_controls_rng_determinism():
    first = compute_clusters(seed=7, n_sites=8, n_transient=2, n_record=4, output_path=None)
    second = compute_clusters(seed=7, n_sites=8, n_transient=2, n_record=4, output_path=None)

    np.testing.assert_array_equal(first["cluster_labels"], second["cluster_labels"])
    np.testing.assert_array_equal(first["x_record"], second["x_record"])

    different_seed = compute_clusters(seed=99, n_sites=8, n_transient=2, n_record=4, output_path=None)

    assert not np.array_equal(
        first["cluster_labels"], different_seed["cluster_labels"]
    ) or not np.array_equal(
        first["x_record"], different_seed["x_record"]
    ), "compute_clusters should produce different stochastic payloads for different seeds"


def test_compute_collective_seed_controls_rng_determinism():
    a_values = np.array([1.4, 1.6, 1.8])

    first = compute_collective(
        seed=7,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )
    second = compute_collective(
        seed=7,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )

    np.testing.assert_array_equal(first["lyap_c"], second["lyap_c"])

    different_seed = compute_collective(
        seed=99,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )

    assert not np.array_equal(
        first["lyap_c"], different_seed["lyap_c"]
    ), "compute_collective should produce different lyap_c values for different seeds"


# ---------------------------------------------------------------------------
# Golden-value regression tests for compute-aggregate functions
# ---------------------------------------------------------------------------


# params: D_values=[1.5, 1.6], n_iter=20, n_transient=5, seed=N/A
def test_compute_delayed_lyapunov_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "lyapunov_golden.npz"
    payload = compute_delayed_lyapunov_spectrum(
        D_values=np.array([1.5, 1.6]),
        n_iter=20,
        n_transient=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["D"], np.array([1.5, 1.6]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["spectra"],
        np.array(
            [
                [0.075066870334274, 0.07028722946827429],
                [0.0970755525312533, 0.09294026515025561],
            ]
        ),
        rtol=1e-9,
    )
    assert payload["spectra"].shape == (2, 2)
    assert np.all(np.isfinite(payload["spectra"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["spectra"], payload["spectra"], rtol=1e-9)


# params: D_values=[1.5, 1.51], n_iter=20, n_transient=5, seed=N/A
def test_compute_map_IV_lyapunov_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "map_iv_lyap_golden.npz"
    payload = compute_map_IV_lyapunov(
        D_values=np.array([1.5, 1.51]),
        n_iter=20,
        n_transient=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["D"], np.array([1.5, 1.51]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["spectra"],
        np.array(
            [
                [
                    0.16140671993531416,
                    0.04742302325520868,
                    0.04219401941837471,
                    -0.07178967726173077,
                ],
                [
                    0.16270287321983554,
                    0.04828748938330477,
                    0.04415935310015298,
                    -0.07025603073637762,
                ],
            ]
        ),
        rtol=1e-9,
    )
    assert payload["spectra"].shape == (2, 4)
    assert np.all(np.isfinite(payload["spectra"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["spectra"], payload["spectra"], rtol=1e-9)


# params: a=1.55, eps=0.1, n_sites=4, n_transient=5, n_record=3, seed=7
def test_compute_clusters_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "gcm_clusters_golden.npz"
    payload = compute_clusters(
        a=1.55,
        eps=0.1,
        n_sites=4,
        n_transient=5,
        n_record=3,
        seed=7,
        output_path=output_path,
    )

    np.testing.assert_array_equal(
        payload["cluster_labels"],
        np.array([[0, 3, 2, 1], [3, 0, 1, 2], [0, 1, 3, 2]]),
    )
    np.testing.assert_allclose(
        payload["x_record"],
        np.array(
            [
                [
                    -0.04545209715542766,
                    0.9563681048236234,
                    0.6832212378395283,
                    0.6648412697350994,
                ],
                [
                    0.9263795270955607,
                    -0.3466612849138426,
                    0.27808764055024543,
                    0.31265203806958264,
                ],
                [
                    -0.24185543228176498,
                    0.7876615172785664,
                    0.8474251456092492,
                    0.8189412528994947,
                ],
            ]
        ),
        rtol=1e-9,
    )

    with np.load(output_path) as saved:
        np.testing.assert_array_equal(saved["cluster_labels"], payload["cluster_labels"])
        np.testing.assert_allclose(saved["x_record"], payload["x_record"], rtol=1e-9)


# params: n_sites=4, a_values=[1.4, 1.6], n_transient=5, n_measure=10, seed=7
def test_compute_collective_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "collective_golden.npz"
    payload = compute_collective(
        n_sites=4,
        a_values=np.array([1.4, 1.6]),
        n_transient=5,
        n_measure=10,
        renorm_interval=5,
        seed=7,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["a_values"], np.array([1.4, 1.6]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["lyap_c"],
        np.array([-0.11970016300271111, 0.17056225610801404]),
        rtol=1e-9,
    )
    assert payload["lyap_c"].shape == (2,)
    assert np.all(np.isfinite(payload["lyap_c"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["lyap_c"], payload["lyap_c"], rtol=1e-9)


# params: n_grid=4, n_transient=2, reference_transient=2, period=2, seed=N/A
def test_compute_coupled_basins_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "basins_golden.npz"
    payload = compute_coupled_basins(
        n_grid=4,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=output_path,
    )

    np.testing.assert_allclose(
        payload["x"],
        np.array([-1.0, -0.33333333333333337, 0.33333333333333326, 1.0]),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        payload["y"],
        np.array([-1.0, -0.33333333333333337, 0.33333333333333326, 1.0]),
        rtol=1e-9,
    )
    np.testing.assert_array_equal(
        payload["basin"],
        np.array(
            [[0, 1, 1, 1], [2, 0, 1, 1], [2, 2, 0, 1], [2, 2, 2, 0]],
            dtype=np.int8,
        ),
    )

    with np.load(output_path) as saved:
        np.testing.assert_array_equal(saved["basin"], payload["basin"])


# params: A_values=[0.8, 1.0, 1.2], D_values=[0.0, 0.1], n_transient=5, seed=N/A
def test_compute_coupled_phase_diagram_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "phase_golden.npz"
    payload = compute_coupled_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=5,
        n_sample=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(
        payload["asym"],
        np.array(
            [
                [
                    8.2293861882232425e-03,
                    5.9876141248649259e-05,
                    3.0112671401466472e-02,
                ],
                [
                    8.0161532921512221e-02,
                    1.3210878773309020e-02,
                    4.5589216524537955e-03,
                ],
            ]
        ),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload["lyap"],
        np.array(
            [
                [-0.0575617156373394, -4.103877162577709, 0.07108327581839047],
                [-0.04912671322212415, -0.5509515979462181, 0.08136671975106122],
            ]
        ),
        rtol=1e-9,
    )

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["asym"], payload["asym"], rtol=1e-9)
        np.testing.assert_allclose(saved["lyap"], payload["lyap"], rtol=1e-9)
