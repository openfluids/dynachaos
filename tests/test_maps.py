import numpy as np
import pytest

from dynachaos.cml.gcm_clusters import broad_positive_mask
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
    run_transient,
    sample_trajectory,
    trajectory_after_transient,
)
from dynachaos.maps.circle_map import circle_map, circle_map_derivative
from dynachaos.maps.coupled_logistic import coupled_logistic, coupled_logistic_jac
from dynachaos.maps.henon import henon, henon_jac
from dynachaos.maps.modulated_circle import longest_plateau_window, modulated_circle
from dynachaos.maps.primitives import (
    delayed_logistic,
    delayed_logistic_jac,
    logistic,
    logistic_derivative,
)
from dynachaos.maps.torus_doubling import map_I, map_I_jac, map_IV, map_IV_jac


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
