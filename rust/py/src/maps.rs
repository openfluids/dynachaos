//! Python bindings for the low-dimensional map-iteration kernels.

use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;

/// Delayed-logistic attractor tile: one `(x, y)` trajectory per `D` value.
///
/// Mirrors `compute_attractor` in `dynachaos.maps.delayed_logistic`: for each
/// `d` in `d_values` the kernel iterates `n_transient` steps of
/// `x' = A * x + (1 - A) * (1 - D * y * y)`, `y' = x` from `state0`, then
/// records `n_plot` states.
///
/// Returns an array of shape `(n_D, n_plot, 2)`. A `D` whose orbit diverges
/// (any `|state| > 1e10`) yields a block of `NaN` rows, matching the `None`
/// the Python helper returns for that parameter value.
#[pyfunction]
#[pyo3(signature = (A, d_values, n_transient, n_plot, state0))]
#[allow(non_snake_case)]
pub fn delayed_logistic_attractor_tile<'py>(
    py: Python<'py>,
    A: f64,
    d_values: PyReadonlyArray1<'py, f64>,
    n_transient: usize,
    n_plot: usize,
    state0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let flat = dynachaos_core::delayed_logistic_attractor_tile(
        A,
        d_values.as_slice()?,
        n_transient,
        n_plot,
        state0.as_slice()?,
    )
    .map_err(crate::core_to_py)?;
    let n_d = d_values.as_slice()?.len();
    let tile = Array3::from_shape_vec((n_d, n_plot, 2), flat)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, tile))
}

/// Torus-doubling attractor tile: one trajectory of map I or map IV.
///
/// Mirrors `iterate_map` in `dynachaos.maps.torus_doubling`: `map_kind` 1 is
/// map (I) on `(X, Y, Z)`, `map_kind` 4 is map (IV) on `(X, Y, Z, W)`, and the
/// kernel iterates `n_transient` steps from `state0`, then records up to
/// `n_plot` states.
///
/// Returns an array of shape `(n_produced, dim)`. Divergence (any
/// `|state| > 1e10`) stops the record and returns the samples produced so
/// far, possibly zero rows — the same partial trajectory `iterate_map`
/// returns with `allow_partial = True`.
#[pyfunction]
#[pyo3(signature = (map_kind, A, D, n_transient, n_plot, state0))]
#[allow(non_snake_case)]
pub fn torus_doubling_attractor_tile<'py>(
    py: Python<'py>,
    map_kind: u8,
    A: f64,
    D: f64,
    n_transient: usize,
    n_plot: usize,
    state0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat = dynachaos_core::torus_doubling_attractor_tile(
        map_kind,
        A,
        D,
        n_transient,
        n_plot,
        state0.as_slice()?,
    )
    .map_err(crate::core_to_py)?;
    let dim = match map_kind {
        1 => 3,
        _ => 4,
    };
    let tile = Array2::from_shape_vec((flat.len() / dim, dim), flat)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, tile))
}

/// Modulated-circle rotation numbers: one `(rho_theta, rho_phi)` pair per `D`.
///
/// Mirrors `rotation_numbers` in `dynachaos.maps.modulated_circle`: for each
/// `d` in `d_values` the kernel accumulates the unwrapped increments of
/// `theta' = theta + A * sin(2 pi theta) + D + eps * sin(2 pi phi)`,
/// `phi' = phi + C` from `state0 = (theta0, phi0)` for `n_transient` steps,
/// then returns the mean increments over the next `n_iter` steps.
///
/// Returns an array of shape `(n_D, 2)`.
#[pyfunction]
#[pyo3(signature = (A, C, d_values, eps, n_transient, n_iter, state0))]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn modulated_circle_rotation_tile<'py>(
    py: Python<'py>,
    A: f64,
    C: f64,
    d_values: PyReadonlyArray1<'py, f64>,
    eps: f64,
    n_transient: usize,
    n_iter: usize,
    state0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat = dynachaos_core::modulated_circle_rotation_tile(
        A,
        C,
        d_values.as_slice()?,
        eps,
        n_transient,
        n_iter,
        state0.as_slice()?,
    )
    .map_err(crate::core_to_py)?;
    let n_d = d_values.as_slice()?.len();
    let tile = Array2::from_shape_vec((n_d, 2), flat)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, tile))
}
