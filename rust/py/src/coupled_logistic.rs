//! Python bindings for coupled logistic map kernels.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute basin labels for a fixed two-site coupled logistic map.
///
/// Python keeps ownership of reference-orbit construction, x/y grids, payload
/// writing, and plotting. This kernel only accelerates the grid transient and
/// classification loop.
#[pyfunction]
#[pyo3(signature = (x_values, y_values, A, D, n_transient, ref_a))]
#[allow(non_snake_case)]
pub fn coupled_logistic_basin_grid<'py>(
    py: Python<'py>,
    x_values: PyReadonlyArray1<'py, f64>,
    y_values: PyReadonlyArray1<'py, f64>,
    A: f64,
    D: f64,
    n_transient: usize,
    ref_a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let basin = dynachaos_core::coupled_logistic_basin_grid(
        x_values.as_slice()?,
        y_values.as_slice()?,
        A,
        D,
        n_transient,
        ref_a.as_array(),
    )
    .map_err(crate::core_to_py)?;
    Ok(PyArray2::from_owned_array(py, basin))
}
