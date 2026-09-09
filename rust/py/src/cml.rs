//! Python bindings for the coupled-map-lattice kernels.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Logistic CML subsystem Jacobian for sites 0..L-1.
///
/// Returns the row-major flattened L x L block used by the Python caller.
#[pyfunction]
#[pyo3(signature = (x, a, eps, L))]
#[allow(non_snake_case)]
pub fn cml_jacobian_logistic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: f64,
    eps: f64,
    L: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let jacobian = dynachaos_core::cml_jacobian_logistic(x.as_slice()?, a, eps, L)
        .map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, jacobian))
}
