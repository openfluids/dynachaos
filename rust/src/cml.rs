//! Coupled-map lattice kernels.

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
    let l = L;
    let x_slice = x.as_slice()?;
    let n = x_slice.len();
    if l < 1 || l > n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "L must satisfy 1 <= L <= N (got L={l}, N={n})"
        )));
    }

    let matrix_len = l.checked_mul(l).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("L is too large for an L x L matrix")
    })?;
    let mut jacobian = vec![0.0_f64; matrix_len];
    let diagonal_scale = 1.0 - eps;
    let off_diagonal_scale = eps / 2.0;

    for i in 0..l {
        let row = i * l;
        jacobian[row + i] = diagonal_scale * logistic_derivative(x_slice[i], a);

        let i_left = if i == 0 { n - 1 } else { i - 1 };
        if i_left < l {
            jacobian[row + i_left] = off_diagonal_scale * logistic_derivative(x_slice[i_left], a);
        }

        let i_right = (i + 1) % n;
        if i_right < l {
            jacobian[row + i_right] = off_diagonal_scale * logistic_derivative(x_slice[i_right], a);
        }
    }

    Ok(PyArray1::from_vec(py, jacobian))
}

#[inline]
fn logistic_derivative(x: f64, a: f64) -> f64 {
    -2.0 * a * x
}
