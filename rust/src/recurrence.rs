//! Recurrence Quantification Analysis — diagonal and vertical line extraction.
//!
//! These are the hot inner loops of RQA: scanning an N×N boolean recurrence
//! matrix for consecutive runs of `true` along diagonals (determinism) and
//! columns (laminarity).
//! They validate non-empty square shape only; public RQA semantics such as
//! recurrence-matrix symmetry are enforced by `dynachaos.diagnostics.recurrence.rqa`.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Extract diagonal line lengths from the upper triangle of a recurrence matrix.
///
/// Scans every super-diagonal k = 1, 2, ..., N-1 and records the length of
/// each consecutive run of `true` values that meets the minimum threshold.
///
/// Parameters
/// ----------
/// r : numpy.ndarray, shape (N, N), dtype bool
///     The recurrence matrix.
/// l_min : int
///     Minimum diagonal line length to record (default 2).
///
/// Returns
/// -------
/// numpy.ndarray of int64
///     Array of diagonal line lengths (each ≥ l_min).
#[pyfunction]
#[pyo3(signature = (r, l_min = 2))]
pub fn diagonal_lines<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, bool>,
    l_min: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if l_min == 0 {
        return Err(PyValueError::new_err("l_min must be > 0"));
    }
    let arr = r.as_array();
    let n = arr.shape()[0];
    if n == 0 || arr.shape()[1] != n {
        return Err(PyValueError::new_err("R must be a non-empty square matrix"));
    }
    let mut lengths: Vec<i64> = Vec::new();

    for k in 1..n {
        let diag_len = n - k;
        let mut current: usize = 0;

        for i in 0..diag_len {
            if arr[[i, i + k]] {
                current += 1;
            } else {
                if current >= l_min {
                    lengths.push(current as i64);
                }
                current = 0;
            }
        }
        if current >= l_min {
            lengths.push(current as i64);
        }
    }

    Ok(PyArray1::from_vec(py, lengths))
}

/// Extract vertical line lengths from a recurrence matrix.
///
/// Scans each column j and records consecutive runs of `true` values.
///
/// Parameters
/// ----------
/// r : numpy.ndarray, shape (N, N), dtype bool
///     The recurrence matrix.
/// v_min : int
///     Minimum vertical line length to record (default 2).
///
/// Returns
/// -------
/// numpy.ndarray of int64
///     Array of vertical line lengths (each ≥ v_min).
#[pyfunction]
#[pyo3(signature = (r, v_min = 2))]
pub fn vertical_lines<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, bool>,
    v_min: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if v_min == 0 {
        return Err(PyValueError::new_err("v_min must be > 0"));
    }
    let arr = r.as_array();
    let n = arr.shape()[0];
    if n == 0 || arr.shape()[1] != n {
        return Err(PyValueError::new_err("R must be a non-empty square matrix"));
    }
    let mut lengths: Vec<i64> = Vec::new();

    // Transpose so column scans become row scans (cache-friendly).
    let rt = arr.t();

    for j in 0..n {
        let mut current: usize = 0;

        for i in 0..n {
            if rt[[j, i]] {
                current += 1;
            } else {
                if current >= v_min {
                    lengths.push(current as i64);
                }
                current = 0;
            }
        }
        if current >= v_min {
            lengths.push(current as i64);
        }
    }

    Ok(PyArray1::from_vec(py, lengths))
}
