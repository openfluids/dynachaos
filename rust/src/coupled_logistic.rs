//! Coupled logistic map kernels.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

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
    let x_slice = x_values.as_slice()?;
    let y_slice = y_values.as_slice()?;
    let ref_arr = ref_a.as_array();
    let ref_shape = ref_arr.shape();
    if ref_shape.len() != 2 || ref_shape[1] != 2 {
        return Err(PyValueError::new_err("ref_a must have shape (period, 2)"));
    }

    let nx = x_slice.len();
    let ny = y_slice.len();
    let period = ref_shape[0];
    let basin_len = nx
        .checked_mul(ny)
        .ok_or_else(|| PyValueError::new_err("basin grid is too large"))?;

    let x_owned = x_slice.to_vec();
    let y_owned = y_slice.to_vec();
    let ref_pairs: Vec<(f64, f64)> = (0..period)
        .map(|k| (ref_arr[[k, 0]], ref_arr[[k, 1]]))
        .collect();

    let mut basin = Vec::with_capacity(basin_len);
    for &y0 in &y_owned {
        py.check_signals()?;
        let mut row = py.detach(|| basin_row(&x_owned, y0, A, D, n_transient, &ref_pairs));
        basin.append(&mut row);
    }
    py.check_signals()?;

    let basin_arr = Array2::from_shape_vec((ny, nx), basin)
        .map_err(|e| PyRuntimeError::new_err(format!("shape error basin: {e}")))?;
    Ok(PyArray2::from_owned_array(py, basin_arr))
}

#[inline]
fn logistic(x: f64, a: f64) -> f64 {
    1.0 - a * x * x
}

fn basin_row(
    x_values: &[f64],
    y0: f64,
    a: f64,
    d: f64,
    n_transient: usize,
    ref_pairs: &[(f64, f64)],
) -> Vec<i8> {
    x_values
        .par_iter()
        .map(|&x0| basin_point(x0, y0, a, d, n_transient, ref_pairs))
        .collect()
}

fn basin_point(
    x0: f64,
    y0: f64,
    a: f64,
    d: f64,
    n_transient: usize,
    ref_pairs: &[(f64, f64)],
) -> i8 {
    let mut x = x0;
    let mut y = y0;
    for _ in 0..n_transient {
        let old_x = x;
        let old_y = y;
        let new_x = logistic(old_x, a) + d * (old_y - old_x);
        let new_y = logistic(old_y, a) + d * (old_x - old_y);
        if new_x.abs() > 100.0 || new_y.abs() > 100.0 {
            x = f64::NAN;
            y = f64::NAN;
        } else {
            x = new_x;
            y = new_y;
        }
    }

    if x.is_nan() {
        return -1;
    }

    let mut dist_a = f64::INFINITY;
    let mut dist_b = f64::INFINITY;
    for &(ax, ay) in ref_pairs {
        let da = squared_distance(x, y, ax, ay);
        let db = squared_distance(x, y, ay, ax);
        if da < dist_a {
            dist_a = da;
        }
        if db < dist_b {
            dist_b = db;
        }
    }

    if dist_a < dist_b {
        1
    } else if dist_b < dist_a {
        2
    } else {
        0
    }
}

#[inline]
fn squared_distance(x: f64, y: f64, ref_x: f64, ref_y: f64) -> f64 {
    let dx = x - ref_x;
    let dy = y - ref_y;
    dx * dx + dy * dy
}
