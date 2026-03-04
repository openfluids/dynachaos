//! Multifractal partition moments over dyadic box scales.
//!
//! For each box size `r` and moment order `q`, computes:
//! - `log_z(r, q) = ln(sum_i p_i(r)^q)`
//! - `alpha_num(r, q) = sum_i mu_i(r, q) ln p_i(r)`
//! - `f_num(r, q) = sum_i mu_i(r, q) ln mu_i(r, q)`
//!
//! where `p_i(r)` are box probabilities and
//! `mu_i(r, q) = p_i(r)^q / sum_j p_j(r)^q`.
//!
//! These are the canonical moments used to recover `tau(q)`, `D_q`,
//! `alpha(q)`, and `f(alpha)` via log-log regressions in Python.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute multifractal canonical moments for a 2D nonnegative measure field.
///
/// Parameters
/// ----------
/// field : numpy.ndarray of float64, shape (ny, nx)
///     Nonnegative measure field.
/// box_sizes : numpy.ndarray of int64, shape (n_scales,)
///     Box side lengths. Each scale uses non-overlapping boxes and truncates
///     edge remainders.
/// q_values : numpy.ndarray of float64, shape (n_q,)
///     Moment orders q.
///
/// Returns
/// -------
/// tuple
///     (log_z, alpha_num, f_num, ln_scales), where:
///     - log_z: ndarray, shape (n_scales, n_q)
///     - alpha_num: ndarray, shape (n_scales, n_q)
///     - f_num: ndarray, shape (n_scales, n_q)
///     - ln_scales: ndarray, shape (n_scales,)
#[pyfunction]
#[pyo3(signature = (field, box_sizes, q_values))]
pub fn multifractal_moments<'py>(
    py: Python<'py>,
    field: PyReadonlyArray2<'py, f64>,
    box_sizes: PyReadonlyArray1<'py, i64>,
    q_values: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    #![allow(clippy::type_complexity)]
    let arr = field.as_array();
    let ny = arr.shape()[0];
    let nx = arr.shape()[1];
    let field_slice = arr.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("field must be C-contiguous")
    })?;

    let scales = box_sizes.as_slice()?;
    let qs = q_values.as_slice()?;
    let n_scales = scales.len();
    let n_q = qs.len();

    let mut ln_scales = vec![f64::NAN; n_scales];
    let mut log_z = vec![f64::NAN; n_scales * n_q];
    let mut alpha_num = vec![f64::NAN; n_scales * n_q];
    let mut f_num = vec![f64::NAN; n_scales * n_q];

    let total_mass: f64 = field_slice.iter().copied().sum();
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "field must have a positive finite total mass",
        ));
    }
    if field_slice.iter().any(|&v| !v.is_finite() || v < 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "field must contain only finite nonnegative values",
        ));
    }

    for (si, &b_i64) in scales.iter().enumerate() {
        if b_i64 <= 0 {
            continue;
        }
        let b = b_i64 as usize;
        let n_by = ny / b;
        let n_bx = nx / b;
        if n_by == 0 || n_bx == 0 {
            continue;
        }
        ln_scales[si] = (b as f64).ln();

        let n_boxes = n_by * n_bx;
        let mut probs: Vec<f64> = Vec::with_capacity(n_boxes);
        let mut used_mass = 0.0f64;

        for by in 0..n_by {
            let y0 = by * b;
            for bx in 0..n_bx {
                let x0 = bx * b;
                let mut mass = 0.0f64;
                for yy in 0..b {
                    let row = y0 + yy;
                    let base = row * nx;
                    for xx in 0..b {
                        mass += field_slice[base + x0 + xx];
                    }
                }
                if mass > 0.0 {
                    probs.push(mass);
                    used_mass += mass;
                }
            }
        }

        if probs.is_empty() || !used_mass.is_finite() || used_mass <= 0.0 {
            continue;
        }
        let inv_used_mass = 1.0 / used_mass;
        for p in &mut probs {
            *p *= inv_used_mass;
        }

        for (qi, &q) in qs.iter().enumerate() {
            let idx = si * n_q + qi;
            if !q.is_finite() {
                continue;
            }

            if (q - 1.0).abs() < 1e-12 {
                let mut shannon = 0.0f64;
                for &p in &probs {
                    shannon += p * p.ln();
                }
                // Z_1(r) = sum_i p_i = 1 by normalization.
                log_z[idx] = 0.0;
                alpha_num[idx] = shannon;
                f_num[idx] = shannon;
                continue;
            }

            let mut z = 0.0f64;
            for &p in &probs {
                z += p.powf(q);
            }
            if !z.is_finite() || z <= 0.0 {
                continue;
            }

            let ln_z = z.ln();
            let mut a = 0.0f64;
            let mut f = 0.0f64;
            for &p in &probs {
                let p_q = p.powf(q);
                let mu = p_q / z;
                if mu > 0.0 && mu.is_finite() {
                    a += mu * p.ln();
                    f += mu * mu.ln();
                }
            }

            log_z[idx] = ln_z;
            alpha_num[idx] = a;
            f_num[idx] = f;
        }
    }

    let log_z_arr = Array2::from_shape_vec((n_scales, n_q), log_z).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("shape error log_z: {e}"))
    })?;
    let alpha_arr = Array2::from_shape_vec((n_scales, n_q), alpha_num).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("shape error alpha_num: {e}"))
    })?;
    let f_arr = Array2::from_shape_vec((n_scales, n_q), f_num).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("shape error f_num: {e}"))
    })?;

    Ok((
        PyArray2::from_owned_array(py, log_z_arr),
        PyArray2::from_owned_array(py, alpha_arr),
        PyArray2::from_owned_array(py, f_arr),
        PyArray1::from_vec(py, ln_scales),
    ))
}
