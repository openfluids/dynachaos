//! Synthetic intermittency oracle kernels.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (n, x0, eps, a, modulo))]
pub fn pm_type_i_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    eps: f64,
    a: f64,
    modulo: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n(n)?;
    validate_finite(&[x0, eps, a])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = x + eps + a * x * x;
        if modulo {
            x = x.rem_euclid(1.0);
        }
        out.push(x);
    }
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (n, x0, y0, eps, a, theta))]
pub fn pm_type_ii_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    y0: f64,
    eps: f64,
    a: f64,
    theta: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_n(n)?;
    validate_finite(&[x0, y0, eps, a, theta])?;

    let mut x = x0;
    let mut y = y0;
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let r2 = x * x + y * y;
        let growth = 1.0 + eps + a * r2;
        let xr = cos_theta * x - sin_theta * y;
        let yr = sin_theta * x + cos_theta * y;
        x = growth * xr;
        y = growth * yr;
        out.push(x);
        out.push(y);
    }

    let arr = Array2::from_shape_vec((n, 2), out)
        .map_err(|err| PyValueError::new_err(format!("shape error: {err}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
#[pyo3(signature = (n, x0, eps, a))]
pub fn pm_type_iii_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    eps: f64,
    a: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n(n)?;
    validate_finite(&[x0, eps, a])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = -(1.0 + eps) * x - a * x * x * x;
        out.push(x);
    }
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (driver, x0, transverse_lyapunov, noise_scale))]
pub fn on_off_oracle<'py>(
    py: Python<'py>,
    driver: PyReadonlyArray1<'py, f64>,
    x0: f64,
    transverse_lyapunov: f64,
    noise_scale: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let driver_slice = driver.as_slice()?;
    if driver_slice.is_empty() {
        return Err(PyValueError::new_err("driver must be non-empty"));
    }
    validate_finite(&[x0, transverse_lyapunov, noise_scale])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(driver_slice.len());
    for &eta in driver_slice {
        if !eta.is_finite() {
            return Err(PyValueError::new_err("driver must contain only finite values"));
        }
        let multiplier = (transverse_lyapunov + noise_scale * eta).exp();
        x = multiplier * x / (1.0 + x * x);
        out.push(x);
    }
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (n, x0, y0, eps))]
pub fn on_off_skew_logistic_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    y0: f64,
    eps: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_n(n)?;
    validate_finite(&[x0, y0, eps])?;

    let mut x = x0;
    let mut y = y0;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let driver = 4.0 * x * (1.0 - x);
        let multiplier = 4.0 * eps * (1.0 - 2.0 * x);
        y = multiplier * y / (1.0 + y * y);
        x = driver;
        out.push(x);
        out.push(y);
    }

    let arr = Array2::from_shape_vec((n, 2), out)
        .map_err(|err| PyValueError::new_err(format!("shape error: {err}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
#[pyo3(signature = (n, x0, r))]
pub fn logistic_type_i_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    r: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n(n)?;
    validate_finite(&[x0, r])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = r * x * (1.0 - x);
        out.push(x);
    }
    Ok(PyArray1::from_vec(py, out))
}

fn validate_n(n: usize) -> PyResult<()> {
    if n == 0 {
        Err(PyValueError::new_err("n must be positive"))
    } else {
        Ok(())
    }
}

fn validate_finite(values: &[f64]) -> PyResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err("parameters must be finite"))
    }
}
