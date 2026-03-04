//! Cao's method and False Nearest Neighbors with brute-force NN search.
//!
//! Both algorithms require nearest-neighbor queries in delay-coordinate
//! embeddings.  For d ≤ ~10 (typical for these algorithms), brute-force
//! search with Chebyshev early exit is competitive with KD-tree and avoids
//! Python/C boundary crossings per query.
//!
//! References:
//! - Cao, L. (1997) Physica D 110(1-2), 43-50.
//! - Kennel, M.B. et al. (1992) Phys. Rev. A 45(6), 3403-3411.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Embed a 1D series into d-dimensional delay coordinates.
///
/// Returns a flat Vec of length M*d (row-major), where M = N - (d-1)*tau.
#[inline]
fn embed(x: &[f64], d: usize, tau: usize) -> Option<(Vec<f64>, usize)> {
    if d == 0 {
        return None;
    }
    let n = x.len();
    let lag = (d - 1).checked_mul(tau)?;
    if n <= lag {
        return None;
    }
    let m = n - lag;
    let total = m.checked_mul(d)?;
    let mut out = vec![0.0f64; total];
    for i in 0..m {
        for j in 0..d {
            out[i * d + j] = x[i + j * tau];
        }
    }
    Some((out, m))
}

/// Find nearest neighbor for each point using brute-force Chebyshev distance.
///
/// Returns (nn_indices, nn_distances) each of length n.
/// Respects a Theiler window: |i - j| > theiler_window required.
fn nearest_neighbor_chebyshev(
    points: &[f64],
    n: usize,
    dim: usize,
    theiler_window: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut nn_idx = vec![0usize; n];
    let mut nn_dist = vec![f64::INFINITY; n];

    for i in 0..n {
        let pi = &points[i * dim..(i + 1) * dim];
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0usize;

        for j in 0..n {
            if i == j {
                continue;
            }
            // Theiler window check
            if (i as isize - j as isize).unsigned_abs() <= theiler_window {
                continue;
            }

            // Chebyshev distance with early exit
            let pj = &points[j * dim..(j + 1) * dim];
            let mut d_max = 0.0f64;
            let mut early_exit = false;
            for k in 0..dim {
                let diff = (pi[k] - pj[k]).abs();
                if diff > best_dist {
                    early_exit = true;
                    break;
                }
                if diff > d_max {
                    d_max = diff;
                }
            }
            if early_exit {
                continue;
            }
            if d_max < best_dist {
                best_dist = d_max;
                best_j = j;
            }
        }
        nn_idx[i] = best_j;
        nn_dist[i] = best_dist;
    }

    (nn_idx, nn_dist)
}

/// Find nearest neighbor using Euclidean distance (brute-force).
fn nearest_neighbor_euclidean(
    points: &[f64],
    n: usize,
    dim: usize,
    theiler_window: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut nn_idx = vec![0usize; n];
    let mut nn_dist_sq = vec![f64::INFINITY; n];

    for i in 0..n {
        let pi = &points[i * dim..(i + 1) * dim];
        let mut best_sq = f64::INFINITY;
        let mut best_j = 0usize;

        for j in 0..n {
            if i == j {
                continue;
            }
            if (i as isize - j as isize).unsigned_abs() <= theiler_window {
                continue;
            }

            let pj = &points[j * dim..(j + 1) * dim];
            let mut sq = 0.0f64;
            let mut early_exit = false;
            for k in 0..dim {
                let diff = pi[k] - pj[k];
                sq += diff * diff;
                if sq > best_sq {
                    early_exit = true;
                    break;
                }
            }
            if early_exit {
                continue;
            }
            if sq < best_sq {
                best_sq = sq;
                best_j = j;
            }
        }
        nn_idx[i] = best_j;
        nn_dist_sq[i] = best_sq;
    }

    // Convert squared distances to actual distances
    let nn_dist: Vec<f64> = nn_dist_sq.iter().map(|d| d.sqrt()).collect();
    (nn_idx, nn_dist)
}

/// Moving-average smoothing helper for 1D diagnostics.
#[inline]
fn smooth_series(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if window <= 1 || n < window {
        return values.to_vec();
    }
    let left = window / 2;
    let right = window - 1 - left;
    let mut padded = Vec::with_capacity(n + left + right);
    padded.extend(std::iter::repeat(values[0]).take(left));
    padded.extend(values.iter().copied());
    padded.extend(std::iter::repeat(values[n - 1]).take(right));

    let mut out = vec![0.0f64; n];
    let inv = 1.0f64 / window as f64;
    for i in 0..n {
        let mut acc = 0.0f64;
        for v in &padded[i..i + window] {
            acc += *v;
        }
        out[i] = acc * inv;
    }
    out
}

/// Compute Cao's E(d) and E*(d) statistics for d = 1..d_max.
///
/// Parameters
/// ----------
/// x : numpy.ndarray of float64, shape (N,)
///     Scalar time series.
/// tau : int
///     Time delay.
/// d_max : int
///     Maximum embedding dimension.
/// theiler_window : int
///     Minimum temporal separation for NN search.
///
/// Returns
/// -------
/// (E, E_star) : tuple of numpy.ndarray, each shape (d_max,)
///     Raw E(d) and E*(d) values.  Python computes E1 = E(d+1)/E(d).
#[pyfunction]
#[pyo3(signature = (x, tau, d_max, theiler_window = 0))]
pub fn cao_statistic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    tau: usize,
    d_max: usize,
    theiler_window: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    #![allow(clippy::type_complexity)]
    if tau == 0 {
        return Err(PyValueError::new_err("tau must be >= 1"));
    }
    if d_max == 0 {
        return Err(PyValueError::new_err("d_max must be >= 1"));
    }

    let arr = x.as_slice()?;
    let n = arr.len();

    let mut e_values = vec![0.0f64; d_max];
    let mut e_star_values = vec![0.0f64; d_max];

    for d in 1..=d_max {
        let dt = match d.checked_mul(tau) {
            Some(v) => v,
            None => {
                e_values[d - 1] = f64::NAN;
                e_star_values[d - 1] = f64::NAN;
                continue;
            }
        };
        let m = n.saturating_sub(dt);
        if m < 2 {
            e_values[d - 1] = f64::NAN;
            e_star_values[d - 1] = f64::NAN;
            continue;
        }

        // Embed in d dimensions: use x[0..m+(d-1)*tau]
        let end_d = match (d - 1).checked_mul(tau).and_then(|v| m.checked_add(v)) {
            Some(v) if v <= n => v,
            _ => {
                e_values[d - 1] = f64::NAN;
                e_star_values[d - 1] = f64::NAN;
                continue;
            }
        };
        let Some((y1, m1)) = embed(&arr[..end_d], d, tau) else {
            e_values[d - 1] = f64::NAN;
            e_star_values[d - 1] = f64::NAN;
            continue;
        };
        debug_assert_eq!(m1, m);

        // Embed in d+1 dimensions: use x[0..m+d*tau]
        let end_d1 = match d.checked_mul(tau).and_then(|v| m.checked_add(v)) {
            Some(v) if v <= n => v,
            _ => {
                e_values[d - 1] = f64::NAN;
                e_star_values[d - 1] = f64::NAN;
                continue;
            }
        };
        let Some((y2, m2)) = embed(&arr[..end_d1], d + 1, tau) else {
            e_values[d - 1] = f64::NAN;
            e_star_values[d - 1] = f64::NAN;
            continue;
        };
        debug_assert_eq!(m2, m);

        // Find NN in d-dimensional space (Chebyshev)
        let (nn_idx, nn_dist_d) = nearest_neighbor_chebyshev(&y1, m, d, theiler_window);

        // Compute a(i) and a_star(i)
        let mut sum_a = 0.0f64;
        let mut sum_a_star = 0.0f64;
        let mut count = 0usize;

        for i in 0..m {
            let j = nn_idx[i];
            let dist_d = nn_dist_d[i];

            // Skip points with no valid neighbor (Theiler window too large)
            if dist_d.is_infinite() {
                continue;
            }

            // Distance in (d+1)-dimensional space (Chebyshev)
            let mut dist_d1 = 0.0f64;
            for k in 0..(d + 1) {
                let diff = (y2[i * (d + 1) + k] - y2[j * (d + 1) + k]).abs();
                if diff > dist_d1 {
                    dist_d1 = diff;
                }
            }

            if dist_d > 0.0 {
                sum_a += dist_d1 / dist_d;
            } else {
                sum_a += 1.0; // avoid division by zero
            }

            // E*(d): |x[i + d*tau] - x[nn + d*tau]| (last coordinate diff)
            let last_diff = (y2[i * (d + 1) + d] - y2[j * (d + 1) + d]).abs();
            sum_a_star += last_diff;

            count += 1;
        }

        e_values[d - 1] = if count > 0 { sum_a / count as f64 } else { f64::NAN };
        e_star_values[d - 1] = if count > 0 {
            sum_a_star / count as f64
        } else {
            f64::NAN
        };
    }

    Ok((
        PyArray1::from_vec(py, e_values),
        PyArray1::from_vec(py, e_star_values),
    ))
}

/// Compute False Nearest Neighbor fractions for d = 1..d_max.
///
/// Parameters
/// ----------
/// x : numpy.ndarray of float64, shape (N,)
///     Scalar time series.
/// tau : int
///     Time delay.
/// d_max : int
///     Maximum embedding dimension.
/// r_tol : float
///     Threshold for Test I (default 15.0).
/// a_tol : float
///     Threshold for Test II (default 2.0).
/// theiler_window : int
///     Minimum temporal separation for NN search.
///
/// Returns
/// -------
/// (f1, f2, f3) : tuple of numpy.ndarray, each shape (d_max,)
///     FNN fractions for Test I, Test II, and union.
#[pyfunction]
#[pyo3(signature = (x, tau, d_max, r_tol = 15.0, a_tol = 2.0, theiler_window = 0))]
pub fn fnn_statistic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    tau: usize,
    d_max: usize,
    r_tol: f64,
    a_tol: f64,
    theiler_window: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    #![allow(clippy::type_complexity)]
    if tau == 0 {
        return Err(PyValueError::new_err("tau must be >= 1"));
    }
    if d_max == 0 {
        return Err(PyValueError::new_err("d_max must be >= 1"));
    }

    let arr = x.as_slice()?;
    let n = arr.len();
    if n == 0 {
        let nan_vec = vec![f64::NAN; d_max];
        return Ok((
            PyArray1::from_vec(py, nan_vec.clone()),
            PyArray1::from_vec(py, nan_vec.clone()),
            PyArray1::from_vec(py, nan_vec),
        ));
    }

    // Compute sigma (std of full series)
    let mean: f64 = arr.iter().sum::<f64>() / n as f64;
    let var: f64 = arr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let sigma = var.sqrt();

    let mut f1_values = vec![0.0f64; d_max];
    let mut f2_values = vec![0.0f64; d_max];
    let mut f3_values = vec![0.0f64; d_max];

    for d in 1..=d_max {
        let dt = match d.checked_mul(tau) {
            Some(v) => v,
            None => {
                f1_values[d - 1] = f64::NAN;
                f2_values[d - 1] = f64::NAN;
                f3_values[d - 1] = f64::NAN;
                continue;
            }
        };
        let m = n.saturating_sub(dt);
        if m < 2 {
            f1_values[d - 1] = f64::NAN;
            f2_values[d - 1] = f64::NAN;
            f3_values[d - 1] = f64::NAN;
            continue;
        }

        // Embed in d dimensions
        let end_d = match (d - 1).checked_mul(tau).and_then(|v| m.checked_add(v)) {
            Some(v) if v <= n => v,
            _ => {
                f1_values[d - 1] = f64::NAN;
                f2_values[d - 1] = f64::NAN;
                f3_values[d - 1] = f64::NAN;
                continue;
            }
        };
        let Some((y1, m1)) = embed(&arr[..end_d], d, tau) else {
            f1_values[d - 1] = f64::NAN;
            f2_values[d - 1] = f64::NAN;
            f3_values[d - 1] = f64::NAN;
            continue;
        };
        debug_assert_eq!(m1, m);

        // Embed in d+1 dimensions
        let end_d1 = match d.checked_mul(tau).and_then(|v| m.checked_add(v)) {
            Some(v) if v <= n => v,
            _ => {
                f1_values[d - 1] = f64::NAN;
                f2_values[d - 1] = f64::NAN;
                f3_values[d - 1] = f64::NAN;
                continue;
            }
        };
        let Some((y2, m2)) = embed(&arr[..end_d1], d + 1, tau) else {
            f1_values[d - 1] = f64::NAN;
            f2_values[d - 1] = f64::NAN;
            f3_values[d - 1] = f64::NAN;
            continue;
        };
        debug_assert_eq!(m2, m);

        // Find NN in d-dimensional space (Euclidean, standard for FNN)
        let (nn_idx, nn_dist) = nearest_neighbor_euclidean(&y1, m, d, theiler_window);

        let mut count1 = 0usize;
        let mut count2 = 0usize;
        let mut count3 = 0usize;
        let mut total = 0usize;

        for i in 0..m {
            let j = nn_idx[i];
            let dist = nn_dist[i];

            // Skip points with no valid neighbor (Theiler window too large)
            if dist.is_infinite() {
                continue;
            }

            // Extra distance in the (d+1)-th coordinate
            let extra = (y2[i * (d + 1) + d] - y2[j * (d + 1) + d]).abs();

            // Full Euclidean distance in (d+1) dimensions
            let mut full_sq = 0.0f64;
            for k in 0..(d + 1) {
                let diff = y2[i * (d + 1) + k] - y2[j * (d + 1) + k];
                full_sq += diff * diff;
            }
            let full_dist = full_sq.sqrt();

            // Test I: extra_dist / nn_dist > R_tol
            let t1 = if dist > 0.0 {
                (extra / dist) > r_tol
            } else {
                false
            };

            // Test II: full_dist / sigma > A_tol
            let t2 = (full_dist / sigma) > a_tol;

            if t1 {
                count1 += 1;
            }
            if t2 {
                count2 += 1;
            }
            if t1 || t2 {
                count3 += 1;
            }
            total += 1;
        }

        let tf = total as f64;
        f1_values[d - 1] = if total > 0 { count1 as f64 / tf } else { f64::NAN };
        f2_values[d - 1] = if total > 0 { count2 as f64 / tf } else { f64::NAN };
        f3_values[d - 1] = if total > 0 { count3 as f64 / tf } else { f64::NAN };
    }

    Ok((
        PyArray1::from_vec(py, f1_values),
        PyArray1::from_vec(py, f2_values),
        PyArray1::from_vec(py, f3_values),
    ))
}

/// Select embedding dimension from a Cao E1(d) curve.
///
/// This mirrors the Python selector logic:
/// 1) choose onset of a stable near-1 plateau (forward window),
/// 2) fallback to first near-one crossing,
/// 3) fallback to closest value to 1.
#[pyfunction]
#[pyo3(signature = (
    e1,
    near_one_lower = 0.95,
    near_one_upper = 1.05,
    saturation_tol = 0.02,
    plateau_span = 3,
    smoothing_window = 1,
    min_dim = 2,
    max_dim = None
))]
pub fn select_dimension_cao(
    e1: PyReadonlyArray1<'_, f64>,
    near_one_lower: f64,
    near_one_upper: f64,
    saturation_tol: f64,
    plateau_span: usize,
    smoothing_window: usize,
    min_dim: usize,
    max_dim: Option<usize>,
) -> PyResult<usize> {
    let e1 = e1.as_slice()?;
    let min_dim = min_dim.max(1);
    if e1.is_empty() {
        return Ok(min_dim);
    }

    let mut lo = near_one_lower;
    let mut hi = near_one_upper;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }

    let smoothed = smooth_series(e1, smoothing_window);
    if !smoothed.iter().any(|v| v.is_finite()) {
        return Ok(min_dim);
    }

    let span = plateau_span.max(2);
    let max_allowed = max_dim.unwrap_or(smoothed.len() + 1).max(min_dim);
    let clamp = |d: usize| -> usize { d.max(min_dim).min(max_allowed) };

    // Primary: plateau onset in forward window.
    for idx in 0..smoothed.len() {
        let dim = idx + 1;
        let value = smoothed[idx];
        if dim < min_dim || !value.is_finite() {
            continue;
        }
        if value < lo || value > hi {
            continue;
        }
        if idx + span > smoothed.len() {
            continue;
        }

        let window = &smoothed[idx..idx + span];
        if window.iter().any(|v| !v.is_finite() || *v < lo || *v > hi) {
            continue;
        }

        let mut w_min = f64::INFINITY;
        let mut w_max = f64::NEG_INFINITY;
        let mut max_diff = 0.0f64;
        for (k, v) in window.iter().enumerate() {
            w_min = w_min.min(*v);
            w_max = w_max.max(*v);
            if k > 0 {
                let diff = (window[k] - window[k - 1]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        if (w_max - w_min) <= 1.5 * saturation_tol || max_diff <= saturation_tol {
            return Ok(clamp(dim));
        }
    }

    // Fallback 1: first near-one crossing.
    for (idx, value) in smoothed.iter().enumerate() {
        let dim = idx + 1;
        if dim >= min_dim && value.is_finite() && *value >= lo {
            return Ok(clamp(dim));
        }
    }

    // Fallback 2: closest to 1 among valid dimensions.
    let mut best_dim: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for (idx, value) in smoothed.iter().enumerate() {
        let dim = idx + 1;
        if dim < min_dim || !value.is_finite() {
            continue;
        }
        let dist = (*value - 1.0).abs();
        if dist < best_dist {
            best_dist = dist;
            best_dim = Some(dim);
        }
    }
    if let Some(dim) = best_dim {
        return Ok(clamp(dim));
    }

    Ok(clamp(e1.len() + 1))
}
