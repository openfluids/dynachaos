//! Coupled logistic map kernels.

use ndarray::{Array2, ArrayView2};

use crate::CoreError;

/// Compute basin labels for a fixed two-site coupled logistic map.
///
/// The caller owns reference-orbit construction, grids, and plotting.
/// This kernel runs the grid transient and classifies each point.
pub fn coupled_logistic_basin_grid(
    x_values: &[f64],
    y_values: &[f64],
    a: f64,
    d: f64,
    n_transient: usize,
    ref_a: ArrayView2<f64>,
) -> Result<Array2<i8>, CoreError> {
    if ref_a.ndim() != 2 || ref_a.shape()[1] != 2 {
        return Err(CoreError::invalid_argument(
            "ref_a must have shape (period, 2)",
        ));
    }

    let nx = x_values.len();
    let ny = y_values.len();
    let period = ref_a.shape()[0];
    let basin_len = nx
        .checked_mul(ny)
        .ok_or_else(|| CoreError::invalid_argument("basin grid is too large"))?;

    let ref_pairs: Vec<(f64, f64)> = (0..period)
        .map(|k| (ref_a[[k, 0]], ref_a[[k, 1]]))
        .collect();

    let mut basin = Vec::with_capacity(basin_len);
    for &y0 in y_values {
        basin.extend(basin_row(x_values, y0, a, d, n_transient, &ref_pairs));
    }

    Array2::from_shape_vec((ny, nx), basin)
        .map_err(|e| CoreError::runtime(format!("shape error basin: {e}")))
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
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        x_values
            .par_iter()
            .map(|&x0| basin_point(x0, y0, a, d, n_transient, ref_pairs))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        x_values
            .iter()
            .map(|&x0| basin_point(x0, y0, a, d, n_transient, ref_pairs))
            .collect()
    }
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
