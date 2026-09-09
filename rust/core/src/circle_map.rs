//! Sine circle map rotation numbers over a rectangular (Omega, K) tile.
//!
//! The map is the one the reproduction pipeline uses
//! (`src/dynachaos/maps/arnold_tongues.py`, `src/dynachaos/maps/circle_map.py`):
//!
//! ```text
//! theta_{n+1} = theta_n + Omega + K * sin(2 * pi * theta_n)
//! ```
//!
//! Note the sine term carries `K` directly, not `K / (2 * pi)`. In this
//! convention the map loses invertibility at `K = 1 / (2 * pi)`, about 0.159,
//! which is the critical line the Arnold-tongue figure draws.
//!
//! The iteration is deliberately not reduced modulo 1. The rotation number is
//! the mean drift of the unwrapped angle, so wrapping would destroy it.

use crate::CoreError;

/// Values equally spaced from `start` to `stop`, `stop` included.
///
/// This reproduces `numpy.linspace`: each value is `start + i * step`, and the
/// last value is forced to `stop` so the endpoint is exact. A single point
/// returns `start`, as numpy does.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    let mut values: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    values[n - 1] = stop;
    values
}

/// Rotation number of one (Omega, K) cell.
///
/// `theta` runs `n_transient` steps to settle onto the attractor, then
/// `n_iter` more steps. The rotation number is the mean drift per step over
/// the second stretch.
#[inline]
fn rotation_number(omega: f64, k: f64, n_transient: usize, n_iter: usize, theta0: f64) -> f64 {
    const TWO_PI: f64 = std::f64::consts::TAU;
    let mut theta = theta0;
    for _ in 0..n_transient {
        theta += omega + k * (TWO_PI * theta).sin();
    }
    let theta_start = theta;
    for _ in 0..n_iter {
        theta += omega + k * (TWO_PI * theta).sin();
    }
    (theta - theta_start) / n_iter as f64
}

/// Rotation numbers over a rectangular tile of the (Omega, K) plane.
///
/// The tile has `n_k` rows and `n_omega` columns. Omega runs from `omega_min`
/// to `omega_max` along each row; K runs from `k_min` to `k_max` down the
/// rows. Both ends are included, as in `numpy.linspace`.
///
/// Returns one `Vec<f64>` of `n_k * n_omega` values in row-major order: the
/// cell at row `i` (K index) and column `j` (Omega index) is at
/// `i * n_omega + j`. This matches the `rho` array the reproduction pipeline
/// stores, which has shape `(n_K, n_omega)`.
///
/// Every cell is independent, so the parallel and sequential builds return
/// identical values; there is no reduction whose order could differ.
///
/// This function validates its arguments and reports a message on bad input.
/// It does not clamp: clamping ranges for untrusted callers belongs in the
/// browser binding, which has to accept whatever a slider sends, while the
/// Python caller must still be able to ask for a paper-scale tile.
#[allow(clippy::too_many_arguments)]
pub fn rotation_number_tile(
    omega_min: f64,
    omega_max: f64,
    n_omega: usize,
    k_min: f64,
    k_max: f64,
    n_k: usize,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> Result<Vec<f64>, CoreError> {
    if n_omega < 1 || n_k < 1 {
        return Err(CoreError::invalid_argument("n_omega and n_k must be >= 1"));
    }
    if n_iter < 1 {
        return Err(CoreError::invalid_argument("n_iter must be >= 1"));
    }
    for (name, value) in [
        ("omega_min", omega_min),
        ("omega_max", omega_max),
        ("k_min", k_min),
        ("k_max", k_max),
        ("theta0", theta0),
    ] {
        if !value.is_finite() {
            return Err(CoreError::invalid_argument(format!(
                "{name} must be finite"
            )));
        }
    }
    let cells = n_k
        .checked_mul(n_omega)
        .ok_or_else(|| CoreError::invalid_argument("n_k * n_omega overflows"))?;

    let omega_values = linspace(omega_min, omega_max, n_omega);
    let k_values = linspace(k_min, k_max, n_k);
    let mut rho = vec![0.0_f64; cells];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        rho.par_chunks_mut(n_omega)
            .zip(k_values.par_iter())
            .for_each(|(row, &k)| {
                for (cell, &omega) in row.iter_mut().zip(omega_values.iter()) {
                    *cell = rotation_number(omega, k, n_transient, n_iter, theta0);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (row, &k) in rho.chunks_mut(n_omega).zip(k_values.iter()) {
            for (cell, &omega) in row.iter_mut().zip(omega_values.iter()) {
                *cell = rotation_number(omega, k, n_transient, n_iter, theta0);
            }
        }
    }

    Ok(rho)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values come from the Python pipeline, measured at
    // n_transient=2000, n_iter=5000, theta0=0.1.
    const N_TRANSIENT: usize = 2000;
    const N_ITER: usize = 5000;
    const THETA0: f64 = 0.1;

    fn single(omega: f64, k: f64) -> f64 {
        let tile =
            rotation_number_tile(omega, omega, 1, k, k, 1, N_TRANSIENT, N_ITER, THETA0).unwrap();
        tile[0]
    }

    #[test]
    fn without_coupling_the_rotation_number_is_omega() {
        // K = 0 leaves theta += Omega, so the drift per step is Omega. The
        // tolerance is not zero because 5000 unwrapped additions accumulate
        // rounding; Python gives 0.29999999999996246 for the same input.
        assert!((single(0.3, 0.0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn zero_tongue_locks_the_rotation_number_to_zero() {
        // Omega = 0 with K > 0 drives theta to the fixed point at 0.5, so the
        // angle stops drifting: this is the 0/1 Arnold tongue.
        assert_eq!(single(0.0, 0.2), 0.0);
    }

    #[test]
    fn half_tongue_locks_the_rotation_number_to_one_half() {
        // The 1/2 tongue: a period-2 cycle advancing half a turn per step.
        assert!((single(0.5, 0.2) - 0.5).abs() < 1e-12);
        assert!((single(0.5, 0.05) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tile_layout_is_row_major_with_k_down_the_rows() {
        // Two K rows, three Omega columns. Row 0 is K = 0, where every cell
        // must equal its own Omega.
        let tile = rotation_number_tile(0.0, 0.5, 3, 0.0, 0.2, 2, 200, 500, THETA0).unwrap();
        assert_eq!(tile.len(), 6);
        for (j, expected) in [0.0, 0.25, 0.5].iter().enumerate() {
            assert!((tile[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn bad_arguments_are_reported() {
        assert!(rotation_number_tile(0.0, 1.0, 0, 0.0, 0.3, 4, 10, 10, 0.1).is_err());
        assert!(rotation_number_tile(0.0, 1.0, 4, 0.0, 0.3, 4, 10, 0, 0.1).is_err());
        assert!(rotation_number_tile(f64::NAN, 1.0, 4, 0.0, 0.3, 4, 10, 10, 0.1).is_err());
    }

    #[test]
    fn linspace_matches_numpy_including_the_endpoint() {
        assert_eq!(linspace(0.0, 1.0, 1), vec![0.0]);
        assert_eq!(linspace(0.0, 1.0, 3), vec![0.0, 0.5, 1.0]);
        // The endpoint is exact even when the step does not divide evenly.
        let values = linspace(0.0, 0.3, 1000);
        assert_eq!(values[999], 0.3);
    }
}
