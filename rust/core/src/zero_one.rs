//! The Gottwald–Melbourne 0-1 test for chaos.
//!
//! This is a port of the estimator in
//! `src/dynachaos/diagnostics/zero_one_test.py`, not of the paper: it is the
//! correlation estimator of Gottwald & Melbourne (2009, Eq. 8) as that file
//! implements it, with the mean-square displacement formed from the
//! stationary-process identity `D(k) = 2[C_p(0) + C_q(0)] - 2[C_p(k) + C_q(k)]`
//! and no oscillatory `V_osc` correction. The Python file computes the
//! autocovariance by FFT; this kernel uses direct sums over the lags actually
//! used, which differs only by rounding.
//!
//! Per frequency `c` the estimator is:
//!
//! 1. `p_n = Σ φ_j cos(jc)`, `q_n = Σ φ_j sin(jc)` for `j = 1..N` (cumulative).
//! 2. Centre `p` and `q` on their global means.
//! 3. Autocovariance at lag `k`, normalised by `N − k`, for `k = 0..n_cut−1`.
//! 4. `D(k) = 2[C_p(0) + C_q(0)] − 2[C_p(k) + C_q(k)]`.
//! 5. If `D` is constant to `1e-15`, `K_c = 0`; otherwise `K_c` is the Pearson
//!    correlation of the lag `1..n_cut` with `D`.
//!
//! The caller draws the `c` values (Python draws them uniform in
//! `(π/5, 4π/5)`); the kernel only evaluates them.

use crate::CoreError;

/// Absolute tolerance under which `D` counts as constant and `K_c` is 0.
///
/// Matches the `atol` of the `np.allclose` guard in the Python reference.
const CONSTANT_D_ATOL: f64 = 1e-15;

/// Evaluate the 0-1 test for each frequency in `c_values`.
///
/// Returns one `K` per `c`, in the same order. The median over the returned
/// vector is the statistic; the caller computes it so the per-`c` values stay
/// available for diagnostics.
///
/// # Errors
///
/// - `phi` or `c_values` contains a non-finite value.
/// - `c_values` is empty.
/// - `n_cut` is outside `[2, phi.len()]`. This also rejects a `phi` shorter
///   than two samples, since no `n_cut` can satisfy the bound.
pub fn zero_one_k(phi: &[f64], c_values: &[f64], n_cut: usize) -> Result<Vec<f64>, CoreError> {
    let n = phi.len();
    if phi.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument(
            "phi must contain only finite values",
        ));
    }
    if c_values.is_empty() {
        return Err(CoreError::invalid_argument(
            "c_values must contain at least one value",
        ));
    }
    if c_values.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument(
            "c_values must contain only finite values",
        ));
    }
    if !(2..=n).contains(&n_cut) {
        return Err(CoreError::invalid_argument(
            "n_cut must be in [2, len(phi)]",
        ));
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        Ok(c_values
            .par_iter()
            .map(|&c| k_for_c(phi, c, n_cut))
            .collect())
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok(c_values.iter().map(|&c| k_for_c(phi, c, n_cut)).collect())
    }
}

/// The 0-1 statistic for one frequency `c`.
fn k_for_c(phi: &[f64], c: f64, n_cut: usize) -> f64 {
    let n = phi.len();

    // Translation variables: cumulative sums of φ_j cos(jc), φ_j sin(jc).
    let mut p = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut p_acc = 0.0f64;
    let mut q_acc = 0.0f64;
    for (j, &x) in phi.iter().enumerate() {
        let jc = (j + 1) as f64 * c;
        p_acc += x * jc.cos();
        q_acc += x * jc.sin();
        p.push(p_acc);
        q.push(q_acc);
    }

    // Centre on the global means, as the Python reference does before the
    // autocovariance.
    let mean_p = p.iter().sum::<f64>() / n as f64;
    let mean_q = q.iter().sum::<f64>() / n as f64;
    for v in &mut p {
        *v -= mean_p;
    }
    for v in &mut q {
        *v -= mean_q;
    }

    // Autocovariance at lags 0..n_cut-1, normalised by (N − k): direct sums
    // over the lags used, replacing the FFT of the Python reference.
    let mut d = Vec::with_capacity(n_cut);
    for k in 0..n_cut {
        let mut c_p = 0.0f64;
        let mut c_q = 0.0f64;
        for j in 0..(n - k) {
            c_p += p[j] * p[j + k];
            c_q += q[j] * q[j + k];
        }
        let norm = (n - k) as f64;
        d.push(-2.0 * (c_p + c_q) / norm);
    }
    // D(k) = 2[C_p(0) + C_q(0)] − 2[C_p(k) + C_q(k)]; the first term is
    // folded in here so `d` holds the negated autocovariance sums above.
    let var_sum = -d[0] / 2.0;
    for v in &mut d {
        *v += 2.0 * var_sum;
    }

    // Constant D (to 1e-15) means no growth: K_c = 0. This mirrors the
    // np.allclose(D, D[0], rtol=0, atol=1e-15) guard; D[0] is exactly 0.
    if d.iter().all(|v| (v - d[0]).abs() <= CONSTANT_D_ATOL) {
        return 0.0;
    }

    // K_c is the Pearson correlation of the lag 1..n_cut with D(0..n_cut-1),
    // the modified correlation estimator of G&M 2009 Eq. 8.
    pearson_lag_correlation(&d)
}

/// Pearson correlation of `1..=d.len()` with `d`, matching
/// `np.corrcoef(np.arange(1, n_cut + 1), D[:n_cut])[0, 1]`.
fn pearson_lag_correlation(d: &[f64]) -> f64 {
    let n = d.len() as f64;
    let mean_lag = (d.len() + 1) as f64 / 2.0;
    let mean_d = d.iter().sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_lag = 0.0f64;
    let mut var_d = 0.0f64;
    for (i, &v) in d.iter().enumerate() {
        let lag_dev = (i + 1) as f64 - mean_lag;
        let d_dev = v - mean_d;
        cov += lag_dev * d_dev;
        var_lag += lag_dev * lag_dev;
        var_d += d_dev * d_dev;
    }
    cov / (var_lag * var_d).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Frequencies spread over the (π/5, 4π/5) band the Python caller draws
    /// from. Deterministic so the native tests need no RNG.
    fn c_values(n_c: usize) -> Vec<f64> {
        let lo = PI / 5.0;
        let hi = 4.0 * PI / 5.0;
        (0..n_c)
            .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n_c as f64)
            .collect()
    }

    /// The regular two-tone series of `test_zero_one_regular_vs_chaotic`.
    fn regular_series(n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let t = t as f64;
                (2.0 * PI * 0.071 * t).sin() + 0.2 * (2.0 * PI * 0.113 * t).sin()
            })
            .collect()
    }

    /// Logistic map x -> r x (1 - x) at r = 4, fully chaotic.
    fn logistic_r4(n: usize, burn: usize) -> Vec<f64> {
        let mut x = 0.123456789f64;
        let mut series = Vec::with_capacity(n);
        for i in 0..(n + burn) {
            x = 4.0 * x * (1.0 - x);
            if i >= burn {
                series.push(x);
            }
        }
        series
    }

    /// The Kaneko logistic x -> 1 - a x² used by the Python test helper
    /// (`logistic_series` in tests/conftest.py, a = 1.99).
    fn logistic_kaneko(n: usize, burn: usize) -> Vec<f64> {
        let mut x = 0.123456789f64;
        let mut series = Vec::with_capacity(n);
        for i in 0..(n + burn) {
            x = 1.0 - 1.99 * x * x;
            if i >= burn {
                series.push(x);
            }
        }
        series
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        let n = values.len();
        if n % 2 == 1 {
            values[n / 2]
        } else {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        }
    }

    #[test]
    fn a_periodic_series_gives_k_near_zero() {
        // Threshold mirrored from tests/test_diagnostics.py: regular < 0.4.
        let phi = regular_series(5000);
        let k = median(zero_one_k(&phi, &c_values(20), 500).unwrap());
        assert!(k < 0.4, "regular series gave K = {k}");
    }

    #[test]
    fn the_logistic_map_at_r4_gives_k_near_one() {
        // Threshold mirrored from tests/test_diagnostics.py: chaotic > 0.6.
        let phi = logistic_r4(5000, 2000);
        let k = median(zero_one_k(&phi, &c_values(20), 500).unwrap());
        assert!(k > 0.6, "logistic r = 4 gave K = {k}");
    }

    #[test]
    fn the_kaneko_logistic_series_gives_k_near_one() {
        // Same generator and length as the Python test.
        let phi = logistic_kaneko(5000, 2000);
        let k = median(zero_one_k(&phi, &c_values(20), 500).unwrap());
        assert!(k > 0.6, "Kaneko logistic gave K = {k}");
    }

    #[test]
    fn an_all_zero_series_gives_exactly_zero() {
        let phi = vec![0.0f64; 100];
        let ks = zero_one_k(&phi, &c_values(5), 10).unwrap();
        assert_eq!(ks, vec![0.0; 5]);
    }

    #[test]
    fn bad_arguments_are_reported() {
        let phi = [0.1, 0.2, 0.3, 0.4];
        let c = c_values(3);

        assert!(zero_one_k(&phi, &c, 1).is_err());
        assert!(zero_one_k(&phi, &c, 5).is_err());
        assert!(zero_one_k(&phi, &[], 2).is_err());
        assert!(zero_one_k(&[0.1, f64::NAN, 0.3], &c, 2).is_err());
        assert!(zero_one_k(&phi, &[0.5, f64::INFINITY], 2).is_err());
        // n_cut == len(phi) is allowed.
        assert!(zero_one_k(&phi, &c, 4).is_ok());
    }
}
