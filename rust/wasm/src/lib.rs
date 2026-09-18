//! Browser surface over the dynachaos kernels.
//!
//! Every export here obeys the contract in `docs/wasm-architecture.md`, which
//! exists because a web page is a hostile caller: its arguments arrive from a
//! slider, a URL or a stranger.
//!
//! 1. Clamp every input into a documented range. Never trust, never reject.
//! 2. Cap every loop. A frozen tab is worse than a coarse picture.
//! 3. Return one flat `Vec<f64>` whose layout is written in the doc comment.
//!
//! The kernels themselves live in `dynachaos-core` and are the same code the
//! published figures were computed with. This crate only converts and guards.
//! Three exports: `rotation_number_tile` for the picture,
//! `rotation_number_point` for the quoted readout, and `zero_one_k` for the
//! 0-1 test for chaos.

use wasm_bindgen::prelude::*;

/// Largest tile side the browser may ask for, in cells.
const MAX_SIDE: usize = 512;
/// Largest transient the browser may ask for, in steps.
const MAX_TRANSIENT: usize = 20_000;
/// Largest measured stretch the browser may ask for, in steps.
const MAX_ITER: usize = 200_000;
/// Total step budget for one call: cells x (transient + iter).
///
/// A worker computes one tile per frame, so this is the ceiling on how long a
/// single call may take. At roughly 2 ns per step this is about half a second
/// of work on one core. `n_iter` is reduced to fit, and the value actually
/// used is reported in the returned header so the caller can label the result.
const MAX_STEPS: u64 = 250_000_000;

/// Number of leading `f64` values that describe the tile before the data.
const HEADER: usize = 4;

/// Largest series the browser may hand to `zero_one_k`, in samples.
const MAX_ZERO_ONE_N: usize = 20_000;
/// Largest c ensemble the browser may ask for.
const MAX_ZERO_ONE_C: usize = 100;
/// Largest regression window the browser may ask for, in lags.
const MAX_ZERO_ONE_CUT: usize = 2_000;
/// Total work budget for one `zero_one_k` call: n_c x n_cut x N pair terms.
///
/// Each lag sum costs one multiply-add per pair, so this is the ceiling on how
/// long a single call may take. `n_cut` is reduced to fit, and the value
/// actually used is reported in the returned header.
const MAX_ZERO_ONE_PAIRS: u64 = 200_000_000;
/// Number of leading `f64` values that describe a `zero_one_k` result.
const ZERO_ONE_HEADER: usize = 3;

/// Rotation numbers of the sine circle map over a tile of the (Omega, K) plane.
///
/// The map is `theta -> theta + Omega + K * sin(2 * pi * theta)`, iterated
/// without reduction modulo 1. It is the map
/// `dynachaos.maps.arnold_tongues` sweeps to draw the Arnold tongues, and in
/// this convention it stops being invertible at `K = 1 / (2 * pi)`, about
/// 0.159.
///
/// # Returned layout
///
/// One flat array of `4 + n_k * n_omega` values:
///
/// - `[0]` = `n_omega` actually used, after clamping.
/// - `[1]` = `n_k` actually used, after clamping.
/// - `[2]` = `n_transient` actually used.
/// - `[3]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[4 ..]` = the rotation numbers, row-major, `n_k` rows of `n_omega`
///   values. K varies down the rows, Omega along the columns, matching the
///   `rho` array the reproduction pipeline stores.
///
/// Read the header rather than assuming the values you passed were honoured.
/// Every input is clamped, so a request outside the documented range comes
/// back as the nearest allowed one instead of an error.
///
/// # Clamping
///
/// - `n_omega`, `n_k`: 1 to 512 cells.
/// - `n_transient`: 0 to 20000 steps.
/// - `n_iter`: 1 to 200000 steps, reduced further to respect the step budget.
/// - `omega_min`, `omega_max`, `k_min`, `k_max`, `theta0`: a non-finite value
///   falls back to the value the published figure uses.
#[wasm_bindgen]
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
) -> Vec<f64> {
    let omega_min = finite_or(omega_min, 0.0);
    let omega_max = finite_or(omega_max, 1.0);
    let k_min = finite_or(k_min, 0.0);
    let k_max = finite_or(k_max, 0.3);
    let theta0 = finite_or(theta0, 0.1);

    let n_omega = n_omega.clamp(1, MAX_SIDE);
    let n_k = n_k.clamp(1, MAX_SIDE);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(n_omega, n_k, n_transient, n_iter);

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty tile rather than trapping and killing the worker.
    let Ok(rho) = dynachaos_core::rotation_number_tile(
        omega_min,
        omega_max,
        n_omega,
        k_min,
        k_max,
        n_k,
        n_transient,
        n_iter,
        theta0,
    ) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + rho.len());
    out.push(n_omega as f64);
    out.push(n_k as f64);
    out.push(n_transient as f64);
    out.push(n_iter as f64);
    out.extend_from_slice(&rho);
    out
}

/// Rotation number of one (Omega, K) point, lock detection on, display stop off.
///
/// This is the value the live figure quotes on hover. Locked orbits return the
/// exact rational; unlocked orbits run the full `n_iter` average. The tile
/// export keeps the display-tolerance stop; this one does not.
///
/// # Returned layout
///
/// One `f64`: `[0]` is the rotation number after clamping.
///
/// # Clamping
///
/// - `n_transient`: 0 to 20000 steps.
/// - `n_iter`: 1 to 200000 steps, reduced further to respect the step budget
///   for a single cell.
/// - `omega`, `k`, `theta0`: a non-finite value falls back to the published
///   figure's defaults (0, 0, 0.1).
#[wasm_bindgen]
pub fn rotation_number_point(
    omega: f64,
    k: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> Vec<f64> {
    let omega = finite_or(omega, 0.0);
    let k = finite_or(k, 0.0);
    let theta0 = finite_or(theta0, 0.1);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(1, 1, n_transient, n_iter);
    vec![dynachaos_core::rotation_number_point(
        omega,
        k,
        n_transient,
        n_iter,
        theta0,
    )]
}

/// The 0-1 test for chaos: one K per frequency in `c_values`.
///
/// This is the same estimator `dynachaos.diagnostics.zero_one_test` runs in
/// Python: per frequency `c` it builds the translation variables
/// `p_n = Σ φ_j cos(jc)`, `q_n = Σ φ_j sin(jc)`, forms the mean-square
/// displacement from the autocovariance identity, and returns the Pearson
/// correlation of the lag with `D`. The caller draws the frequencies; the
/// kernel only evaluates them.
///
/// # Returned layout
///
/// One flat array of `3 + n_c` values:
///
/// - `[0]` = `N` actually used, after truncation.
/// - `[1]` = `n_c` actually used, after truncation.
/// - `[2]` = `n_cut` actually used, which may be lower than requested when
///   the work budget binds.
/// - `[3 ..]` = one K per c value kept, in the order they were passed.
/// Read the header rather than assuming the values you passed were honoured.
/// An oversized request is cut, not refused. A request that cannot produce a
/// statistic at all — a non-finite sample or frequency, fewer than two
/// samples, or no frequencies — returns an empty array, never a trap.
///
/// # Clamping
///
/// - `phi`: truncated to 20000 samples; a non-finite sample anywhere in the
///   kept prefix returns an empty array.
/// - `c_values`: truncated to 100 frequencies; a non-finite frequency
///   anywhere in the kept prefix returns an empty array.
/// - `n_cut`: clamped to `[2, N]` and to 2000, then reduced further until
///   `n_c x n_cut x N` fits the 2e8 pair budget.
#[wasm_bindgen]
pub fn zero_one_k(phi: &[f64], c_values: &[f64], n_cut: usize) -> Vec<f64> {
    let n = phi.len().min(MAX_ZERO_ONE_N);
    let n_c = c_values.len().min(MAX_ZERO_ONE_C);
    if n < 2 || n_c == 0 {
        return Vec::new();
    }
    let phi = &phi[..n];
    let c_values = &c_values[..n_c];
    if phi.iter().any(|v| !v.is_finite()) || c_values.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }

    let n_cut = n_cut.clamp(2, n).min(MAX_ZERO_ONE_CUT);
    let n_cut = fit_zero_one_budget(n, n_c, n_cut);

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty result rather than trapping and killing the worker.
    let Ok(k_values) = dynachaos_core::zero_one_k(phi, c_values, n_cut) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(ZERO_ONE_HEADER + k_values.len());
    out.push(n as f64);
    out.push(n_c as f64);
    out.push(n_cut as f64);
    out.extend_from_slice(&k_values);
    out
}

/// Reduce `n_cut` until `n_c x n_cut x N` fits the pair budget.
///
/// The result is at least 2, the smallest window the estimator accepts: a
/// regression over two lags is poor but honest, and the header reports what
/// was used.
fn fit_zero_one_budget(n: usize, n_c: usize, n_cut: usize) -> usize {
    let per_lag = (n_c as u64).saturating_mul(n as u64);
    if per_lag.saturating_mul(n_cut as u64) <= MAX_ZERO_ONE_PAIRS {
        return n_cut;
    }
    let affordable = MAX_ZERO_ONE_PAIRS / per_lag.max(1);
    (affordable as usize).max(2)
}

/// Return `value` when it is finite, otherwise `fallback`.
fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

/// Reduce `n_iter` until the call fits the step budget.
///
/// The transient is charged too, because it costs the same per step. The
/// result is at least 1: a tile computed from a single measured step is poor
/// but honest, and the header reports what was used.
fn fit_step_budget(n_omega: usize, n_k: usize, n_transient: usize, n_iter: usize) -> usize {
    let cells = (n_omega as u64) * (n_k as u64);
    let requested = cells.saturating_mul(n_transient as u64 + n_iter as u64);
    if requested <= MAX_STEPS {
        return n_iter;
    }
    let affordable = MAX_STEPS / cells.max(1);
    let measured = affordable.saturating_sub(n_transient as u64);
    measured.max(1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_reports_the_values_actually_used() {
        let out = rotation_number_tile(0.0, 1.0, 3, 0.0, 0.3, 2, 100, 200, 0.1);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 100.0);
        assert_eq!(out[3], 200.0);
        assert_eq!(out.len(), HEADER + 6);
    }

    #[test]
    fn oversized_requests_are_clamped_not_refused() {
        let out = rotation_number_tile(0.0, 1.0, 100_000, 0.0, 0.3, 100_000, 0, 1, 0.1);
        assert_eq!(out[0], MAX_SIDE as f64);
        assert_eq!(out[1], MAX_SIDE as f64);
        assert_eq!(out.len(), HEADER + MAX_SIDE * MAX_SIDE);
    }

    #[test]
    fn a_huge_iteration_count_is_cut_to_the_step_budget() {
        let out = rotation_number_tile(0.0, 1.0, 256, 0.0, 0.3, 256, 0, MAX_ITER, 0.1);
        let used = out[3] as u64;
        assert!(used >= 1);
        assert!(used < MAX_ITER as u64);
        assert!((256u64 * 256) * used <= MAX_STEPS);
    }

    #[test]
    fn non_finite_arguments_fall_back_to_the_published_range() {
        let out = rotation_number_tile(f64::NAN, f64::INFINITY, 2, 0.0, 0.3, 1, 10, 10, f64::NAN);
        // Omega fell back to 0..1 and theta0 to 0.1, so the call still ran.
        assert_eq!(out.len(), HEADER + 2);
        assert!(out[HEADER..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_zero_coupling_row_still_returns_omega() {
        // Same analytic check the core crate makes, through the clamped
        // browser entry point: with K = 0 the drift per step is Omega.
        let out = rotation_number_tile(0.0, 0.5, 3, 0.0, 0.0, 1, 200, 500, 0.1);
        let tile = &out[HEADER..];
        for (value, expected) in tile.iter().zip([0.0, 0.25, 0.5]) {
            assert!((value - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn a_locked_point_returns_the_exact_rational() {
        let out = rotation_number_point(0.05, 0.12, 200, 2000, 0.1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn a_non_finite_point_still_returns_a_number() {
        let out = rotation_number_point(f64::NAN, f64::NAN, 10, 10, f64::NAN);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    fn zero_one_phi(n: usize) -> Vec<f64> {
        // A deterministic chaotic-ish series: logistic map at r = 4.
        let mut x = 0.123456789f64;
        (0..n)
            .map(|_| {
                x = 4.0 * x * (1.0 - x);
                x
            })
            .collect()
    }

    #[test]
    fn zero_one_header_reports_the_values_actually_used() {
        let phi = zero_one_phi(500);
        let c = [0.7, 1.1, 1.9];
        let out = zero_one_k(&phi, &c, 50);
        assert_eq!(out[0], 500.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 50.0);
        assert_eq!(out.len(), ZERO_ONE_HEADER + 3);
        assert!(out[ZERO_ONE_HEADER..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_one_oversized_requests_are_cut_not_refused() {
        let phi = zero_one_phi(MAX_ZERO_ONE_N + 5000);
        let c = vec![0.9; MAX_ZERO_ONE_C + 50];
        let out = zero_one_k(&phi, &c, MAX_ZERO_ONE_CUT + 500);
        assert_eq!(out[0], MAX_ZERO_ONE_N as f64);
        assert_eq!(out[1], MAX_ZERO_ONE_C as f64);
        // The pair budget binds below the n_cut ceiling at this size:
        // 100 c x 20000 N leaves room for 100 lags, not 2000.
        assert_eq!(out[2], 100.0);
    }

    #[test]
    fn zero_one_n_cut_is_cut_to_the_pair_budget() {
        // n_c x n_cut x N = 100 x 2000 x 20000 = 4e9 pairs, over the 2e8
        // budget, so n_cut must come back reduced.
        let phi = zero_one_phi(MAX_ZERO_ONE_N);
        let c = vec![0.9; MAX_ZERO_ONE_C];
        let out = zero_one_k(&phi, &c, MAX_ZERO_ONE_CUT);
        let used = out[2] as u64;
        assert!(used >= 2);
        assert!(used < MAX_ZERO_ONE_CUT as u64);
        assert!((MAX_ZERO_ONE_C as u64) * used * (MAX_ZERO_ONE_N as u64) <= MAX_ZERO_ONE_PAIRS);
    }

    #[test]
    fn zero_one_n_cut_is_clamped_into_the_series() {
        let phi = zero_one_phi(50);
        let c = [0.9];
        let out = zero_one_k(&phi, &c, 10_000);
        assert_eq!(out[2], 50.0);
        assert_eq!(out.len(), ZERO_ONE_HEADER + 1);
    }

    #[test]
    fn zero_one_non_finite_input_returns_an_empty_result() {
        let mut phi = zero_one_phi(100);
        phi[40] = f64::NAN;
        assert!(zero_one_k(&phi, &[0.9], 10).is_empty());

        let phi = zero_one_phi(100);
        assert!(zero_one_k(&phi, &[0.9, f64::INFINITY], 10).is_empty());
    }

    #[test]
    fn zero_one_unusable_requests_return_an_empty_result() {
        let phi = zero_one_phi(100);
        assert!(zero_one_k(&phi, &[], 10).is_empty());
        assert!(zero_one_k(&phi[..1], &[0.9], 10).is_empty());
    }

    #[test]
    fn zero_one_a_zero_series_returns_exact_zeros() {
        let phi = vec![0.0f64; 100];
        let out = zero_one_k(&phi, &[0.7, 1.1, 1.9], 10);
        assert_eq!(out.len(), ZERO_ONE_HEADER + 3);
        assert!(out[ZERO_ONE_HEADER..].iter().all(|&v| v == 0.0));
    }
}
