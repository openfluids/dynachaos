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
}
