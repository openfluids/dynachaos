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
//! Seven exports: `rotation_number_tile` for the picture,
//! `rotation_number_point` for the quoted readout, `zero_one_k` for the
//! 0-1 test for chaos, and `correlation_counts`, `apen_counts`,
//! `fuzzy_entropy_sum` and `ordinal_distribution` for the diagnostics panel.

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

/// Largest embedded trajectory the browser may hand to a diagnostics kernel,
/// in rows.
const MAX_DIAG_N: usize = 20_000;
/// Largest embedding dimension a diagnostics kernel may be asked to use.
const MAX_DIAG_DIM: usize = 32;
/// Largest radius list `correlation_counts` may be asked to evaluate.
const MAX_DIAG_R: usize = 256;
/// Total work budget for one diagnostics call: pairs x dim terms.
///
/// One pair term is one coordinate subtraction and comparison (or one
/// membership evaluation for the fuzzy sum), so this is the ceiling on how
/// long a single call may take. The trajectory is shortened to fit, and the
/// row count actually used is reported in the returned header.
const MAX_DIAG_TERMS: u64 = 200_000_000;
/// Largest fuzzy exponent `fuzzy_entropy_sum` may be asked to use.
const MAX_FUZZY_N: u32 = 16;
/// Largest pattern length `ordinal_distribution` may be asked to use.
///
/// The output carries `d!` counts, so `d` is capped at 8 (40320 values)
/// rather than the kernel's 10 (3.6 million) to keep the flat result small
/// enough to post back to a page.
const MAX_ORDINAL_D: usize = 8;
/// Number of leading `f64` values that describe a `correlation_counts`
/// result, before the radii actually used.
const CORRELATION_HEADER: usize = 5;
/// Number of leading `f64` values that describe an `apen_counts` result.
const APEN_HEADER: usize = 3;
/// Number of leading `f64` values that describe a `fuzzy_entropy_sum` result.
const FUZZY_HEADER: usize = 5;
/// Number of leading `f64` values that describe an `ordinal_distribution`
/// result.
const ORDINAL_HEADER: usize = 4;

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

/// Correlation integral counts: pairs within each distance threshold.
///
/// This is the same estimator `dynachaos.diagnostics.correlation_integral`
/// runs in Python: `traj` is a row-major embedded trajectory of `n_pts` rows
/// of `dim` coordinates, and `counts[k]` is the number of pairs with
/// `|i - j| > theiler_window` and distance below `r_values[k]`. The kernel
/// requires the thresholds sorted ascending, so this export sorts a copy and
/// returns the radii it actually used; the counts line up with that list,
/// not with the caller's order.
///
/// # Returned layout
///
/// One flat array of `5 + n_r + n_r` values:
///
/// - `[0]` = `n_pts` actually used, after truncation.
/// - `[1]` = `dim` actually used, after clamping.
/// - `[2]` = `n_r` actually used, after truncation.
/// - `[3]` = `theiler_window` actually used, after clamping.
/// - `[4]` = `use_chebyshev` actually used: 1 for the max norm, 0 for
///   Euclidean.
/// - `[5 .. 5 + n_r]` = the radii actually used, sorted ascending.
/// - `[5 + n_r ..]` = one count per radius, in the same order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than one complete
/// row, no radii, or a non-finite entry anywhere in the kept trajectory or
/// radius prefix — returns an empty array, never a trap.
///
/// # Clamping
///
/// - `traj`: `n_pts = traj.len() / dim` complete rows, truncated to 20000;
///   a non-finite entry anywhere in the kept prefix returns an empty array.
/// - `dim`: clamped to `[1, 32]`.
/// - `r_values`: truncated to 256 entries; a non-finite entry in the kept
///   prefix returns an empty array; negative radii are clamped to 0 and the
///   list is sorted ascending.
/// - `theiler_window`: clamped to `n_pts - 1`.
/// - `n_pts` is reduced further until `pairs x dim` fits the 2e8 pair-term
///   budget; the value actually used is reported in the header.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn correlation_counts(
    traj: &[f64],
    dim: usize,
    r_values: &[f64],
    theiler_window: usize,
    use_chebyshev: bool,
) -> Vec<f64> {
    let dim = dim.clamp(1, MAX_DIAG_DIM);
    let mut n_pts = (traj.len() / dim).min(MAX_DIAG_N);
    if n_pts == 0 {
        return Vec::new();
    }
    let theiler_window = theiler_window.min(n_pts - 1);
    n_pts = fit_diag_budget(n_pts, dim, theiler_window, false);
    let theiler_window = theiler_window.min(n_pts - 1);
    let traj = &traj[..n_pts * dim];
    if traj.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }

    let mut r_used: Vec<f64> = r_values[..r_values.len().min(MAX_DIAG_R)].to_vec();
    if r_used.is_empty() || r_used.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    for r in &mut r_used {
        *r = r.max(0.0);
    }
    r_used.sort_by(f64::total_cmp);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(counts) = dynachaos_core::correlation_counts(
        traj,
        n_pts,
        dim,
        &r_used,
        theiler_window,
        use_chebyshev,
    ) else {
        return Vec::new();
    };

    let n_r = r_used.len();
    let mut out = Vec::with_capacity(CORRELATION_HEADER + n_r + counts.len());
    out.push(n_pts as f64);
    out.push(dim as f64);
    out.push(n_r as f64);
    out.push(theiler_window as f64);
    out.push(if use_chebyshev { 1.0 } else { 0.0 });
    out.extend_from_slice(&r_used);
    out.extend(counts.iter().map(|&c| c as f64));
    out
}

/// Approximate-entropy template matches: one count per template row.
///
/// This is the same estimator `dynachaos.diagnostics.approximate_entropy`
/// runs in Python: `traj` is a row-major embedded trajectory of `n_pts` rows
/// of `dim` coordinates, and `counts[i]` is the number of rows `j` with
/// Chebyshev distance `max(abs(traj[i] - traj[j])) <= r`, self-matches
/// included as Pincus (1991) defines them.
///
/// # Returned layout
///
/// One flat array of `3 + n_pts` values:
///
/// - `[0]` = `n_pts` actually used, after truncation.
/// - `[1]` = `dim` actually used, after clamping.
/// - `[2]` = `r` actually used, after the fallback below.
/// - `[3 ..]` = one match count per template row, in row order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than one complete
/// row, a non-finite entry anywhere in the kept trajectory, or a tolerance
/// that stays non-positive after the fallback — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `traj`: `n_pts = traj.len() / dim` complete rows, truncated to 20000;
///   a non-finite entry anywhere in the kept prefix returns an empty array.
/// - `dim`: clamped to `[1, 32]`.
/// - `r`: a non-finite or non-positive value falls back to `0.2 * std`
///   (ddof = 1) over all coordinates of the kept trajectory. The Python
///   diagnostics take the same fraction of the scalar series, which gives a
///   slightly different value.
/// - `n_pts` is reduced further until `n_pts^2 x dim` fits the 2e8
///   pair-term budget; the value actually used is reported in the header.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn apen_counts(traj: &[f64], dim: usize, r: f64) -> Vec<f64> {
    let dim = dim.clamp(1, MAX_DIAG_DIM);
    let mut n_pts = (traj.len() / dim).min(MAX_DIAG_N);
    if n_pts == 0 {
        return Vec::new();
    }
    n_pts = fit_diag_budget(n_pts, dim, 0, true);
    let traj = &traj[..n_pts * dim];
    if traj.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let Some(r) = tolerance_or_default(r, traj) else {
        return Vec::new();
    };

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(counts) = dynachaos_core::apen_counts(traj, n_pts, dim, r) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(APEN_HEADER + counts.len());
    out.push(n_pts as f64);
    out.push(dim as f64);
    out.push(r);
    out.extend(counts.iter().map(|&c| c as f64));
    out
}

/// Fuzzy-entropy membership sum over the valid template pairs.
///
/// This is the same accumulator `dynachaos.diagnostics.fuzzy_entropy` runs
/// in Python: `traj` is a row-major embedded trajectory of `n_pts` rows of
/// `dim` coordinates, and the result is the sum of `exp(-(d / r)^n)` over
/// the pairs with `j > i + theiler_window`, Chebyshev distance `d`. The
/// Python diagnostic mean-centres its templates before calling the kernel;
/// this export sums over the trajectory it is given, so centre first if
/// that is the comparison you want.
///
/// # Returned layout
///
/// One flat array of `6` values:
///
/// - `[0]` = `n_pts` actually used, after truncation.
/// - `[1]` = `dim` actually used, after clamping.
/// - `[2]` = `r` actually used, after the fallback below.
/// - `[3]` = `n` actually used, after clamping.
/// - `[4]` = `theiler_window` actually used, after clamping.
/// - `[5]` = the membership sum.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than one complete
/// row, a non-finite entry anywhere in the kept trajectory, or a tolerance
/// that stays non-positive after the fallback — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `traj`: `n_pts = traj.len() / dim` complete rows, truncated to 20000;
///   a non-finite entry anywhere in the kept prefix returns an empty array.
/// - `dim`: clamped to `[1, 32]`.
/// - `r`: a non-finite or non-positive value falls back to `0.2 * std`
///   (ddof = 1) over all coordinates of the kept trajectory. The Python
///   diagnostics take the same fraction of the scalar series, which gives a
///   slightly different value.
/// - `n`: clamped to `[1, 16]`.
/// - `theiler_window`: clamped to `n_pts - 1`.
/// - `n_pts` is reduced further until `pairs x dim` fits the 2e8 pair-term
///   budget; the value actually used is reported in the header.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn fuzzy_entropy_sum(
    traj: &[f64],
    dim: usize,
    r: f64,
    n: u32,
    theiler_window: usize,
) -> Vec<f64> {
    let dim = dim.clamp(1, MAX_DIAG_DIM);
    let mut n_pts = (traj.len() / dim).min(MAX_DIAG_N);
    if n_pts == 0 {
        return Vec::new();
    }
    let theiler_window = theiler_window.min(n_pts - 1);
    n_pts = fit_diag_budget(n_pts, dim, theiler_window, false);
    let theiler_window = theiler_window.min(n_pts - 1);
    let traj = &traj[..n_pts * dim];
    if traj.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let Some(r) = tolerance_or_default(r, traj) else {
        return Vec::new();
    };
    let n = n.clamp(1, MAX_FUZZY_N);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(sum) = dynachaos_core::fuzzy_entropy_sum(traj, n_pts, dim, r, n, theiler_window) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(FUZZY_HEADER + 1);
    out.push(n_pts as f64);
    out.push(dim as f64);
    out.push(r);
    out.push(n as f64);
    out.push(theiler_window as f64);
    out.push(sum);
    out
}

/// Ordinal pattern distribution: raw Lehmer-code counts of length `d!`.
///
/// This is the same estimator `dynachaos.diagnostics.ordinal_distribution`
/// runs in Python: each sliding window of `d` samples spaced `tau` apart is
/// encoded as the Lehmer code of its argsort permutation, and `counts[k]` is
/// the number of windows that landed on code `k`.
///
/// # Returned layout
///
/// One flat array of `4 + d!` values:
///
/// - `[0]` = `N` actually used, after truncation.
/// - `[1]` = `d` actually used, after clamping.
/// - `[2]` = `tau` actually used, after clamping.
/// - `[3]` = `n_windows` actually analysed.
/// - `[4 ..]` = `d!` counts in Lehmer-code order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — a non-finite sample
/// anywhere in the kept prefix, or a series too short for a single window —
/// returns an empty array, never a trap.
///
/// # Clamping
///
/// - `x`: truncated to 20000 samples; a non-finite sample anywhere in the
///   kept prefix returns an empty array.
/// - `d`: clamped to `[2, 8]` (the kernel accepts 10, but `10!` counts will
///   not fit a browser message).
/// - `tau`: clamped to `[1, N]`.
/// - The sample cap is the work budget: at most 20000 windows of at most 8
///   samples each, sized for interactive use.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn ordinal_distribution(x: &[f64], d: usize, tau: usize) -> Vec<f64> {
    let n = x.len().min(MAX_DIAG_N);
    if n == 0 {
        return Vec::new();
    }
    let x = &x[..n];
    if x.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let d = d.clamp(2, MAX_ORDINAL_D);
    let tau = tau.clamp(1, n);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok((counts, n_windows)) = dynachaos_core::ordinal_distribution(x, d, tau) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(ORDINAL_HEADER + counts.len());
    out.push(n as f64);
    out.push(d as f64);
    out.push(tau as f64);
    out.push(n_windows as f64);
    out.extend(counts.iter().map(|&c| c as f64));
    out
}

/// Reduce `n_pts` until `pairs x dim` fits the diagnostics pair-term budget.
///
/// With `self_pairs` the kernel compares every row to every row (`n_pts^2`
/// pairs, ApEn); otherwise it compares the upper triangle beyond the Theiler
/// window. The result is at least 1: a statistic over a single row is poor
/// but honest, and the header reports what was used.
fn fit_diag_budget(n_pts: usize, dim: usize, theiler_window: usize, self_pairs: bool) -> usize {
    let terms = |m: usize| -> u64 {
        let pairs = if self_pairs {
            (m as u64).saturating_mul(m as u64)
        } else {
            let n_eff = (m as u64).saturating_sub(theiler_window as u64 + 1);
            n_eff.saturating_mul(n_eff + 1) / 2
        };
        pairs.saturating_mul(dim as u64)
    };
    if terms(n_pts) <= MAX_DIAG_TERMS {
        return n_pts;
    }
    // Solve pairs(m) = budget for m, then walk down to the exact integer fit.
    let budget_pairs = MAX_DIAG_TERMS / (dim as u64).max(1);
    let mut m = if self_pairs {
        (budget_pairs as f64).sqrt() as usize
    } else {
        let n_eff = ((1.0 + 8.0 * budget_pairs as f64).sqrt() - 1.0) / 2.0;
        n_eff as usize + theiler_window + 1
    };
    while terms(m) > MAX_DIAG_TERMS {
        m -= 1;
    }
    m.min(n_pts).max(1)
}

/// Return `r` when it is finite and positive, otherwise `0.2 * std`, with
/// ddof = 1, over all coordinates of the kept trajectory. The Python diagnostics
/// take the same fraction of the scalar series; an embedding repeats most
/// samples `dim` times, so the two values differ slightly. `None` when even
/// the fallback cannot produce a positive tolerance.
fn tolerance_or_default(r: f64, traj: &[f64]) -> Option<f64> {
    if r.is_finite() && r > 0.0 {
        return Some(r);
    }
    let n = traj.len();
    if n < 2 {
        return None;
    }
    let mean = traj.iter().sum::<f64>() / n as f64;
    let variance = traj.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
    let fallback = 0.2 * variance.sqrt();
    if fallback.is_finite() && fallback > 0.0 {
        Some(fallback)
    } else {
        None
    }
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

    /// A deterministic embedded trajectory: logistic map at r = 4, tiled
    /// into `dim`-wide rows.
    fn diag_traj(n_pts: usize, dim: usize) -> Vec<f64> {
        let mut x = 0.123456789f64;
        (0..n_pts * dim)
            .map(|_| {
                x = 4.0 * x * (1.0 - x);
                x
            })
            .collect()
    }

    #[test]
    fn correlation_header_reports_the_values_actually_used() {
        let traj = diag_traj(200, 2);
        let r = [0.05, 0.1, 0.2];
        let out = correlation_counts(&traj, 2, &r, 3, true);
        assert_eq!(out[0], 200.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 3.0);
        assert_eq!(out[3], 3.0);
        assert_eq!(out[4], 1.0);
        assert_eq!(out.len(), CORRELATION_HEADER + 3 + 3);
        // The radii actually used come back sorted, then the counts.
        assert_eq!(&out[5..8], &[0.05, 0.1, 0.2]);
        let counts = &out[CORRELATION_HEADER + 3..];
        assert!(counts.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn correlation_radii_are_sorted_and_clamped_not_refused() {
        let traj = diag_traj(50, 1);
        let out = correlation_counts(&traj, 1, &[0.3, -1.0, 0.1], 0, true);
        assert_eq!(out[2], 3.0);
        assert_eq!(&out[5..8], &[0.0, 0.1, 0.3]);
    }

    #[test]
    fn correlation_pair_work_is_cut_to_the_budget() {
        // 20000 rows at dim 2 is 4e8 pair terms, over the 2e8 budget, so
        // n_pts must come back reduced.
        let traj = diag_traj(MAX_DIAG_N, 2);
        let out = correlation_counts(&traj, 2, &[0.5], 0, true);
        let used = out[0] as u64;
        assert!(used >= 1);
        assert!(used < MAX_DIAG_N as u64);
        let pairs = used * (used - 1) / 2;
        assert!(pairs * 2 <= MAX_DIAG_TERMS);
    }

    #[test]
    fn correlation_unusable_requests_return_an_empty_result() {
        let traj = diag_traj(100, 2);
        // Fewer than one complete row.
        assert!(correlation_counts(&traj[..1], 2, &[0.1], 0, true).is_empty());
        // No radii.
        assert!(correlation_counts(&traj, 2, &[], 0, true).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 2);
        bad[7] = f64::NAN;
        assert!(correlation_counts(&bad, 2, &[0.1], 0, true).is_empty());
        // A non-finite radius in the kept prefix.
        assert!(correlation_counts(&traj, 2, &[0.1, f64::INFINITY], 0, true).is_empty());
    }

    #[test]
    fn apen_header_reports_the_values_actually_used() {
        let traj = diag_traj(150, 2);
        let out = apen_counts(&traj, 2, 0.15);
        assert_eq!(out[0], 150.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.15);
        assert_eq!(out.len(), APEN_HEADER + 150);
        // Self-matches are included, so every row matches at least itself.
        assert!(out[APEN_HEADER..].iter().all(|&c| c >= 1.0));
    }

    #[test]
    fn apen_pair_work_is_cut_to_the_budget() {
        // 20000 rows is 4e8 pair terms at dim 1, over the 2e8 budget, so
        // n_pts must come back reduced.
        let traj = diag_traj(MAX_DIAG_N, 1);
        let out = apen_counts(&traj, 1, 0.2);
        let used = out[0] as u64;
        assert!(used >= 1);
        assert!(used < MAX_DIAG_N as u64);
        assert!(used * used <= MAX_DIAG_TERMS);
    }

    #[test]
    fn apen_a_non_positive_tolerance_falls_back() {
        let traj = diag_traj(100, 1);
        let out = apen_counts(&traj, 1, -1.0);
        assert!(!out.is_empty());
        let used_r = out[2];
        assert!(used_r > 0.0);
        assert!(used_r.is_finite());
    }

    #[test]
    fn apen_unusable_requests_return_an_empty_result() {
        let traj = diag_traj(100, 2);
        // Fewer than one complete row.
        assert!(apen_counts(&traj[..1], 2, 0.1).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 2);
        bad[5] = f64::NAN;
        assert!(apen_counts(&bad, 2, 0.1).is_empty());
        // A constant series leaves the fallback tolerance at zero.
        let flat = vec![0.5f64; 200];
        assert!(apen_counts(&flat, 1, f64::NAN).is_empty());
    }

    #[test]
    fn fuzzy_header_reports_the_values_actually_used() {
        let traj = diag_traj(120, 2);
        let out = fuzzy_entropy_sum(&traj, 2, 0.2, 2, 1);
        assert_eq!(out[0], 120.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.2);
        assert_eq!(out[3], 2.0);
        assert_eq!(out[4], 1.0);
        assert_eq!(out.len(), FUZZY_HEADER + 1);
        assert!(out[FUZZY_HEADER] > 0.0);
        assert!(out[FUZZY_HEADER].is_finite());
    }

    #[test]
    fn fuzzy_exponent_is_clamped_not_refused() {
        let traj = diag_traj(60, 1);
        let out = fuzzy_entropy_sum(&traj, 1, 0.2, 1000, 0);
        assert_eq!(out[3], MAX_FUZZY_N as f64);
    }

    #[test]
    fn fuzzy_pair_work_is_cut_to_the_budget() {
        // Same shape as the correlation budget cut: 20000 rows at dim 2 is
        // 4e8 pair terms, over the 2e8 budget.
        let traj = diag_traj(MAX_DIAG_N, 2);
        let out = fuzzy_entropy_sum(&traj, 2, 0.2, 2, 0);
        let used = out[0] as u64;
        assert!(used >= 1);
        assert!(used < MAX_DIAG_N as u64);
        let pairs = used * (used - 1) / 2;
        assert!(pairs * 2 <= MAX_DIAG_TERMS);
    }

    #[test]
    fn fuzzy_unusable_requests_return_an_empty_result() {
        let traj = diag_traj(100, 2);
        // Fewer than one complete row.
        assert!(fuzzy_entropy_sum(&traj[..1], 2, 0.1, 2, 0).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 2);
        bad[9] = f64::INFINITY;
        assert!(fuzzy_entropy_sum(&bad, 2, 0.1, 2, 0).is_empty());
        // A constant series leaves the fallback tolerance at zero.
        let flat = vec![0.5f64; 200];
        assert!(fuzzy_entropy_sum(&flat, 1, 0.0, 2, 0).is_empty());
    }

    #[test]
    fn ordinal_header_reports_the_values_actually_used() {
        let x = diag_traj(500, 1);
        let out = ordinal_distribution(&x, 3, 2);
        assert_eq!(out[0], 500.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 496.0); // 500 - (3 - 1) * 2 windows
        assert_eq!(out.len(), ORDINAL_HEADER + 6); // 3! = 6 patterns
        let total: f64 = out[ORDINAL_HEADER..].iter().sum();
        assert_eq!(total, out[3]);
    }

    #[test]
    fn ordinal_d_and_tau_are_clamped_not_refused() {
        let x = diag_traj(500, 1);
        let out = ordinal_distribution(&x, 20, 0);
        assert_eq!(out[1], MAX_ORDINAL_D as f64);
        assert_eq!(out[2], 1.0);
        assert_eq!(out.len(), ORDINAL_HEADER + 40320); // 8! patterns
    }

    #[test]
    fn ordinal_the_sample_cap_is_the_budget() {
        let x = diag_traj(MAX_DIAG_N + 5000, 1);
        let out = ordinal_distribution(&x, 3, 1);
        assert_eq!(out[0], MAX_DIAG_N as f64);
    }

    #[test]
    fn ordinal_unusable_requests_return_an_empty_result() {
        let x = diag_traj(100, 1);
        // Too short for a single window at d = 3, tau = 200.
        assert!(ordinal_distribution(&x, 3, 200).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 1);
        bad[42] = f64::NAN;
        assert!(ordinal_distribution(&bad, 3, 1).is_empty());
        // An empty series.
        assert!(ordinal_distribution(&[], 3, 1).is_empty());
    }

    #[test]
    fn ordinal_a_monotone_series_lands_in_one_pattern() {
        let x: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let out = ordinal_distribution(&x, 3, 2);
        // An ascending window always sorts to the identity permutation, so
        // every window lands in a single Lehmer code.
        let nonzero = out[ORDINAL_HEADER..].iter().filter(|&&c| c > 0.0).count();
        assert_eq!(nonzero, 1);
        assert_eq!(out[ORDINAL_HEADER], 496.0);
    }

    #[test]
    fn fit_diag_budget_leaves_small_requests_alone() {
        assert_eq!(fit_diag_budget(100, 2, 0, false), 100);
        assert_eq!(fit_diag_budget(100, 2, 0, true), 100);
    }

    #[test]
    fn fit_diag_budget_counts_self_pairs_for_apen() {
        // ApEn compares every row to every row: n^2 terms at dim 1.
        let used = fit_diag_budget(MAX_DIAG_N, 1, 0, true);
        assert!((used as u64) * (used as u64) <= MAX_DIAG_TERMS);
        assert!(used < MAX_DIAG_N);
    }

    #[test]
    fn fit_diag_budget_counts_valid_pairs_for_theiler() {
        // At dim 2 the budget binds below 20000 rows. A Theiler window
        // removes pairs, so more rows fit it.
        let plain = fit_diag_budget(MAX_DIAG_N, 2, 0, false);
        let windowed = fit_diag_budget(MAX_DIAG_N, 2, 100, false);
        assert!(plain < MAX_DIAG_N);
        assert!(windowed > plain);
    }

    #[test]
    fn fit_diag_budget_keeps_every_row_that_fits() {
        for (theiler_window, self_pairs) in [(0, false), (100, false), (0, true)] {
            let used = fit_diag_budget(MAX_DIAG_N, 2, theiler_window, self_pairs);
            // The fitted count fits, and one more row would not.
            assert_eq!(fit_diag_budget(used, 2, theiler_window, self_pairs), used);
            assert!(fit_diag_budget(used + 1, 2, theiler_window, self_pairs) < used + 1);
        }
    }

    #[test]
    fn diagnostics_dim_is_clamped_not_refused() {
        let traj = diag_traj(40, MAX_DIAG_DIM);
        // Dimension 0 reads the trajectory as one coordinate per row.
        assert_eq!(correlation_counts(&traj, 0, &[0.1], 0, true)[1], 1.0);
        assert_eq!(apen_counts(&traj, 0, 0.1)[1], 1.0);
        assert_eq!(fuzzy_entropy_sum(&traj, 0, 0.1, 2, 0)[1], 1.0);
        // A dimension above the cap is cut to the cap.
        let cap = MAX_DIAG_DIM as f64;
        assert_eq!(correlation_counts(&traj, 1000, &[0.1], 0, true)[1], cap);
        assert_eq!(apen_counts(&traj, 1000, 0.1)[1], cap);
        assert_eq!(fuzzy_entropy_sum(&traj, 1000, 0.1, 2, 0)[1], cap);
    }

    #[test]
    fn diagnostics_theiler_window_is_clamped_not_refused() {
        let traj = diag_traj(30, 1);
        assert_eq!(correlation_counts(&traj, 1, &[0.1], 1000, true)[3], 29.0);
        assert_eq!(fuzzy_entropy_sum(&traj, 1, 0.1, 2, 1000)[4], 29.0);
    }

    #[test]
    fn diagnostics_row_cap_binds_when_a_theiler_window_frees_the_budget() {
        // At dim 1 a Theiler window of 10000 leaves few enough pairs that
        // 25000 rows fit the pair budget, so only the row cap stops them.
        let traj = diag_traj(MAX_DIAG_N + 5000, 1);
        let cap = MAX_DIAG_N as f64;
        assert_eq!(correlation_counts(&traj, 1, &[0.1], 10_000, true)[0], cap);
        assert_eq!(fuzzy_entropy_sum(&traj, 1, 0.1, 2, 10_000)[0], cap);
    }

    #[test]
    fn tolerance_fallback_is_a_fifth_of_the_sample_std() {
        // Mean 2.5, squared deviations sum to 5, ddof = 1 gives variance 5/3.
        let expected = 0.2 * (5.0f64 / 3.0).sqrt();
        assert_eq!(
            tolerance_or_default(f64::NAN, &[1.0, 2.0, 3.0, 4.0]),
            Some(expected)
        );
        assert_eq!(tolerance_or_default(0.3, &[1.0, 2.0, 3.0, 4.0]), Some(0.3));
    }
}
