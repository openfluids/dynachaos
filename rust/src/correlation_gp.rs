//! Exact all-pairs correlation integral (Grassberger-Procaccia).
//!
//! Streams through all valid pairs (i, j) with |i-j| > theiler_window,
//! computes the distance, and increments count bins for each r threshold
//! exceeded.  Uses O(n_r) memory for the count array — no distance matrix.
//!
//! Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), 189-208.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Count pairs within each distance threshold (exact all-pairs).
///
/// Parameters
/// ----------
/// traj : numpy.ndarray of float64, shape (N, d)
///     Trajectory points (already embedded if needed).  Must be C-contiguous.
/// r_values : numpy.ndarray of float64, shape (n_r,)
///     Distance thresholds (must be sorted ascending).
/// theiler_window : int
///     Minimum temporal separation: only pairs with |i-j| > w are counted.
/// use_chebyshev : bool
///     If True, use Chebyshev (max-norm); else Euclidean.
///
/// Returns
/// -------
/// numpy.ndarray of int64, shape (n_r,)
///     counts[k] = number of valid pairs with dist < r_values[k].
#[pyfunction]
#[pyo3(signature = (traj, r_values, theiler_window = 0, use_chebyshev = true))]
pub fn correlation_counts<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray2<'py, f64>,
    r_values: PyReadonlyArray1<'py, f64>,
    theiler_window: usize,
    use_chebyshev: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let traj_arr = traj.as_array();
    let n = traj_arr.shape()[0];
    let dim = traj_arr.shape()[1];

    let r_slice = r_values.as_slice()?;
    let n_r = r_slice.len();

    // Runtime validation: r_values must be sorted ascending
    if !r_slice.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "r_values must be sorted in ascending order",
        ));
    }

    if n_r == 0 {
        return Ok(PyArray1::from_vec(py, vec![0i64; 0]));
    }

    // Raw slice access: avoids ndarray's bounds-checked Index trait.
    let traj_slice = traj_arr.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("traj must be C-contiguous")
    })?;

    // Copy data to owned Vecs so we can release the GIL.
    let traj_owned: Vec<f64> = traj_slice.to_vec();
    let r_owned: Vec<f64> = r_slice.to_vec();

    // Pre-compute squared thresholds for Euclidean mode (eliminates sqrt).
    let r_sq: Vec<f64> = r_owned.iter().map(|&r| r * r).collect();

    let r_max = r_owned[n_r - 1];
    let r_max_sq = r_max * r_max;

    // Release the GIL and parallelise with rayon.
    let diff_counts = py.detach(|| {
        if use_chebyshev {
            // ── Chebyshev (max-norm) path ──
            (0..n)
                .into_par_iter()
                .fold(
                    || vec![0i64; n_r],
                    |mut local, i| {
                        let j_start = i.saturating_add(theiler_window).saturating_add(1);
                        if j_start >= n {
                            return local;
                        }
                        let row_i = i * dim;
                        for j in j_start..n {
                            let row_j = j * dim;
                            let mut d_max = 0.0f64;
                            let mut skip = false;
                            for k in 0..dim {
                                let diff = (traj_owned[row_i + k] - traj_owned[row_j + k]).abs();
                                if diff > r_max {
                                    skip = true;
                                    break;
                                }
                                if diff > d_max {
                                    d_max = diff;
                                }
                            }
                            if skip {
                                continue;
                            }
                            let pos = r_owned.partition_point(|&r| r <= d_max);
                            if pos < n_r {
                                local[pos] += 1;
                            }
                        }
                        local
                    },
                )
                .reduce(
                    || vec![0i64; n_r],
                    |mut a, b| {
                        a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                        a
                    },
                )
        } else {
            // ── Euclidean path (squared distances, no sqrt) ──
            (0..n)
                .into_par_iter()
                .fold(
                    || vec![0i64; n_r],
                    |mut local, i| {
                        let j_start = i.saturating_add(theiler_window).saturating_add(1);
                        if j_start >= n {
                            return local;
                        }
                        let row_i = i * dim;
                        for j in j_start..n {
                            let row_j = j * dim;
                            let mut sq = 0.0f64;
                            let mut skip = false;
                            for k in 0..dim {
                                let diff = traj_owned[row_i + k] - traj_owned[row_j + k];
                                sq += diff * diff;
                                if sq > r_max_sq {
                                    skip = true;
                                    break;
                                }
                            }
                            if skip {
                                continue;
                            }
                            let pos = r_sq.partition_point(|&rsq| rsq <= sq);
                            if pos < n_r {
                                local[pos] += 1;
                            }
                        }
                        local
                    },
                )
                .reduce(
                    || vec![0i64; n_r],
                    |mut a, b| {
                        a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                        a
                    },
                )
        }
    });

    // Forward prefix sum: convert differential counts to cumulative.
    // counts[k] = #{pairs with dist < r_values[k]}.
    let mut counts = diff_counts;
    for k in 1..n_r {
        counts[k] += counts[k - 1];
    }

    Ok(PyArray1::from_vec(py, counts))
}
