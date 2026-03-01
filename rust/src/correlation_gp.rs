//! Exact all-pairs correlation integral (Grassberger-Procaccia).
//!
//! Streams through all valid pairs (i, j) with |i-j| > theiler_window,
//! computes the distance, and increments count bins for each r threshold
//! exceeded.  Uses O(n_r) memory for the count array — no distance matrix.
//!
//! Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), 189-208.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Count pairs within each distance threshold (exact all-pairs).
///
/// Parameters
/// ----------
/// traj : numpy.ndarray of float64, shape (N, d)
///     Trajectory points (already embedded if needed).
/// r_values : numpy.ndarray of float64, shape (n_r,)
///     Distance thresholds (should be sorted ascending).
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

    debug_assert!(
        r_slice.windows(2).all(|w| w[0] <= w[1]),
        "r_values must be sorted ascending"
    );

    let mut counts = vec![0i64; n_r];

    // Find the maximum r value for early skipping
    let r_max = if n_r > 0 {
        r_slice[n_r - 1]
    } else {
        return Ok(PyArray1::from_vec(py, counts));
    };

    for i in 0..n {
        let j_start = i + theiler_window + 1;
        if j_start >= n {
            continue;
        }

        for j in j_start..n {
            // Compute distance
            let dist = if use_chebyshev {
                // Chebyshev (max-norm) with early exit
                let mut d_max = 0.0f64;
                let mut skip = false;
                for k in 0..dim {
                    let diff = (traj_arr[[i, k]] - traj_arr[[j, k]]).abs();
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
                d_max
            } else {
                // Euclidean with early exit
                let mut sq = 0.0f64;
                let r_max_sq = r_max * r_max;
                for k in 0..dim {
                    let diff = traj_arr[[i, k]] - traj_arr[[j, k]];
                    sq += diff * diff;
                    if sq > r_max_sq {
                        break;
                    }
                }
                if sq > r_max_sq {
                    continue;
                }
                sq.sqrt()
            };

            // Increment counts: find first r_values[k] > dist using binary search,
            // then increment counts[k..n_r]
            let pos = r_slice.partition_point(|&r| r <= dist);
            for c in &mut counts[pos..n_r] {
                *c += 1;
            }
        }
    }

    Ok(PyArray1::from_vec(py, counts))
}
