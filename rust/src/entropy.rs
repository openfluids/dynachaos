//! Fuzzy entropy pairwise-membership accumulator (Chen et al., 2007).
//!
//! Computes the upper-triangle sum of fuzzy memberships
//! `exp(-(d / r)^n)` using Chebyshev distance for all valid pairs with
//! `j > i + theiler_window`.
//!
//! Reference: Chen, W. et al. (2007), Medical Engineering & Physics 29(2), 164-169.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute the total fuzzy-membership sum over all valid template pairs.
///
/// Parameters
/// ----------
/// traj : numpy.ndarray of float64, shape (N, d)
///     Embedded trajectory templates. Must be C-contiguous.
/// r : float
///     Positive tolerance scale in the fuzzy membership.
/// n : int
///     Fuzzy exponent in `exp(-(d / r)^n)`.
/// theiler_window : int
///     Minimum temporal separation: only pairs with `j > i + theiler_window`.
///
/// Returns
/// -------
/// float
///     Sum of fuzzy memberships over all valid upper-triangle pairs.
#[pyfunction]
#[pyo3(signature = (traj, r, n, theiler_window = 0))]
pub fn fuzzy_entropy_sum<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray2<'py, f64>,
    r: f64,
    n: u32,
    theiler_window: usize,
) -> PyResult<f64> {
    if r <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "r must be positive",
        ));
    }

    let traj_arr = traj.as_array();
    let n_pts = traj_arr.shape()[0];
    let dim = traj_arr.shape()[1];

    // Raw slice access: avoids ndarray's bounds-checked Index trait.
    let traj_slice = traj_arr.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("traj must be C-contiguous")
    })?;

    // Copy data to owned Vec so we can release the GIL.
    let traj_owned: Vec<f64> = traj_slice.to_vec();
    let n_f = n as f64;

    // Release the GIL and parallelise with rayon.
    let total = py.detach(|| {
        (0..n_pts)
            .into_par_iter()
            .fold(
                || 0.0_f64,
                |mut local_sum, i| {
                    let j_start = i.saturating_add(theiler_window).saturating_add(1);
                    if j_start >= n_pts {
                        return local_sum;
                    }

                    let row_i = i * dim;
                    for j in j_start..n_pts {
                        let row_j = j * dim;
                        let mut d_max = 0.0_f64;
                        for k in 0..dim {
                            let diff = (traj_owned[row_i + k] - traj_owned[row_j + k]).abs();
                            if diff > d_max {
                                d_max = diff;
                            }
                        }
                        let scaled = d_max / r;
                        local_sum += (-(scaled.powf(n_f))).exp();
                    }
                    local_sum
                },
            )
            .reduce(|| 0.0_f64, |a, b| a + b)
    });

    Ok(total)
}
