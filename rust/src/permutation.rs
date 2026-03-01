//! Ordinal pattern distribution for permutation entropy.
//!
//! Encodes each sliding window's ordinal pattern (argsort permutation) as a
//! Lehmer-code integer and accumulates counts in a flat array.  This avoids
//! Python-level loops and hash-map overhead entirely.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute the argsort of a small window, then encode it as a Lehmer code.
///
/// This matches Python's `tuple(np.argsort(window))` convention:
/// the pattern records the *indices* that would sort the window.
#[inline]
fn pattern_index(window: &[f64]) -> usize {
    let d = window.len();

    // Step 1: argsort — indices that sort the window in ascending order.
    // For d ≤ 10, insertion sort is optimal (no allocation needed beyond stack).
    let mut indices = [0usize; 10]; // d ≤ 10 in practice
    for (i, slot) in indices.iter_mut().enumerate().take(d) {
        *slot = i;
    }
    // Insertion sort on indices by window value
    for i in 1..d {
        let key = indices[i];
        let key_val = window[key];
        let mut j = i;
        while j > 0 && window[indices[j - 1]] > key_val {
            indices[j] = indices[j - 1];
            j -= 1;
        }
        indices[j] = key;
    }

    // Step 2: Lehmer code of the argsort permutation.
    let mut index: usize = 0;
    let mut factor: usize = 1;
    for i in (0..d).rev() {
        let mut count = 0usize;
        for j in (i + 1)..d {
            if indices[j] < indices[i] {
                count += 1;
            }
        }
        index += count * factor;
        if i > 0 {
            factor *= d - i;
        }
    }
    index
}

/// Factorial of a small integer (d ≤ 10 in practice).
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Compute the ordinal pattern distribution of a time series.
///
/// Parameters
/// ----------
/// x : numpy.ndarray of float64, shape (N,)
///     Scalar time series.
/// d : int
///     Embedding dimension (pattern length), typically 3–7.
/// tau : int
///     Time delay between successive elements.
///
/// Returns
/// -------
/// counts : numpy.ndarray of int64, shape (d!,)
///     Raw counts for each ordinal pattern (indexed by Lehmer code).
/// n_windows : int
///     Total number of windows analysed.
#[pyfunction]
#[pyo3(signature = (x, d = 5, tau = 1))]
pub fn ordinal_distribution<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    d: usize,
    tau: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
    let arr = x.as_slice()?;
    let n = arr.len();
    let n_windows = n.saturating_sub((d - 1) * tau);
    let n_perm = factorial(d);

    let mut counts = vec![0i64; n_perm];
    let mut window_buf = vec![0.0f64; d];

    for i in 0..n_windows {
        // Gather the delayed window
        for j in 0..d {
            window_buf[j] = arr[i + j * tau];
        }
        let idx = pattern_index(&window_buf);
        counts[idx] += 1;
    }

    Ok((PyArray1::from_vec(py, counts), n_windows as i64))
}
