//! Coupled-map lattice kernels.

use crate::CoreError;

/// Compute the logistic CML subsystem Jacobian for sites 0..l-1.
///
/// Returns the row-major flattened l x l block. `x` is the full lattice.
pub fn cml_jacobian_logistic(x: &[f64], a: f64, eps: f64, l: usize) -> Result<Vec<f64>, CoreError> {
    let n = x.len();
    if l < 1 || l > n {
        return Err(CoreError::invalid_argument(format!(
            "L must satisfy 1 <= L <= N (got L={l}, N={n})"
        )));
    }

    let matrix_len = l
        .checked_mul(l)
        .ok_or_else(|| CoreError::invalid_argument("L is too large for an L x L matrix"))?;
    let mut jacobian = vec![0.0_f64; matrix_len];
    let diagonal_scale = 1.0 - eps;
    let off_diagonal_scale = eps / 2.0;

    for i in 0..l {
        let row = i * l;
        jacobian[row + i] = diagonal_scale * logistic_derivative(x[i], a);

        let i_left = if i == 0 { n - 1 } else { i - 1 };
        if i_left < l {
            jacobian[row + i_left] = off_diagonal_scale * logistic_derivative(x[i_left], a);
        }

        let i_right = (i + 1) % n;
        if i_right < l {
            jacobian[row + i_right] = off_diagonal_scale * logistic_derivative(x[i_right], a);
        }
    }

    Ok(jacobian)
}

#[inline]
fn logistic_derivative(x: f64, a: f64) -> f64 {
    -2.0 * a * x
}
