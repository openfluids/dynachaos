use pyo3::prelude::*;

mod ami;
mod correlation_gp;
mod embedding;
mod entropy;
mod multifractal;
mod permutation;
mod recurrence;

/// Rust-accelerated backends for dynachaos.
///
/// This module is optional — all algorithms have pure-Python fallbacks.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recurrence::diagonal_lines, m)?)?;
    m.add_function(wrap_pyfunction!(recurrence::vertical_lines, m)?)?;
    m.add_function(wrap_pyfunction!(permutation::ordinal_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(ami::ami_histogram, m)?)?;
    // Cao/FNN nearest-neighbor statistics remain in Python/SciPy; cKDTree is faster.
    m.add_function(wrap_pyfunction!(embedding::select_dimension_cao, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_gp::correlation_counts, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::fuzzy_entropy_sum, m)?)?;
    m.add_function(wrap_pyfunction!(multifractal::multifractal_moments, m)?)?;
    Ok(())
}
