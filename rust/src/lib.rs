use pyo3::prelude::*;

mod ami;
mod correlation_gp;
mod embedding;
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
    m.add_function(wrap_pyfunction!(embedding::cao_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::fnn_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_gp::correlation_counts, m)?)?;
    Ok(())
}
