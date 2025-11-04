use pyo3::prelude::*;

pub mod metrics;
pub mod hnsw_index;

#[pymodule]
fn _hnsw_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hnsw_index::HnswIndex>()?;
    Ok(())
}
