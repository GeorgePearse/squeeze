use pyo3::prelude::*;

pub mod metrics;
pub mod hnsw_index;
pub mod sparse_metrics;
pub mod sparse_hnsw_index;
pub mod hnsw_algo;

#[cfg(not(test))]
#[pymodule]
fn _hnsw_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hnsw_index::HnswIndex>()?;
    m.add_class::<sparse_hnsw_index::SparseHnswIndex>()?;
    Ok(())
}