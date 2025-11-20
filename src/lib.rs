pub mod metrics;
pub mod metrics_simd;
pub mod hnsw_index;
pub mod sparse_metrics;
pub mod sparse_hnsw_index;
pub mod hnsw_algo;

#[cfg(not(test))]
#[pyo3::pymodule]
fn _hnsw_backend(_py: pyo3::Python, m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<hnsw_index::HnswIndex>()?;
    m.add_class::<sparse_hnsw_index::SparseHnswIndex>()?;
    Ok(())
}