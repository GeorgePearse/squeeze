pub mod metrics;
pub mod metrics_simd;
pub mod sparse_metrics;
pub mod hnsw_algo;
pub mod hnsw_index;
pub mod sparse_hnsw_index;
pub mod mixed_precision;
pub mod cache_aligned;

// Dimensionality reduction algorithms
pub mod pca;
pub mod tsne;
pub mod mds;
pub mod isomap;
pub mod lle;
pub mod phate;
pub mod trimap;
pub mod pacmap;

#[cfg(not(test))]
#[pyo3::pymodule]
fn _hnsw_backend(_py: pyo3::Python, m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    // HNSW index classes
    m.add_class::<hnsw_index::HnswIndex>()?;
    m.add_class::<sparse_hnsw_index::SparseHnswIndex>()?;
    
    // Dimensionality reduction algorithms
    m.add_class::<pca::PCA>()?;
    m.add_class::<tsne::TSNE>()?;
    m.add_class::<mds::MDS>()?;
    m.add_class::<isomap::Isomap>()?;
    m.add_class::<lle::LLE>()?;
    m.add_class::<phate::PHATE>()?;
    m.add_class::<trimap::TriMap>()?;
    m.add_class::<pacmap::PaCMAP>()?;
    
    Ok(())
}
