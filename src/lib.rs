//! CPU-optimized dimensionality reduction algorithms.
//!
//! The public API is intentionally split into algorithms, nearest-neighbor
//! infrastructure, and reusable numerical primitives.

mod barnes_hut;
mod hnsw_algo;
mod isomap;
mod lle;
mod mds;
mod metrics;
mod metrics_simd;
mod pacmap;
mod pca;
mod phate;
mod sparse_metrics;
mod trimap;
mod tsne;

pub mod error;

/// Dimensionality-reduction algorithms.
pub mod algorithms {
    pub use crate::isomap::Isomap;
    pub use crate::lle::LLE;
    pub use crate::mds::MDS;
    pub use crate::pacmap::PaCMAP;
    pub use crate::pca::PCA;
    pub use crate::phate::PHATE;
    pub use crate::trimap::TriMap;
    pub use crate::tsne::TSNE;
}

/// Approximate nearest-neighbor infrastructure shared by the algorithms.
pub mod neighbors {
    pub use crate::hnsw_algo::{Hnsw, Node, PruneStrategy};
}

/// Distance functions and other reusable numerical building blocks.
pub mod distance {
    pub use crate::metrics::{
        chebyshev, cosine, euclidean, hamming, manhattan, minkowski, MetricError, MetricResult,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::metrics_simd::has_avx2;
    #[cfg(target_arch = "aarch64")]
    pub use crate::metrics_simd::has_neon;
    pub use crate::metrics_simd::{
        cosine as cosine_simd, euclidean as euclidean_simd, has_simd, manhattan as manhattan_simd,
    };
    pub use crate::sparse_metrics::{sparse_cosine, sparse_euclidean, sparse_manhattan};
}

pub use error::{Error, Result};
