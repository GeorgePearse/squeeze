//! Error types shared by Squeeze algorithms.

use thiserror::Error;

/// Errors returned by dimensionality-reduction algorithms.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("algorithm has not been fitted")]
    NotFitted,

    #[error("numerical computation failed: {0}")]
    Computation(String),
}

/// Result type used throughout the public API.
pub type Result<T> = std::result::Result<T, Error>;
