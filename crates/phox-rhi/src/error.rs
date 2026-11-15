/// Errors every backend reports.
#[derive(Debug, thiserror::Error)]
pub enum RhiError {
    /// A handle that was never valid, or whose resource has been destroyed.
    #[error("stale or invalid {kind} handle")]
    StaleHandle { kind: &'static str },
    /// A description that cannot be created (zero size, missing shader stage, ...).
    #[error("{0}")]
    InvalidDesc(String),
    /// The device lacks something the call needs; names the feature.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A call misused a resource (e.g. mapping GPU-only memory, writing out of bounds).
    #[error("{0}")]
    Misuse(String),
    /// The backend's own failure (a Vulkan result, an allocation, ...).
    #[error("{0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, RhiError>;
