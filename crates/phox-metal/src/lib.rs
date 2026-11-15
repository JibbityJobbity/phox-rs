//! The Metal backend. Stubbed: it is not part of the first Rust pass, so creating a device always fails
//! with [`phox_rhi::RhiError::Unsupported`].

use phox_rhi::{Result, RhiError};

/// Always fails: the Metal backend is not implemented yet.
pub fn create_device() -> Result<std::convert::Infallible> {
    Err(RhiError::Unsupported("the Metal backend is not implemented in this version of Phox".into()))
}
