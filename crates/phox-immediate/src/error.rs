use phox_core::graph::GraphError;
use phox_rhi::RhiError;

#[derive(Debug, thiserror::Error)]
pub enum ImmediateError {
    #[error(transparent)]
    Device(#[from] RhiError),
    #[error(transparent)]
    Graph(#[from] GraphError),
}

pub type Result<T> = std::result::Result<T, ImmediateError>;
