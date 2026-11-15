use phox_core::graph::GraphError;
use phox_rhi::RhiError;

#[derive(Debug, thiserror::Error)]
pub enum ForwardError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Device(#[from] RhiError),
    #[error(transparent)]
    Core(#[from] phox_core::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ForwardError>;
