use std::path::PathBuf;

/// Errors of the core crate. Phox returns these instead of panicking.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mesh index count {0} is not a multiple of 3")]
    IndexCount(usize),
    #[error("mesh index {index} is out of range ({vertices} vertices)")]
    IndexOutOfRange { index: u32, vertices: usize },
    #[error("texture '{0}' has no texels")]
    EmptyTexture(String),
    #[error("texture '{name}': expected {expected} bytes of RGBA, got {got}")]
    TextureSize { name: String, expected: usize, got: usize },
    #[error("no object with id {index}.{generation}")]
    StaleObject { index: u32, generation: u32 },
    #[error("too many objects")]
    TooManyObjects,
    #[error("cannot read {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("cannot write {path}: {source}")]
    Write { path: PathBuf, source: std::io::Error },
    #[error("{path}: {message}")]
    Format { path: PathBuf, message: String },
    #[error("{0} contains no triangles")]
    NoTriangles(PathBuf),
    #[error("{0} has nothing to draw")]
    NothingToDraw(PathBuf),
    #[error("{path} requires the unsupported extension {extension}")]
    UnsupportedExtension { path: PathBuf, extension: String },
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;
