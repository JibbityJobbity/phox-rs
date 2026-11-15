//! The render graph: passes declare how they use transient and imported resources; the graph culls what no
//! live pass needs, derives usages and barriers, creates the transients from a pool and records the frame
//! through an [`phox_rhi::Device`].
//!
//! A frame is declared on a [`GraphBuilder`] made by [`RenderGraph::begin`]. The `RenderGraph` itself only
//! holds the transient pool, so keep one per device and reuse it every frame.

mod barriers;
mod builder;
mod compile;
mod dot;
mod execute;

pub mod component;

pub use builder::{BufferAccess, GraphBuilder, PassBuilder, PassContext, TextureAccess};
pub use component::{Component, Technique};

use phox_rhi::{BufferHandle, BufferUsage, Device, Feature, ResourceState, RhiError, TextureHandle, TextureUsage};
use std::sync::Arc;

/// A texture declared in the current frame's graph. Only meaningful for the builder that made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphTexture {
    pub id: u32,
}

/// A buffer declared in the current frame's graph. Same lifetime rules as [`GraphTexture`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphBuffer {
    pub id: u32,
}

impl Default for GraphTexture {
    /// An invalid handle.
    fn default() -> Self {
        Self { id: u32::MAX }
    }
}
impl Default for GraphBuffer {
    fn default() -> Self {
        Self { id: u32::MAX }
    }
}

/// What can go wrong declaring, compiling or executing a graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    /// A pass or resource was declared wrongly; the message names the pass and resource.
    #[error("{0}")]
    Declaration(String),
    /// The device lacks features the graph requires. Lists every missing one, once, in declaration order.
    #[error("{message}")]
    MissingFeatures { message: String, features: Vec<Feature> },
    #[error("the graph has not been compiled: call compile() first")]
    NotCompiled,
    #[error(transparent)]
    Device(#[from] RhiError),
}

impl GraphError {
    /// The features a [`GraphError::MissingFeatures`] reports; empty for other errors.
    pub fn missing_features(&self) -> &[Feature] {
        match self {
            GraphError::MissingFeatures { features, .. } => features,
            _ => &[],
        }
    }
}

pub type GraphResult<T> = std::result::Result<T, GraphError>;

const POOL_RETIRE_FRAMES: u64 = 3;

struct PooledTexture {
    desc: phox_rhi::TextureDesc,
    handle: TextureHandle,
    last_used: u64,
    in_use: bool,
}

struct PooledBuffer {
    desc: phox_rhi::BufferDesc,
    handle: BufferHandle,
    last_used: u64,
    in_use: bool,
}

/// Owns the pool of transient resources. Keep one per device and call [`RenderGraph::begin`] every frame.
pub struct RenderGraph {
    device: Arc<dyn Device>,
    textures: Vec<PooledTexture>,
    buffers: Vec<PooledBuffer>,
    frame: u64,
}

impl RenderGraph {
    pub fn new(device: Arc<dyn Device>) -> Self {
        Self { device, textures: Vec::new(), buffers: Vec::new(), frame: 0 }
    }

    pub fn device(&self) -> &Arc<dyn Device> {
        &self.device
    }

    /// Starts a new frame's declarations, dropping the previous frame's.
    pub fn begin(&mut self) -> GraphBuilder<'_> {
        GraphBuilder::new(self)
    }
}

impl Drop for RenderGraph {
    fn drop(&mut self) {
        for t in &self.textures {
            self.device.destroy_texture(t.handle);
        }
        for b in &self.buffers {
            self.device.destroy_buffer(b.handle);
        }
    }
}

// ---- the declarations of one frame ----

pub(crate) struct TextureNode {
    pub desc: phox_rhi::TextureDesc,
    pub name: String,
    pub imported: bool,
    /// Imported: given. Transient: set by compile() when a live pass uses it.
    pub handle: TextureHandle,
    pub before: ResourceState,
    pub after: ResourceState,
}

pub(crate) struct BufferNode {
    pub desc: phox_rhi::BufferDesc,
    pub name: String,
    pub imported: bool,
    pub handle: BufferHandle,
    pub before: ResourceState,
    pub after: ResourceState,
}

pub(crate) struct TextureUse {
    pub id: u32,
    pub state: ResourceState,
    pub usage: TextureUsage,
    pub write: bool,
}

pub(crate) struct BufferUse {
    pub id: u32,
    pub state: ResourceState,
    pub usage: BufferUsage,
    pub write: bool,
    /// HostWrite: the CPU wrote it before submit, so no barrier and no state tracking.
    pub host: bool,
}

pub(crate) struct ColorDecl {
    pub id: u32,
    pub clear: bool,
    pub value: [f32; 4],
}

type ExecuteFn<'g> = Box<dyn FnOnce(&mut PassContext<'_>) -> GraphResult<()> + 'g>;

/// What a pass's setup declared.
#[derive(Default)]
pub(crate) struct PassDecl {
    pub name: String,
    /// Includes colour and depth targets.
    pub textures: Vec<TextureUse>,
    pub buffers: Vec<BufferUse>,
    pub colors: Vec<ColorDecl>,
    pub depth: Option<DepthDecl>,
    pub side_effect: bool,
}

pub(crate) struct PassNode<'g> {
    pub decl: PassDecl,
    pub execute: Option<ExecuteFn<'g>>,
    // Filled in by compile().
    pub live: bool,
    pub texture_barriers: Vec<phox_rhi::TextureBarrier>,
    pub buffer_barriers: Vec<phox_rhi::BufferBarrier>,
}

impl std::ops::Deref for PassNode<'_> {
    type Target = PassDecl;
    fn deref(&self) -> &PassDecl {
        &self.decl
    }
}

pub(crate) struct DepthDecl {
    pub id: u32,
    pub clear: bool,
    pub value: f32,
    pub read_only: bool,
}

#[derive(Default)]
pub(crate) struct GraphData<'g> {
    pub textures: Vec<TextureNode>,
    pub buffers: Vec<BufferNode>,
    pub passes: Vec<PassNode<'g>>,
    pub required: Vec<Feature>,
    pub compiled: bool,
    pub final_texture_barriers: Vec<phox_rhi::TextureBarrier>,
    pub final_buffer_barriers: Vec<phox_rhi::BufferBarrier>,
}

pub(crate) fn label(name: &str, kind: &str, id: u32) -> String {
    if name.is_empty() { format!("{kind}#{id}") } else { format!("'{name}'") }
}
