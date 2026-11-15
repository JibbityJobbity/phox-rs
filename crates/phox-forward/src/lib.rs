//! The forward technique: a scene drawn with indexed indirect draws into an HDR target, tone mapped into the
//! display format. Its passes are graph components written once against `phox_rhi`; [`ForwardRenderer`]
//! drives them headless on any device.

mod error;
mod frame;
mod gpu_types;
mod mesh_cache;
mod passes;
mod renderer;
pub mod shaders;
mod technique;
mod texture_cache;

pub use error::{ForwardError, Result};
pub use gpu_types::{DrawCommand, GpuDraw, VertexDrawPushConstants};
pub use mesh_cache::{GpuMeshInfo, MeshCache};
pub use passes::{DEPTH_FORMAT, ForwardPass, HDR_FORMAT, HEADLESS_FORMAT, ReadbackPass, TonemapPass};
pub use renderer::{ForwardRenderer, PassTiming, RendererConfig};
pub use texture_cache::TextureCache;
pub use technique::{ForwardTechnique, FrameInputs};
