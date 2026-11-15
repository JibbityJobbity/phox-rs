//! The GPU-driven scene's structures, as shaders/common.glsl declares them (std430).

use bytemuck::{Pod, Zeroable};
use phox_core::math::{Mat4, Vec4};

/// One instance the cull pass decided to draw (GpuDraw): its row in the draw list that the vertex shader
/// reads at `gl_InstanceIndex`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuDraw {
    pub model: Mat4,
    /// xyz: the camera in the instance's object space.
    pub camera_position: Vec4,
    /// Device address of the mesh's vertex array.
    pub vertices: [u32; 2],
    /// Meshlet path: the address of the mesh's meshlets; unused on the vertex path.
    pub primitives: [u32; 2],
    /// Meshlets, or indices.
    pub count: u32,
    /// Index into the material array.
    pub material: u32,
    /// First word of meshlet visibility bits.
    pub visibility_offset: u32,
    /// `DRAW_CONE_CULLING | DRAW_EARLY`.
    pub flags: u32,
}
const _: () = assert!(size_of::<GpuDraw>() == 112);

/// A VkDrawIndexedIndirectCommand; `first_instance` is the draw (row of the draw list).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
const _: () = assert!(size_of::<DrawCommand>() == 20);

/// The vertex shader's push constants (VertexDrawPushConstants): where the draw list is.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VertexDrawPushConstants {
    pub draws: [u32; 2],
}

pub fn split_address(address: u64) -> [u32; 2] {
    [address as u32, (address >> 32) as u32]
}
