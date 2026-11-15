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

/// `GpuDraw::flags`: single-sided, so meshlets facing away may be skipped.
pub const DRAW_CONE_CULLING: u32 = 1;
/// `GpuDraw::flags`: occlusion culling, the early pass drew this instance's visible meshlets.
pub const DRAW_EARLY: u32 = 2;

/// `MeshletDrawPushConstants::flags`.
pub const MESHLET_FRUSTUM_CULLING: u32 = 1;
pub const MESHLET_CONE_CULLING: u32 = 2;

/// Per indirect draw of the meshlet path (task and mesh stages): which list of work items to run.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshletDrawPushConstants {
    pub draws: [u32; 2],
    /// Bucket b's work items start at `b * work_capacity`.
    pub work: [u32; 2],
    pub counters: [u32; 2],
    /// The meshlet visibility bits (occlusion culling).
    pub visibility: [u32; 2],
    pub bucket: u32,
    pub work_capacity: u32,
    pub flags: u32,
    /// Cluster LOD: the error, in pixels, a meshlet may show.
    pub lod_threshold: f32,
}
const _: () = assert!(size_of::<MeshletDrawPushConstants>() == 48);

/// What one cull pass produced (DrawCounters): the work per bucket, and the indirect dispatch arguments of
/// each bucket's task shader (`VkDrawMeshTasksIndirectCommandEXT`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DrawCounters {
    pub draw_count: u32,
    pub work_count: [u32; 2],
    pub padding: u32,
    pub args: [[u32; 4]; 2],
}
const _: () = assert!(size_of::<DrawCounters>() == 48);

/// Byte offset of bucket `bucket`'s indirect arguments inside a `DrawCounters`.
pub const fn counters_args_offset(bucket: u32) -> u64 {
    16 + 16 * bucket as u64
}
