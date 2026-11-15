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

/// `GpuInstance::mesh`: an empty slot, or an object the CPU draws (blended).
pub const NO_MESH: u32 = 0xFFFF_FFFF;
/// `GpuInstance::flags`: single-sided and not mirrored, drawn by the pipeline that culls back faces.
pub const INSTANCE_CULL_BACK_FACES: u32 = 1;
/// `CullPushConstants::flags`.
pub const CULL_FRUSTUM: u32 = 1;

/// One scene object, an entry of the instance array indexed by the object's slot (`ObjectId::index`), which
/// the GPU scene keeps on the GPU across frames.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuInstance {
    pub model: Mat4,
    /// Index into the frame's `GpuMesh` array; `NO_MESH`: an empty slot, or an object the CPU draws.
    pub mesh: u32,
    pub flags: u32,
    /// First word of the slot's meshlet visibility bits.
    pub visibility_offset: u32,
    pub padding: u32,
}
const _: () = assert!(size_of::<GpuInstance>() == 80);

impl Default for GpuInstance {
    fn default() -> Self {
        Self { model: Mat4::IDENTITY, mesh: NO_MESH, flags: 0, visibility_offset: 0, padding: 0 }
    }
}

/// One mesh of the frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMesh {
    /// Object space: xyz centre, w radius, enclosing the mesh.
    pub sphere: Vec4,
    pub vertices: [u32; 2],
    /// Meshlet path: its meshlets followed by their data.
    pub meshlets: [u32; 2],
    /// Of every LOD level.
    pub meshlet_count: u32,
    /// Vertex path.
    pub index_count: u32,
    /// Vertex path: where its indices start in the index pool.
    pub first_index: u32,
    pub padding: u32,
}
const _: () = assert!(size_of::<GpuMesh>() == 48);

/// Push constants of the instance cull pass (compute stage).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CullPushConstants {
    pub instances: [u32; 2],
    pub meshes: [u32; 2],
    /// `GpuDraw`s, appended to.
    pub draws: [u32; 2],
    /// Work items (meshlet path) or draw commands (vertex path), appended to per bucket.
    pub work: [u32; 2],
    pub counters: [u32; 2],
    /// Per slot: visible at the end of the previous frame (occlusion culling).
    pub history: [u32; 2],
    pub instance_count: u32,
    /// Entries per bucket.
    pub work_capacity: u32,
    pub flags: u32,
    pub padding: u32,
}
const _: () = assert!(size_of::<CullPushConstants>() == 64);

/// A changed instance for the scene update pass to write to its slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceUpdate {
    pub instance: GpuInstance,
    pub slot: u32,
    pub padding: [u32; 3],
}
const _: () = assert!(size_of::<InstanceUpdate>() == 96);

/// A changed material for the scene update pass to write to its slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialUpdate {
    pub material: phox_core::shader_interface::MaterialData,
    pub slot: u32,
    pub padding: [u32; 3],
}
const _: () = assert!(size_of::<MaterialUpdate>() == 96);

/// Push constants of the scene update pass (compute stage).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SceneUpdatePushConstants {
    pub instance_updates: [u32; 2],
    pub material_updates: [u32; 2],
    pub instances: [u32; 2],
    pub materials: [u32; 2],
    pub instance_update_count: u32,
    pub material_update_count: u32,
}
const _: () = assert!(size_of::<SceneUpdatePushConstants>() == 40);

/// Push constants of the draw arguments pass (compute stage).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ArgsPushConstants {
    pub counters: [u32; 2],
    /// The device's largest task shader dispatch in x.
    pub max_groups_x: u32,
    pub padding: u32,
}
const _: () = assert!(size_of::<ArgsPushConstants>() == 16);
