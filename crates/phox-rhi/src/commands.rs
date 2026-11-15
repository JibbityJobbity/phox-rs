use crate::formats::ShaderStages;
use crate::handles::*;
use std::any::Any;

/// The state a resource is in between graph passes; each implies a layout, stages and accesses in the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ResourceState {
    #[default]
    Undefined,
    ColorAttachment,
    DepthWrite,
    DepthRead,
    ShaderRead,
    ShaderWrite,
    TransferSrc,
    TransferDst,
    VertexOrIndex,
    Uniform,
    /// Indirect draw or dispatch arguments, which shaders may read too.
    IndirectArgs,
    /// Read by the CPU once the frame has completed (a final state, for readback buffers).
    HostRead,
    Present,
}

impl ResourceState {
    pub const fn name(self) -> &'static str {
        match self {
            ResourceState::Undefined => "Undefined",
            ResourceState::ColorAttachment => "ColorAttachment",
            ResourceState::DepthWrite => "DepthWrite",
            ResourceState::DepthRead => "DepthRead",
            ResourceState::ShaderRead => "ShaderRead",
            ResourceState::ShaderWrite => "ShaderWrite",
            ResourceState::TransferSrc => "TransferSrc",
            ResourceState::TransferDst => "TransferDst",
            ResourceState::VertexOrIndex => "VertexOrIndex",
            ResourceState::Uniform => "Uniform",
            ResourceState::IndirectArgs => "IndirectArgs",
            ResourceState::HostRead => "HostRead",
            ResourceState::Present => "Present",
        }
    }

    /// States whose accesses write. Two consecutive accesses in the same write state still need a barrier,
    /// because the second one must wait for (and see) the first one's writes.
    pub const fn is_write(self) -> bool {
        matches!(self, ResourceState::ColorAttachment | ResourceState::DepthWrite | ResourceState::ShaderWrite | ResourceState::TransferDst)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureBarrier {
    pub texture: TextureHandle,
    pub from: ResourceState,
    pub to: ResourceState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBarrier {
    pub buffer: BufferHandle,
    pub from: ResourceState,
    pub to: ResourceState,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorTarget {
    pub texture: TextureHandle,
    pub clear: bool,
    pub clear_color: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct DepthTarget {
    pub texture: TextureHandle,
    pub clear: bool,
    pub clear_depth: f32,
    /// Depth is tested but not written, and may be sampled in the same pass.
    pub read_only: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RenderingInfo {
    pub colors: Vec<ColorTarget>,
    pub depth: Option<DepthTarget>,
    pub width: u32,
    pub height: u32,
}

/// Records one frame's commands. Methods do not fail: a misuse (e.g. mesh tasks without the feature) is
/// remembered and reported by [`crate::Device::submit`].
pub trait CommandList: Send {
    fn barrier(&mut self, textures: &[TextureBarrier], buffers: &[BufferBarrier]);
    /// Sets viewport and scissor to `width` x `height`.
    fn begin_rendering(&mut self, info: &RenderingInfo);
    fn end_rendering(&mut self);
    fn bind_pipeline(&mut self, pipeline: PipelineHandle);
    fn bind_vertex_buffer(&mut self, buffer: BufferHandle, offset: u64);
    /// `u32` indices only.
    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64);
    fn bind_uniform_buffer(&mut self, set: u32, binding: u32, buffer: BufferHandle);
    fn bind_storage_buffer(&mut self, set: u32, binding: u32, buffer: BufferHandle);
    fn bind_sampled_texture(&mut self, set: u32, binding: u32, texture: TextureHandle);
    /// Binds the device's bindless texture table as descriptor set `set`.
    fn bind_bindless_table(&mut self, set: u32);
    fn push_constants(&mut self, stages: ShaderStages, offset: u32, data: &[u8]);
    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32);
    fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32);
    /// `draw_count` commands of `VkDrawIndexedIndirectCommand` layout, `stride` bytes apart. Needs
    /// multi-draw-indirect when `draw_count > 1`.
    fn draw_indexed_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32);
    /// As above, with the count read from `count_buffer` (at most `max_draw_count`). Needs
    /// [`crate::Feature::DrawIndirectCount`].
    fn draw_indexed_indirect_count(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        count_buffer: BufferHandle,
        count_offset: u64,
        max_draw_count: u32,
        stride: u32,
    );
    /// Needs [`crate::Feature::MeshShader`].
    fn draw_mesh_tasks(&mut self, x: u32, y: u32, z: u32);
    /// Needs [`crate::Feature::MeshShader`].
    fn draw_mesh_tasks_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32);
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64);
    fn copy_buffer(&mut self, src: BufferHandle, src_offset: u64, dst: BufferHandle, dst_offset: u64, size: u64);
    /// Fills `size` bytes (a multiple of 4) at `offset` with `value`.
    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32);
    /// Copies tightly packed texels into `mip` of `dst`.
    fn copy_buffer_to_texture(&mut self, src: BufferHandle, src_offset: u64, dst: TextureHandle, mip: u32);
    /// Tightly packed, mip 0.
    fn copy_texture_to_buffer(&mut self, src: TextureHandle, dst: BufferHandle);
    /// Records a GPU timestamp under `name`; read back with [`crate::Device::read_timestamps`].
    fn write_timestamp(&mut self, name: &str);
    fn begin_debug_label(&mut self, name: &str);
    fn end_debug_label(&mut self);

    /// For a backend to get its concrete list back in `submit`.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}
