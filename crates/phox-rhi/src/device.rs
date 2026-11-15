use crate::capabilities::Capabilities;
use crate::commands::CommandList;
use crate::descs::*;
use crate::error::Result;
use crate::handles::*;

/// A timeline semaphore value: the submission is complete once the device's timeline reaches it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubmitTicket {
    pub value: u64,
}

/// One GPU timestamp written with [`CommandList::write_timestamp`].
#[derive(Clone, Debug, PartialEq)]
pub struct Timestamp {
    pub name: String,
    pub nanoseconds: f64,
}

/// A GPU device. Every method takes `&self`: backends keep their state behind locks, so passes can use the
/// device while a command list is being recorded.
pub trait Device: Send + Sync {
    fn capabilities(&self) -> Capabilities;

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferHandle>;
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureHandle>;
    fn create_shader(&self, desc: &ShaderDesc) -> Result<ShaderHandle>;
    fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle>;
    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle>;
    /// Destruction is deferred until the GPU is done with the resource. An invalid or stale handle is a no-op.
    fn destroy_buffer(&self, handle: BufferHandle);
    fn destroy_texture(&self, handle: TextureHandle);
    fn destroy_shader(&self, handle: ShaderHandle);
    fn destroy_pipeline(&self, handle: PipelineHandle);

    /// Writes into an `Upload` (or `Readback`) buffer's persistent mapping.
    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;
    /// Reads from an `Upload` or `Readback` buffer's persistent mapping (after the GPU has finished with it).
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, out: &mut [u8]) -> Result<()>;
    fn describe_texture(&self, handle: TextureHandle) -> Result<TextureDesc>;
    fn describe_buffer(&self, handle: BufferHandle) -> Result<BufferDesc>;

    /// Puts `texture` (in the shader-read state) in the bindless table with `sampler`. Needs
    /// [`crate::Feature::BindlessDescriptors`].
    fn register_bindless_texture(&self, texture: TextureHandle, sampler: &SamplerDesc) -> Result<BindlessIndex>;
    fn unregister_bindless_texture(&self, index: BindlessIndex);

    /// One list per call; submit it with [`Device::submit`].
    fn begin_commands(&self) -> Result<Box<dyn CommandList>>;
    /// Submits the list and signals the device's timeline semaphore. Reports misuse recorded in the list.
    fn submit(&self, commands: Box<dyn CommandList>) -> Result<SubmitTicket>;
    fn is_complete(&self, ticket: SubmitTicket) -> bool;
    fn wait(&self, ticket: SubmitTicket) -> Result<()>;
    fn wait_idle(&self) -> Result<()>;
    /// The timestamps of a completed submission, in recording order. Empty without
    /// [`crate::Feature::Timestamps`].
    fn read_timestamps(&self, ticket: SubmitTicket) -> Result<Vec<Timestamp>>;
}
