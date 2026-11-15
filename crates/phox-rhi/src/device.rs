use crate::capabilities::Capabilities;
use crate::commands::CommandList;
use crate::descs::*;
use crate::error::{Result, RhiError};
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

/// A window's next image to render to (see [`Device::acquire_backbuffer`]).
#[derive(Clone, Copy, Debug)]
pub struct Backbuffer {
    /// The swapchain image, usable in a graph like any texture (import it from `Undefined`, end it in `Present`).
    pub texture: TextureHandle,
    pub format: crate::Format,
    pub width: u32,
    pub height: u32,
    /// Which swapchain image, for [`Device::submit_for_present`].
    pub index: u32,
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
    /// The device address of a buffer created with `device_address: true`, for shaders that read it through a
    /// pointer.
    fn buffer_address(&self, buffer: BufferHandle) -> Result<u64>;

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

    /// The next image of the window's swapchain, which is recreated when `width` x `height` (the window's
    /// size) differ from its own or it is out of date. `None`: nothing to draw into now (a minimized
    /// window); skip the frame. Fails with `Unsupported` for devices without a window.
    fn acquire_backbuffer(&self, _width: u32, _height: u32) -> Result<Option<Backbuffer>> {
        Err(RhiError::Unsupported("presenting to a window".into()))
    }

    /// Submits `commands`, which render to `backbuffer` and leave it in the `Present` state, after the image
    /// is available, and presents it when they are done.
    fn submit_for_present(&self, _commands: Box<dyn CommandList>, _backbuffer: &Backbuffer) -> Result<SubmitTicket> {
        Err(RhiError::Unsupported("presenting to a window".into()))
    }
}
