//! The Metal device: resources in generation-checked pools, shaders translated from SPIR-V, pipelines, a
//! timeline on an `MTLSharedEvent`, and an optional `CAMetalLayer` to present to.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::MainThreadMarker;
use objc2_app_kit::NSView;
use objc2_core_foundation::{CGRect, CGSize};
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use phox_rhi::pool::Pool;
use phox_rhi::*;
use raw_window_handle::RawWindowHandle;

use crate::commands::MetalCommandList;
use crate::convert;
use crate::shader::{self, slot};
use crate::MetalConfig;

pub(crate) type Device = ProtocolObject<dyn MTLDevice>;
pub(crate) type Buffer = ProtocolObject<dyn MTLBuffer>;
pub(crate) type Texture = ProtocolObject<dyn MTLTexture>;
pub(crate) type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
pub(crate) type SamplerState = ProtocolObject<dyn MTLSamplerState>;
type Function = ProtocolObject<dyn MTLFunction>;

/// The largest push constant block: `set*Bytes` takes at most 4 KB.
const MAX_PUSH_CONSTANT_BYTES: u32 = 4096;
/// The swapchain's format: sRGB-encoding BGRA8, as Vulkan prefers.
const LAYER_FORMAT: Format = Format::BGRA8Srgb;

pub(crate) fn backend(e: impl std::fmt::Display) -> RhiError {
    RhiError::Backend(e.to_string())
}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

pub(crate) struct BufferRes {
    pub buffer: Retained<Buffer>,
    pub desc: BufferDesc,
}

pub(crate) struct TextureRes {
    pub texture: Retained<Texture>,
    pub desc: TextureDesc,
}

pub(crate) struct ShaderRes {
    pub function: Retained<Function>,
    pub stage: ShaderStage,
    pub workgroup: [u32; 3],
}

pub(crate) enum PipelineRes {
    Graphics {
        state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        depth: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
        cull: MTLCullMode,
        primitive: MTLPrimitiveType,
        samplers: Vec<(u64, Retained<SamplerState>)>,
    },
    Compute {
        state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        workgroup: MTLSize,
        samplers: Vec<(u64, Retained<SamplerState>)>,
    },
}

/// A drawable handed out by `acquire_backbuffer` and not yet presented.
struct PendingDrawable {
    index: u32,
    drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
    texture: TextureHandle,
}

pub(crate) struct State {
    pub buffers: Pool<BufferTag, BufferRes>,
    pub textures: Pool<TextureTag, TextureRes>,
    pub shaders: Pool<ShaderTag, ShaderRes>,
    pub pipelines: Pool<PipelineTag, PipelineRes>,
    /// The timeline value the next submission signals.
    next_value: u64,
    /// Submitted command buffers, oldest first, until they are known to have finished.
    in_flight: VecDeque<(u64, Retained<CommandBuffer>)>,
    /// Failures of finished command buffers, reported by the next `wait`.
    gpu_errors: Vec<String>,
    drawables: Vec<PendingDrawable>,
    next_drawable: u32,
}

pub(crate) struct Inner {
    pub device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    layer: Option<Retained<CAMetalLayer>>,
    validation: bool,
    capabilities: Capabilities,
    state: Mutex<State>,
}

// SAFETY: Metal devices, queues, events, resources and pipeline states are thread-safe. Command buffers are
// only touched under the state lock, or by the one command list that owns them. The layer's drawable calls
// are thread-safe.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A Metal 3 device on an Apple Silicon GPU. See [`MetalDevice::new`].
#[derive(Clone)]
pub struct MetalDevice {
    pub(crate) inner: Arc<Inner>,
}

impl MetalDevice {
    /// The system's default GPU, for offscreen work. Fails on GPUs without Metal 3 and the Apple 7 family
    /// (Intel Macs).
    pub fn new(config: &MetalConfig) -> Result<Self> {
        Self::create(config, None)
    }

    /// As [`MetalDevice::new`], presenting to a `CAMetalLayer` added to `window`'s `NSView` (an AppKit
    /// handle), so [`phox_rhi::Device::acquire_backbuffer`] works. Must be called on the main thread.
    ///
    /// # Safety
    /// The view behind the handle must stay alive for as long as the device does.
    pub unsafe fn new_for_view(config: &MetalConfig, window: RawWindowHandle) -> Result<Self> {
        let RawWindowHandle::AppKit(handle) = window else {
            return Err(RhiError::Unsupported(format!("presenting to a {window:?} with Metal (it needs an AppKit NSView)")));
        };
        let Some(mtm) = MainThreadMarker::new() else {
            return Err(RhiError::Misuse("a Metal device for a view must be created on the main thread".into()));
        };
        // SAFETY: the caller guarantees the handle is a live NSView.
        let view: &NSView = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
        Self::create(config, Some((view, mtm)))
    }

    fn create(config: &MetalConfig, view: Option<(&NSView, MainThreadMarker)>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| RhiError::Unsupported("a Metal device".into()))?;
        if !device.supportsFamily(MTLGPUFamily::Metal3) || !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(RhiError::Unsupported(format!(
                "{}: Phox's Metal backend needs an Apple Silicon GPU with Metal 3",
                device.name()
            )));
        }
        let queue = device.newCommandQueue().ok_or_else(|| backend("cannot create a Metal command queue"))?;
        let event = device.newSharedEvent().ok_or_else(|| backend("cannot create a Metal shared event"))?;
        let layer = view.map(|(view, _mtm)| attach_layer(&device, view));
        let capabilities = Capabilities {
            device_name: device.name().to_string(),
            api_major: 3,
            api_minor: 0,
            features: FeatureSet::default(),
            max_texture_size_2d: 16384,
            max_push_constant_bytes: MAX_PUSH_CONSTANT_BYTES,
            max_draw_indirect_count: 1,
            timestamp_period_ns: 1.0,
            max_task_workgroup_count_x: 0,
            bindless_texture_capacity: 0,
        };
        let state = State {
            buffers: Pool::new("buffer"),
            textures: Pool::new("texture"),
            shaders: Pool::new("shader"),
            pipelines: Pool::new("pipeline"),
            next_value: 1,
            in_flight: VecDeque::new(),
            gpu_errors: Vec::new(),
            drawables: Vec::new(),
            next_drawable: 0,
        };
        log::info!("Metal device: {}", capabilities.device_name);
        let inner = Inner { device, queue, event, layer, validation: config.validation, capabilities, state: Mutex::new(state) };
        Ok(Self { inner: Arc::new(inner) })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.inner.capabilities.device_name
    }

    /// Takes finished command buffers off the in-flight list, keeping their failures for `wait` to report.
    fn reap(state: &mut State) {
        while let Some((_, cb)) = state.in_flight.front() {
            match cb.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    let message = cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_else(|| "unknown error".into());
                    state.gpu_errors.push(message);
                }
                _ => break,
            }
            state.in_flight.pop_front();
        }
    }

    fn submit_inner(&self, commands: Box<dyn CommandList>, present: Option<&ProtocolObject<dyn CAMetalDrawable>>) -> Result<SubmitTicket> {
        let list = commands.into_any().downcast::<MetalCommandList>().map_err(|_| RhiError::Misuse("a command list from another device".into()))?;
        let cb = list.finish()?;
        if let Some(drawable) = present {
            cb.presentDrawable(ProtocolObject::from_ref(drawable));
        }
        let mut st = self.inner.lock();
        let value = st.next_value;
        st.next_value += 1;
        cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*self.inner.event), value);
        cb.commit();
        st.in_flight.push_back((value, cb));
        Self::reap(&mut st);
        Ok(SubmitTicket { value })
    }

    fn sampler(&self, desc: &SamplerDesc) -> Result<Retained<SamplerState>> {
        let d = MTLSamplerDescriptor::new();
        d.setMinFilter(convert::filter(desc.min_filter));
        d.setMagFilter(convert::filter(desc.mag_filter));
        d.setMipFilter(convert::mip_filter(desc.mip_filter));
        d.setSAddressMode(convert::wrap(desc.wrap_u));
        d.setTAddressMode(convert::wrap(desc.wrap_v));
        self.inner.device.newSamplerStateWithDescriptor(&d).ok_or_else(|| backend("cannot create a Metal sampler"))
    }

    fn samplers(&self, bindings: &[Binding]) -> Result<Vec<(u64, Retained<SamplerState>)>> {
        let mut out = Vec::new();
        for b in bindings {
            match b.ty {
                BindingType::SampledTexture => {
                    let index = slot(b.set, b.binding).ok_or_else(|| RhiError::Unsupported(format!("binding (set {}, binding {}) on Metal", b.set, b.binding)))?;
                    out.push((index, self.sampler(&b.sampler)?));
                }
                BindingType::BindlessTextures => return Err(RhiError::Unsupported(Feature::BindlessDescriptors.to_string())),
                _ => {}
            }
        }
        Ok(out)
    }

    fn function(&self, handle: ShaderHandle) -> Result<(Retained<Function>, ShaderStage, [u32; 3])> {
        let st = self.inner.lock();
        let s = st.shaders.get(handle)?;
        Ok((s.function.clone(), s.stage, s.workgroup))
    }
}

/// Adds a layer that fills `view` and follows its size, at the screen's scale.
fn attach_layer(device: &Device, view: &NSView) -> Retained<CAMetalLayer> {
    let layer = CAMetalLayer::new();
    layer.setDevice(Some(device));
    layer.setPixelFormat(convert::pixel_format(LAYER_FORMAT).expect("a colour format"));
    view.setWantsLayer(true);
    let bounds: CGRect = view.bounds();
    layer.setFrame(bounds);
    layer.setAutoresizingMask(CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable);
    if let Some(window) = view.window() {
        layer.setContentsScale(window.backingScaleFactor());
    }
    if let Some(host) = view.layer() {
        host.addSublayer(&layer);
    }
    layer
}

impl phox_rhi::Device for MetalDevice {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities.clone()
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferHandle> {
        if desc.size == 0 || desc.usage.is_empty() {
            return Err(RhiError::InvalidDesc(format!("buffer `{}` needs a size and a usage", desc.debug_name)));
        }
        let buffer = self
            .inner
            .device
            .newBufferWithLength_options(desc.size as usize, convert::resource_options(desc.memory))
            .ok_or_else(|| backend(format!("cannot allocate {} bytes for buffer `{}`", desc.size, desc.debug_name)))?;
        if !desc.debug_name.is_empty() {
            buffer.setLabel(Some(&ns(&desc.debug_name)));
        }
        Ok(self.inner.lock().buffers.create(BufferRes { buffer, desc: desc.clone() }))
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureHandle> {
        let format = convert::pixel_format(desc.format).ok_or_else(|| RhiError::InvalidDesc(format!("texture `{}` has no format", desc.debug_name)))?;
        if desc.width == 0 || desc.height == 0 || desc.mips == 0 {
            return Err(RhiError::InvalidDesc(format!("texture `{}` is empty", desc.debug_name)));
        }
        // SAFETY: plain descriptor construction with in-range values.
        let d = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, desc.width as usize, desc.height as usize, desc.mips > 1)
        };
        // SAFETY: the mip count fits the size (checked by Metal, which reports an error instead).
        unsafe { d.setMipmapLevelCount(desc.mips as usize) };
        d.setUsage(convert::texture_usage(desc.usage) | MTLTextureUsage::RenderTarget);
        d.setStorageMode(MTLStorageMode::Private);
        let texture = self
            .inner
            .device
            .newTextureWithDescriptor(&d)
            .ok_or_else(|| backend(format!("cannot create texture `{}`", desc.debug_name)))?;
        if !desc.debug_name.is_empty() {
            texture.setLabel(Some(&ns(&desc.debug_name)));
        }
        Ok(self.inner.lock().textures.create(TextureRes { texture, desc: desc.clone() }))
    }

    fn create_shader(&self, desc: &ShaderDesc) -> Result<ShaderHandle> {
        let translated = shader::translate(&desc.spirv, desc.stage, &desc.entry)?;
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_0);
        let library = self
            .inner
            .device
            .newLibraryWithSource_options_error(&ns(&translated.source), Some(&options))
            .map_err(|e| backend(format!("the Metal compiler rejected the translated {:?} shader: {}\n{}", desc.stage, e.localizedDescription(), translated.source)))?;
        let function = library
            .newFunctionWithName(&ns(&translated.entry))
            .ok_or_else(|| backend(format!("the translated shader has no function `{}`", translated.entry)))?;
        Ok(self.inner.lock().shaders.create(ShaderRes { function, stage: desc.stage, workgroup: translated.workgroup }))
    }

    fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle> {
        let mut vertex = None;
        let mut fragment = None;
        for &handle in &desc.shaders {
            let (function, stage, _) = self.function(handle)?;
            match stage {
                ShaderStage::Vertex => vertex = Some(function),
                ShaderStage::Fragment => fragment = Some(function),
                ShaderStage::Task | ShaderStage::Mesh => return Err(RhiError::Unsupported(Feature::MeshShader.to_string())),
                ShaderStage::Compute => return Err(RhiError::InvalidDesc(format!("pipeline `{}` has a compute shader", desc.debug_name))),
            }
        }
        let vertex = vertex.ok_or_else(|| RhiError::InvalidDesc(format!("pipeline `{}` has no vertex shader", desc.debug_name)))?;

        let d = MTLRenderPipelineDescriptor::new();
        d.setVertexFunction(Some(&vertex));
        d.setFragmentFunction(fragment.as_deref());
        if !desc.debug_name.is_empty() {
            d.setLabel(Some(&ns(&desc.debug_name)));
        }
        if !desc.vertex_layout.attributes.is_empty() {
            let v = MTLVertexDescriptor::new();
            for a in &desc.vertex_layout.attributes {
                // SAFETY: attribute and layout indices are within Metal's 31 slots.
                let attribute = unsafe { v.attributes().objectAtIndexedSubscript(a.location as usize) };
                attribute.setFormat(convert::vertex_format(a.format));
                // SAFETY: offsets come from the layout, which the vertex data matches.
                unsafe {
                    attribute.setOffset(a.offset as usize);
                    attribute.setBufferIndex(shader::VERTEX_BUFFER_SLOT as usize);
                }
            }
            // SAFETY: as above.
            let layout = unsafe { v.layouts().objectAtIndexedSubscript(shader::VERTEX_BUFFER_SLOT as usize) };
            // SAFETY: the stride is the vertex size.
            unsafe { layout.setStride(desc.vertex_layout.stride as usize) };
            layout.setStepFunction(MTLVertexStepFunction::PerVertex);
            d.setVertexDescriptor(Some(&v));
        }
        let blend = desc.blend.is_some_and(|b| b.enabled);
        for (i, &format) in desc.color_formats.iter().enumerate() {
            // SAFETY: at most 8 colour attachments, as on Vulkan.
            let attachment = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
            attachment.setPixelFormat(convert::pixel_format(format).ok_or_else(|| RhiError::InvalidDesc("an undefined colour format".into()))?);
            if blend {
                attachment.setBlendingEnabled(true);
                attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                attachment.setRgbBlendOperation(MTLBlendOperation::Add);
                attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
            }
        }
        if let Some(depth) = convert::pixel_format(desc.depth_format) {
            d.setDepthAttachmentPixelFormat(depth);
        }
        let (primitive, class) = convert::primitive(desc.topology);
        // SAFETY: the class matches the primitive type every draw with this pipeline uses.
        unsafe { d.setInputPrimitiveTopology(class) };
        let state = self
            .inner
            .device
            .newRenderPipelineStateWithDescriptor_error(&d)
            .map_err(|e| backend(format!("pipeline `{}`: {}", desc.debug_name, e.localizedDescription())))?;
        let depth = if desc.depth_format.is_depth() {
            let ds = MTLDepthStencilDescriptor::new();
            ds.setDepthCompareFunction(if desc.depth_test { convert::compare(desc.depth_compare) } else { MTLCompareFunction::Always });
            ds.setDepthWriteEnabled(desc.depth_write);
            Some(self.inner.device.newDepthStencilStateWithDescriptor(&ds).ok_or_else(|| backend("cannot create a depth state"))?)
        } else {
            None
        };
        let samplers = self.samplers(&desc.bindings)?;
        let res = PipelineRes::Graphics { state, depth, cull: convert::cull(desc.cull), primitive, samplers };
        Ok(self.inner.lock().pipelines.create(res))
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle> {
        let (function, stage, [x, y, z]) = self.function(desc.shader)?;
        if stage != ShaderStage::Compute {
            return Err(RhiError::InvalidDesc(format!("pipeline `{}` needs a compute shader", desc.debug_name)));
        }
        let state = self
            .inner
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| backend(format!("pipeline `{}`: {}", desc.debug_name, e.localizedDescription())))?;
        let samplers = self.samplers(&desc.bindings)?;
        let workgroup = MTLSize { width: x as usize, height: y as usize, depth: z as usize };
        Ok(self.inner.lock().pipelines.create(PipelineRes::Compute { state, workgroup, samplers }))
    }

    // Command buffers retain what they use, so a resource can be released as soon as it is destroyed.
    fn destroy_buffer(&self, handle: BufferHandle) {
        self.inner.lock().buffers.destroy(handle);
    }

    fn destroy_texture(&self, handle: TextureHandle) {
        self.inner.lock().textures.destroy(handle);
    }

    fn destroy_shader(&self, handle: ShaderHandle) {
        self.inner.lock().shaders.destroy(handle);
    }

    fn destroy_pipeline(&self, handle: PipelineHandle) {
        self.inner.lock().pipelines.destroy(handle);
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let st = self.inner.lock();
        let res = st.buffers.get(buffer)?;
        let ptr = mapped(res, offset, data.len())?;
        // SAFETY: the range is inside the buffer's shared storage (checked by `mapped`).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len()) };
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, out: &mut [u8]) -> Result<()> {
        let st = self.inner.lock();
        let res = st.buffers.get(buffer)?;
        let ptr = mapped(res, offset, out.len())?;
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(ptr.as_ptr(), out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    fn describe_texture(&self, handle: TextureHandle) -> Result<TextureDesc> {
        Ok(self.inner.lock().textures.get(handle)?.desc.clone())
    }

    fn describe_buffer(&self, handle: BufferHandle) -> Result<BufferDesc> {
        Ok(self.inner.lock().buffers.get(handle)?.desc.clone())
    }

    fn buffer_address(&self, buffer: BufferHandle) -> Result<u64> {
        Ok(self.inner.lock().buffers.get(buffer)?.buffer.gpuAddress())
    }

    fn register_bindless_texture(&self, _texture: TextureHandle, _sampler: &SamplerDesc) -> Result<BindlessIndex> {
        Err(RhiError::Unsupported(Feature::BindlessDescriptors.to_string()))
    }

    fn unregister_bindless_texture(&self, _index: BindlessIndex) {}

    fn begin_commands(&self) -> Result<Box<dyn CommandList>> {
        let cb = if self.inner.validation {
            let d = MTLCommandBufferDescriptor::new();
            d.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
            self.inner.queue.commandBufferWithDescriptor(&d)
        } else {
            self.inner.queue.commandBuffer()
        };
        let cb = cb.ok_or_else(|| backend("cannot create a Metal command buffer"))?;
        Ok(Box::new(MetalCommandList::new(self.inner.clone(), cb)))
    }

    fn submit(&self, commands: Box<dyn CommandList>) -> Result<SubmitTicket> {
        self.submit_inner(commands, None)
    }

    fn is_complete(&self, ticket: SubmitTicket) -> bool {
        self.inner.event.signaledValue() >= ticket.value
    }

    fn wait(&self, ticket: SubmitTicket) -> Result<()> {
        let waiting: Vec<Retained<CommandBuffer>> =
            self.inner.lock().in_flight.iter().filter(|(v, _)| *v <= ticket.value).map(|(_, cb)| cb.clone()).collect();
        for cb in &waiting {
            cb.waitUntilCompleted();
        }
        let mut st = self.inner.lock();
        Self::reap(&mut st);
        if st.gpu_errors.is_empty() {
            Ok(())
        } else {
            Err(backend(format!("the GPU reported: {}", std::mem::take(&mut st.gpu_errors).join("; "))))
        }
    }

    fn wait_idle(&self) -> Result<()> {
        let last = self.inner.lock().next_value - 1;
        self.wait(SubmitTicket { value: last })
    }

    fn read_timestamps(&self, _ticket: SubmitTicket) -> Result<Vec<Timestamp>> {
        Ok(Vec::new())
    }

    fn acquire_backbuffer(&self, width: u32, height: u32) -> Result<Option<Backbuffer>> {
        let Some(layer) = &self.inner.layer else {
            return Err(RhiError::Unsupported("presenting from a Metal device made without a view".into()));
        };
        if width == 0 || height == 0 {
            return Ok(None);
        }
        let size = CGSize { width: f64::from(width), height: f64::from(height) };
        if layer.drawableSize() != size {
            layer.setDrawableSize(size);
        }
        let Some(drawable) = layer.nextDrawable() else { return Ok(None) };
        let texture = drawable.texture();
        let desc = TextureDesc::new(texture.width() as u32, texture.height() as u32, LAYER_FORMAT, TextureUsage::COLOR_ATTACHMENT, "backbuffer");
        let mut st = self.inner.lock();
        let handle = st.textures.create(TextureRes { texture, desc: desc.clone() });
        let index = st.next_drawable;
        st.next_drawable = st.next_drawable.wrapping_add(1);
        st.drawables.push(PendingDrawable { index, drawable, texture: handle });
        Ok(Some(Backbuffer { texture: handle, format: desc.format, width: desc.width, height: desc.height, index }))
    }

    fn submit_for_present(&self, commands: Box<dyn CommandList>, backbuffer: &Backbuffer) -> Result<SubmitTicket> {
        let pending = {
            let mut st = self.inner.lock();
            let i = st
                .drawables
                .iter()
                .position(|d| d.index == backbuffer.index)
                .ok_or_else(|| RhiError::Misuse("presenting a backbuffer that was not acquired, or was already presented".into()))?;
            st.drawables.swap_remove(i)
        };
        let result = self.submit_inner(commands, Some(&pending.drawable));
        self.inner.lock().textures.destroy(pending.texture);
        result
    }
}

/// The CPU address of `len` bytes at `offset` in a shared buffer.
fn mapped(res: &BufferRes, offset: u64, len: usize) -> Result<NonNull<u8>> {
    if res.desc.memory == MemoryKind::GpuOnly {
        return Err(RhiError::Misuse(format!("buffer `{}` is GPU-only and cannot be mapped", res.desc.debug_name)));
    }
    let end = offset.checked_add(len as u64).filter(|&end| end <= res.desc.size);
    if end.is_none() {
        return Err(RhiError::Misuse(format!("{len} bytes at {offset} overrun buffer `{}` ({} bytes)", res.desc.debug_name, res.desc.size)));
    }
    let base: NonNull<c_void> = res.buffer.contents();
    // SAFETY: in bounds, checked above.
    Ok(unsafe { base.cast::<u8>().add(offset as usize) })
}

impl Drop for Inner {
    fn drop(&mut self) {
        let waiting: Vec<_> = self.lock().in_flight.drain(..).map(|(_, cb)| cb).collect();
        for cb in waiting {
            cb.waitUntilCompleted();
        }
        if let Some(layer) = &self.layer {
            layer.removeFromSuperlayer();
        }
    }
}
