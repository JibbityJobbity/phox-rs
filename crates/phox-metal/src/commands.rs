//! The Metal command list. Metal binds resources to encoders, and a list switches encoders as it goes (render
//! inside `begin_rendering`, compute for dispatches, blit for copies), so bindings and push constants are
//! shadowed here and applied to whichever encoder draws or dispatches next.
//!
//! Resources are created with Metal's automatic hazard tracking, so `barrier` has nothing to do in this stage.

use std::any::Any;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use phox_rhi::*;

use crate::device::{Buffer, CommandBuffer, Inner, PipelineRes, SamplerState, Texture};
use crate::shader::{slot, PUSH_CONSTANT_SLOT, VERTEX_BUFFER_SLOT};

type RenderEncoder = ProtocolObject<dyn MTLRenderCommandEncoder>;
type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
type BlitEncoder = ProtocolObject<dyn MTLBlitCommandEncoder>;

enum Encoder {
    None,
    Render(Retained<RenderEncoder>),
    Compute(Retained<ComputeEncoder>),
    Blit(Retained<BlitEncoder>),
}

#[derive(Clone)]
enum Bound {
    Buffer(Retained<Buffer>),
    Texture(Retained<Texture>),
}

#[derive(Clone)]
enum Pipeline {
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

pub(crate) struct MetalCommandList {
    inner: Arc<Inner>,
    cb: Retained<CommandBuffer>,
    encoder: Encoder,
    /// The current render target's size, for clamping scissors.
    target: (u32, u32),
    pipeline: Option<Pipeline>,
    bound: BTreeMap<u64, Bound>,
    vertex_buffer: Option<(Retained<Buffer>, u64)>,
    index_buffer: Option<(Retained<Buffer>, u64)>,
    push: Vec<u8>,
    /// Bindings changed, or the encoder did, since they were last applied.
    dirty: bool,
    errors: Vec<String>,
}

// SAFETY: the list is used by one thread at a time (it is moved, never shared), which is all Metal asks of
// command buffers and encoders.
unsafe impl Send for MetalCommandList {}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

/// `data` as the non-null pointer `set*Bytes` takes.
fn bytes_ptr(data: &[u8]) -> NonNull<c_void> {
    NonNull::new(data.as_ptr() as *mut c_void).expect("a slice pointer is never null")
}

impl MetalCommandList {
    pub(crate) fn new(inner: Arc<Inner>, cb: Retained<CommandBuffer>) -> Self {
        Self {
            inner,
            cb,
            encoder: Encoder::None,
            target: (0, 0),
            pipeline: None,
            bound: BTreeMap::new(),
            vertex_buffer: None,
            index_buffer: None,
            push: Vec::new(),
            dirty: true,
            errors: Vec::new(),
        }
    }

    fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
    }

    /// Ends the current encoder and returns the command buffer, or the misuse recorded while recording.
    pub(crate) fn finish(mut self: Box<Self>) -> Result<Retained<CommandBuffer>> {
        self.end_encoder();
        if !self.errors.is_empty() {
            return Err(RhiError::Misuse(self.errors.join("; ")));
        }
        Ok(self.cb.clone())
    }

    fn end_encoder(&mut self) {
        match std::mem::replace(&mut self.encoder, Encoder::None) {
            Encoder::None => {}
            Encoder::Render(e) => e.endEncoding(),
            Encoder::Compute(e) => e.endEncoding(),
            Encoder::Blit(e) => e.endEncoding(),
        }
        self.dirty = true;
    }

    fn buffer(&mut self, handle: BufferHandle) -> Option<Retained<Buffer>> {
        let found = self.inner.lock().buffers.get(handle).map(|b| b.buffer.clone());
        found.map_err(|_| self.error("a destroyed or invalid buffer")).ok()
    }

    fn texture(&mut self, handle: TextureHandle) -> Option<(Retained<Texture>, TextureDesc)> {
        let found = self.inner.lock().textures.get(handle).map(|t| (t.texture.clone(), t.desc.clone()));
        found.map_err(|_| self.error("a destroyed or invalid texture")).ok()
    }

    fn render_encoder(&mut self, call: &str) -> Option<Retained<RenderEncoder>> {
        match &self.encoder {
            Encoder::Render(e) => Some(e.clone()),
            _ => {
                self.error(format!("{call} outside begin_rendering"));
                None
            }
        }
    }

    fn compute_encoder(&mut self, call: &str) -> Option<Retained<ComputeEncoder>> {
        match &self.encoder {
            Encoder::Compute(e) => return Some(e.clone()),
            Encoder::Render(_) => {
                self.error(format!("{call} inside begin_rendering"));
                return None;
            }
            _ => {}
        }
        self.end_encoder();
        let encoder = self.cb.computeCommandEncoder()?;
        self.encoder = Encoder::Compute(encoder.clone());
        Some(encoder)
    }

    fn blit_encoder(&mut self, call: &str) -> Option<Retained<BlitEncoder>> {
        match &self.encoder {
            Encoder::Blit(e) => return Some(e.clone()),
            Encoder::Render(_) => {
                self.error(format!("{call} inside begin_rendering"));
                return None;
            }
            _ => {}
        }
        self.end_encoder();
        let encoder = self.cb.blitCommandEncoder()?;
        self.encoder = Encoder::Blit(encoder.clone());
        Some(encoder)
    }

    /// Applies the pipeline, bindings and push constants to a render encoder, if they changed.
    fn apply_render(&mut self, e: &RenderEncoder) -> Option<MTLPrimitiveType> {
        let Some(Pipeline::Graphics { state, depth, cull, primitive, samplers }) = self.pipeline.clone() else {
            self.error("a draw without a graphics pipeline");
            return None;
        };
        if !self.dirty {
            return Some(primitive);
        }
        e.setRenderPipelineState(&state);
        if let Some(depth) = &depth {
            e.setDepthStencilState(Some(depth));
        }
        e.setCullMode(cull);
        e.setFrontFacingWinding(MTLWinding::CounterClockwise);
        // SAFETY: every index is below Metal's 31 buffer and texture slots, and the objects are live (retained).
        unsafe {
            for (index, bound) in &self.bound {
                match bound {
                    Bound::Buffer(b) => {
                        e.setVertexBuffer_offset_atIndex(Some(b), 0, *index as usize);
                        e.setFragmentBuffer_offset_atIndex(Some(b), 0, *index as usize);
                    }
                    Bound::Texture(t) => {
                        e.setVertexTexture_atIndex(Some(t), *index as usize);
                        e.setFragmentTexture_atIndex(Some(t), *index as usize);
                    }
                }
            }
            for (index, sampler) in &samplers {
                e.setVertexSamplerState_atIndex(Some(sampler), *index as usize);
                e.setFragmentSamplerState_atIndex(Some(sampler), *index as usize);
            }
            if let Some((b, offset)) = &self.vertex_buffer {
                e.setVertexBuffer_offset_atIndex(Some(b), *offset as usize, VERTEX_BUFFER_SLOT as usize);
            }
            if !self.push.is_empty() {
                e.setVertexBytes_length_atIndex(bytes_ptr(&self.push), self.push.len(), PUSH_CONSTANT_SLOT as usize);
                e.setFragmentBytes_length_atIndex(bytes_ptr(&self.push), self.push.len(), PUSH_CONSTANT_SLOT as usize);
            }
        }
        self.dirty = false;
        Some(primitive)
    }

    /// As `apply_render`, for a compute encoder; returns the pipeline's workgroup size.
    fn apply_compute(&mut self, e: &ComputeEncoder) -> Option<MTLSize> {
        let Some(Pipeline::Compute { state, workgroup, samplers }) = self.pipeline.clone() else {
            self.error("a dispatch without a compute pipeline");
            return None;
        };
        if !self.dirty {
            return Some(workgroup);
        }
        e.setComputePipelineState(&state);
        // SAFETY: as in `apply_render`.
        unsafe {
            for (index, bound) in &self.bound {
                match bound {
                    Bound::Buffer(b) => e.setBuffer_offset_atIndex(Some(b), 0, *index as usize),
                    Bound::Texture(t) => e.setTexture_atIndex(Some(t), *index as usize),
                }
            }
            for (index, sampler) in &samplers {
                e.setSamplerState_atIndex(Some(sampler), *index as usize);
            }
            if !self.push.is_empty() {
                e.setBytes_length_atIndex(bytes_ptr(&self.push), self.push.len(), PUSH_CONSTANT_SLOT as usize);
            }
        }
        self.dirty = false;
        Some(workgroup)
    }

    fn bind(&mut self, set: u32, binding: u32, bound: Option<Bound>) {
        let Some(index) = slot(set, binding) else {
            self.error(format!("binding (set {set}, binding {binding}) is outside the slots Metal maps"));
            return;
        };
        if let Some(bound) = bound {
            self.bound.insert(index, bound);
            self.dirty = true;
        }
    }

    fn unsupported(&mut self, call: &str, feature: Feature) {
        self.error(format!("{call} needs Feature::{feature}, which the Metal backend does not support yet"));
    }
}

impl CommandList for MetalCommandList {
    fn barrier(&mut self, _textures: &[TextureBarrier], _buffers: &[BufferBarrier]) {
        // Tracked resources: Metal orders the accesses itself.
    }

    fn begin_rendering(&mut self, info: &RenderingInfo) {
        if matches!(self.encoder, Encoder::Render(_)) {
            self.error("begin_rendering inside begin_rendering");
            return;
        }
        self.end_encoder();
        let pass = MTLRenderPassDescriptor::new();
        for (i, c) in info.colors.iter().enumerate() {
            let Some((texture, _)) = self.texture(c.texture) else { return };
            // SAFETY: at most 8 colour attachments, as on Vulkan.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(i) };
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(if c.clear { MTLLoadAction::Clear } else { MTLLoadAction::Load });
            attachment.setStoreAction(MTLStoreAction::Store);
            let [r, g, b, a] = c.clear_color.map(f64::from);
            attachment.setClearColor(MTLClearColor { red: r, green: g, blue: b, alpha: a });
        }
        if let Some(d) = &info.depth {
            let Some((texture, _)) = self.texture(d.texture) else { return };
            let attachment = pass.depthAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(if d.clear { MTLLoadAction::Clear } else { MTLLoadAction::Load });
            attachment.setStoreAction(if d.read_only { MTLStoreAction::DontCare } else { MTLStoreAction::Store });
            attachment.setClearDepth(f64::from(d.clear_depth));
        }
        let Some(encoder) = self.cb.renderCommandEncoderWithDescriptor(&pass) else {
            self.error("Metal could not start a render pass");
            return;
        };
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: f64::from(info.width),
            height: f64::from(info.height),
            znear: 0.0,
            zfar: 1.0,
        });
        self.target = (info.width, info.height);
        self.encoder = Encoder::Render(encoder);
        self.dirty = true;
    }

    fn end_rendering(&mut self) {
        match &self.encoder {
            Encoder::Render(_) => self.end_encoder(),
            _ => self.error("end_rendering without begin_rendering"),
        }
    }

    fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let Some(e) = self.render_encoder("set_scissor") else { return };
        // Metal requires the scissor inside the target; an empty one is allowed and draws nothing.
        let (tw, th) = self.target;
        let (x, y) = (x.min(tw), y.min(th));
        let rect = MTLScissorRect { x: x as usize, y: y as usize, width: width.min(tw - x) as usize, height: height.min(th - y) as usize };
        e.setScissorRect(rect);
    }

    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let found = self.inner.lock().pipelines.get(pipeline).map(|p| match p {
            PipelineRes::Graphics { state, depth, cull, primitive, samplers } => Pipeline::Graphics {
                state: state.clone(),
                depth: depth.clone(),
                cull: *cull,
                primitive: *primitive,
                samplers: samplers.clone(),
            },
            PipelineRes::Compute { state, workgroup, samplers } => Pipeline::Compute { state: state.clone(), workgroup: *workgroup, samplers: samplers.clone() },
        });
        match found {
            Ok(p) => {
                self.pipeline = Some(p);
                self.dirty = true;
            }
            Err(_) => self.error("bind_pipeline: a destroyed or invalid pipeline"),
        }
    }

    fn bind_vertex_buffer(&mut self, buffer: BufferHandle, offset: u64) {
        if let Some(b) = self.buffer(buffer) {
            self.vertex_buffer = Some((b, offset));
            self.dirty = true;
        }
    }

    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64) {
        if let Some(b) = self.buffer(buffer) {
            self.index_buffer = Some((b, offset));
        }
    }

    fn bind_uniform_buffer(&mut self, set: u32, binding: u32, buffer: BufferHandle) {
        let b = self.buffer(buffer).map(Bound::Buffer);
        self.bind(set, binding, b);
    }

    fn bind_storage_buffer(&mut self, set: u32, binding: u32, buffer: BufferHandle) {
        let b = self.buffer(buffer).map(Bound::Buffer);
        self.bind(set, binding, b);
    }

    fn bind_sampled_texture(&mut self, set: u32, binding: u32, texture: TextureHandle) {
        let t = self.texture(texture).map(|(t, _)| Bound::Texture(t));
        self.bind(set, binding, t);
    }

    fn bind_storage_texture(&mut self, set: u32, binding: u32, texture: TextureHandle, mip: u32) {
        let Some((t, desc)) = self.texture(texture) else { return };
        let view = if desc.mips > 1 {
            // SAFETY: one mip of the texture's own format, within its mip count (checked).
            if mip >= desc.mips {
                self.error(format!("bind_storage_texture: mip {mip} of a texture with {} mips", desc.mips));
                return;
            }
            let levels = objc2_foundation::NSRange { location: mip as usize, length: 1 };
            let slices = objc2_foundation::NSRange { location: 0, length: 1 };
            unsafe { t.newTextureViewWithPixelFormat_textureType_levels_slices(t.pixelFormat(), MTLTextureType::Type2D, levels, slices) }
        } else {
            Some(t)
        };
        match view {
            Some(v) => self.bind(set, binding, Some(Bound::Texture(v))),
            None => self.error("bind_storage_texture: Metal could not make a view of the mip"),
        }
    }

    fn bind_bindless_table(&mut self, _set: u32) {
        self.unsupported("bind_bindless_table", Feature::BindlessDescriptors);
    }

    fn push_constants(&mut self, _stages: ShaderStages, offset: u32, data: &[u8]) {
        let end = offset as usize + data.len();
        if end > 4096 {
            self.error(format!("push constants up to byte {end}; Metal takes at most 4096"));
            return;
        }
        if self.push.len() < end {
            self.push.resize(end, 0);
        }
        self.push[offset as usize..end].copy_from_slice(data);
        self.dirty = true;
    }

    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32) {
        let Some(e) = self.render_encoder("draw") else { return };
        let Some(primitive) = self.apply_render(&e) else { return };
        if vertex_count == 0 || instance_count == 0 {
            return;
        }
        // SAFETY: the bound vertex buffer holds the vertices (the RHI's contract, as on Vulkan).
        unsafe { e.drawPrimitives_vertexStart_vertexCount_instanceCount(primitive, first_vertex as usize, vertex_count as usize, instance_count as usize) };
    }

    fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        let Some(e) = self.render_encoder("draw_indexed") else { return };
        let Some(primitive) = self.apply_render(&e) else { return };
        let Some((indices, offset)) = self.index_buffer.clone() else {
            self.error("draw_indexed without an index buffer");
            return;
        };
        if index_count == 0 || instance_count == 0 {
            return;
        }
        // SAFETY: the index buffer holds `index_count` u32 indices from `first_index` (the RHI's contract).
        unsafe {
            e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                primitive,
                index_count as usize,
                MTLIndexType::UInt32,
                &indices,
                offset as usize + first_index as usize * 4,
                instance_count as usize,
                vertex_offset as isize,
                first_instance as usize,
            )
        };
    }

    fn draw_indexed_indirect(&mut self, _buffer: BufferHandle, _offset: u64, _draw_count: u32, _stride: u32) {
        self.error("draw_indexed_indirect is not supported by the Metal backend yet (stage 2)");
    }

    fn draw_indexed_indirect_count(&mut self, _: BufferHandle, _: u64, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.unsupported("draw_indexed_indirect_count", Feature::DrawIndirectCount);
    }

    fn draw_mesh_tasks(&mut self, _x: u32, _y: u32, _z: u32) {
        self.unsupported("draw_mesh_tasks", Feature::MeshShader);
    }

    fn draw_mesh_tasks_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.unsupported("draw_mesh_tasks_indirect", Feature::MeshShader);
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let Some(e) = self.compute_encoder("dispatch") else { return };
        let Some(workgroup) = self.apply_compute(&e) else { return };
        let groups = MTLSize { width: x as usize, height: y as usize, depth: z as usize };
        e.dispatchThreadgroups_threadsPerThreadgroup(groups, workgroup);
    }

    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        let Some(args) = self.buffer(buffer) else { return };
        let Some(e) = self.compute_encoder("dispatch_indirect") else { return };
        let Some(workgroup) = self.apply_compute(&e) else { return };
        // SAFETY: the buffer holds three u32 group counts at `offset` (the RHI's contract).
        unsafe { e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&args, offset as usize, workgroup) };
    }

    fn copy_buffer(&mut self, src: BufferHandle, src_offset: u64, dst: BufferHandle, dst_offset: u64, size: u64) {
        let (Some(s), Some(d)) = (self.buffer(src), self.buffer(dst)) else { return };
        let Some(e) = self.blit_encoder("copy_buffer") else { return };
        // SAFETY: Metal validates the ranges against the buffers' lengths.
        unsafe { e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&s, src_offset as usize, &d, dst_offset as usize, size as usize) };
    }

    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32) {
        let Some(b) = self.buffer(buffer) else { return };
        if !size.is_multiple_of(4) || !offset.is_multiple_of(4) {
            self.error("fill_buffer: offset and size must be multiples of 4");
            return;
        }
        let bytes = value.to_le_bytes();
        let Some(e) = self.blit_encoder("fill_buffer") else { return };
        if bytes.iter().all(|&byte| byte == bytes[0]) {
            e.fillBuffer_range_value(&b, objc2_foundation::NSRange { location: offset as usize, length: size as usize }, bytes[0]);
            return;
        }
        // Metal fills with one byte only: copy the pattern from a scratch buffer instead.
        let pattern: Vec<u8> = bytes.iter().copied().cycle().take(size as usize).collect();
        // SAFETY: the pointer and length describe `pattern`, which Metal copies before returning.
        let scratch = unsafe {
            self.inner.device.newBufferWithBytes_length_options(bytes_ptr(&pattern), pattern.len(), MTLResourceOptions::StorageModeShared)
        };
        let Some(scratch) = scratch else {
            self.error("fill_buffer: Metal could not allocate the pattern");
            return;
        };
        // SAFETY: both ranges are in bounds of their buffers (Metal validates the destination).
        unsafe { e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&scratch, 0, &b, offset as usize, size as usize) };
    }

    fn copy_buffer_to_texture(&mut self, src: BufferHandle, src_offset: u64, dst: TextureHandle, mip: u32) {
        let (Some(s), Some((t, desc))) = (self.buffer(src), self.texture(dst)) else { return };
        let (w, h) = ((desc.width >> mip).max(1), (desc.height >> mip).max(1));
        let row = (w * desc.format.bytes_per_pixel()) as usize;
        let Some(e) = self.blit_encoder("copy_buffer_to_texture") else { return };
        // SAFETY: the buffer holds the tightly packed mip (the RHI's contract); Metal validates the region.
        unsafe {
            e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &s,
                src_offset as usize,
                row,
                row * h as usize,
                MTLSize { width: w as usize, height: h as usize, depth: 1 },
                &t,
                0,
                mip as usize,
                MTLOrigin { x: 0, y: 0, z: 0 },
            )
        };
    }

    fn copy_texture_to_buffer(&mut self, src: TextureHandle, dst: BufferHandle) {
        let (Some((t, desc)), Some(d)) = (self.texture(src), self.buffer(dst)) else { return };
        let row = (desc.width * desc.format.bytes_per_pixel()) as usize;
        let Some(e) = self.blit_encoder("copy_texture_to_buffer") else { return };
        // SAFETY: as above.
        unsafe {
            e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &t,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: desc.width as usize, height: desc.height as usize, depth: 1 },
                &d,
                0,
                row,
                row * desc.height as usize,
            )
        };
    }

    fn write_timestamp(&mut self, _name: &str) {
        self.unsupported("write_timestamp", Feature::Timestamps);
    }

    fn begin_debug_label(&mut self, name: &str) {
        self.cb.pushDebugGroup(&ns(name));
    }

    fn end_debug_label(&mut self) {
        self.cb.popDebugGroup();
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}
