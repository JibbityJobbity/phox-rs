//! Recording commands.

use crate::convert;
use crate::device::{CommandEntry, Inner, TIMESTAMP_CAPACITY, backend};
use ash::vk;
use phox_rhi::*;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Bound {
    Uniform(BufferHandle),
    Storage(BufferHandle),
    Texture(TextureHandle),
}

struct CurrentPipeline {
    layout: vk::PipelineLayout,
    bind_point: vk::PipelineBindPoint,
    set_bindings: Vec<Vec<Binding>>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
}

pub(crate) struct VulkanCommandList {
    inner: Arc<Inner>,
    entry: Option<CommandEntry>,
    cb: vk::CommandBuffer,
    pipeline: Option<CurrentPipeline>,
    /// What each set's bindings are bound to, by set then binding.
    bound: Vec<HashMap<u32, Bound>>,
    dirty: Vec<bool>,
    errors: Vec<String>,
    timestamps: Vec<String>,
}

const MAX_SETS: usize = 8;

impl VulkanCommandList {
    pub(crate) fn begin(inner: Arc<Inner>) -> Result<Self> {
        let device = &inner.device;
        let entry = {
            let mut st = inner.lock();
            inner.collect(&mut st);
            st.free_entries.pop()
        };
        let entry = match entry {
            Some(entry) => {
                unsafe {
                    device.reset_command_pool(entry.pool, vk::CommandPoolResetFlags::empty()).map_err(backend)?;
                    for &pool in &entry.descriptor_pools {
                        device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()).map_err(backend)?;
                    }
                }
                entry
            }
            None => {
                let pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(inner.queue_family), None) }.map_err(backend)?;
                let query_pool = if inner.caps.has(Feature::Timestamps) {
                    let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(TIMESTAMP_CAPACITY);
                    Some(unsafe { device.create_query_pool(&info, None) }.map_err(backend)?)
                } else {
                    None
                };
                CommandEntry { pool, descriptor_pools: Vec::new(), query_pool }
            }
        };
        let cb = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(entry.pool).command_buffer_count(1)) }.map_err(backend)?[0];
        unsafe {
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(backend)?;
            if let Some(query_pool) = entry.query_pool {
                device.cmd_reset_query_pool(cb, query_pool, 0, TIMESTAMP_CAPACITY);
            }
        }
        Ok(Self {
            inner,
            entry: Some(entry),
            cb,
            pipeline: None,
            bound: vec![HashMap::new(); MAX_SETS],
            dirty: vec![false; MAX_SETS],
            errors: Vec::new(),
            timestamps: Vec::new(),
        })
    }

    pub(crate) fn submit(mut self: Box<Self>) -> Result<SubmitTicket> {
        let inner = self.inner.clone();
        unsafe { inner.device.end_command_buffer(self.cb) }.map_err(backend)?;
        if let Some(error) = self.errors.first() {
            return Err(RhiError::Unsupported(error.clone()));
        }
        let entry = self.entry.take().expect("a list is submitted once");
        let names = std::mem::take(&mut self.timestamps);
        let mut st = inner.lock();
        inner.collect(&mut st);
        let ticket = st.submitted + 1;
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cb)];
        let signals = [vk::SemaphoreSubmitInfo::default().semaphore(inner.timeline).value(ticket).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default().command_buffer_infos(&command_buffers).signal_semaphore_infos(&signals);
        if let Err(e) = unsafe { inner.device.queue_submit2(inner.queue, &[submit], vk::Fence::null()) } {
            st.free_entries.push(entry);
            return Err(backend(format!("queue submit failed: {e}")));
        }
        st.submitted = ticket;
        st.inflight.push((ticket, entry, names));
        Ok(SubmitTicket { value: ticket })
    }

    fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
    }

    fn require(&mut self, feature: Feature, call: &str) -> bool {
        if self.inner.caps.has(feature) {
            true
        } else {
            self.error(format!("{call} requires Feature::{feature}"));
            false
        }
    }

    /// Allocates a descriptor set from the list's pools, adding a pool when the last is full.
    fn allocate_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        let device = &self.inner.device;
        let entry = self.entry.as_mut().expect("recording");
        let layouts = [layout];
        for attempt in 0..2 {
            if let Some(&pool) = entry.descriptor_pools.last() {
                match unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) } {
                    Ok(sets) => return Ok(sets[0]),
                    Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) if attempt == 0 => {}
                    Err(e) => return Err(backend(e)),
                }
            }
            let sizes = [
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(64),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(256),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(64),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(64),
            ];
            let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(128).pool_sizes(&sizes), None) }.map_err(backend)?;
            entry.descriptor_pools.push(pool);
        }
        Err(RhiError::Backend("descriptor pool allocation failed".into()))
    }

    /// Writes and binds the descriptor sets that changed since the last draw or dispatch.
    fn flush_descriptors(&mut self) {
        let Some(pipeline) = &self.pipeline else { return };
        let (layout, bind_point) = (pipeline.layout, pipeline.bind_point);
        let set_bindings = pipeline.set_bindings.clone();
        let set_layouts = pipeline.set_layouts.clone();
        for (set, bindings) in set_bindings.iter().enumerate() {
            if bindings.is_empty() || !self.dirty[set] || bindings.iter().any(|b| b.ty == BindingType::BindlessTextures) {
                continue;
            }
            let descriptor_set = match self.allocate_set(set_layouts[set]) {
                Ok(s) => s,
                Err(e) => {
                    self.error(e.to_string());
                    return;
                }
            };
            let mut buffer_infos: Vec<Vec<vk::DescriptorBufferInfo>> = Vec::new();
            let mut image_infos: Vec<Vec<vk::DescriptorImageInfo>> = Vec::new();
            let mut plan: Vec<(u32, vk::DescriptorType, bool, usize)> = Vec::new(); // binding, type, is_image, index
            {
                let st = self.inner.lock();
                for b in bindings {
                    let Some(bound) = self.bound[set].get(&b.binding).copied() else {
                        self.errors.push(format!("set {set} binding {} was not bound before a draw or dispatch", b.binding));
                        return;
                    };
                    match (b.ty, bound) {
                        (BindingType::UniformBuffer, Bound::Uniform(h)) | (BindingType::StorageBuffer, Bound::Storage(h)) => {
                            let Ok(res) = st.buffers.get(h) else {
                                self.errors.push("a destroyed buffer was bound".into());
                                return;
                            };
                            buffer_infos.push(vec![vk::DescriptorBufferInfo { buffer: res.buffer, offset: 0, range: vk::WHOLE_SIZE }]);
                            plan.push((b.binding, crate::pipeline::descriptor_type(b.ty), false, buffer_infos.len() - 1));
                        }
                        (BindingType::SampledTexture, Bound::Texture(h)) => {
                            let Ok(res) = st.textures.get(h) else {
                                self.errors.push("a destroyed texture was bound".into());
                                return;
                            };
                            let layout = if res.desc.format.is_depth() { vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL } else { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL };
                            // The sampler is immutable in the layout.
                            image_infos.push(vec![vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: res.view, image_layout: layout }]);
                            plan.push((b.binding, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, true, image_infos.len() - 1));
                        }
                        _ => {
                            self.errors.push(format!("set {set} binding {} is bound to the wrong kind of resource", b.binding));
                            return;
                        }
                    }
                }
            }
            let writes: Vec<vk::WriteDescriptorSet> = plan
                .iter()
                .map(|&(binding, ty, is_image, index)| {
                    let write = vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(binding).descriptor_type(ty);
                    if is_image { write.image_info(&image_infos[index]) } else { write.buffer_info(&buffer_infos[index]) }
                })
                .collect();
            unsafe {
                self.inner.device.update_descriptor_sets(&writes, &[]);
                self.inner.device.cmd_bind_descriptor_sets(self.cb, bind_point, layout, set as u32, &[descriptor_set], &[]);
            }
            self.dirty[set] = false;
        }
    }

    fn bind(&mut self, set: u32, binding: u32, bound: Bound) {
        if set as usize >= MAX_SETS {
            self.error(format!("descriptor set {set} is out of range"));
            return;
        }
        self.bound[set as usize].insert(binding, bound);
        self.dirty[set as usize] = true;
    }

    fn buffer(&mut self, handle: BufferHandle) -> Option<vk::Buffer> {
        match self.inner.lock().buffers.get(handle) {
            Ok(b) => Some(b.buffer),
            Err(_) => {
                self.errors.push("a destroyed buffer was used in a command".into());
                None
            }
        }
    }
}

impl Drop for VulkanCommandList {
    fn drop(&mut self) {
        if let Some(entry) = self.entry.take() {
            // Never submitted: its pool is reset when it is next used.
            self.inner.lock().free_entries.push(entry);
        }
    }
}

impl CommandList for VulkanCommandList {
    fn barrier(&mut self, textures: &[TextureBarrier], buffers: &[BufferBarrier]) {
        let mesh = self.inner.mesh_shaders();
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        {
            let st = self.inner.lock();
            for b in textures {
                let Ok(t) = st.textures.get(b.texture) else {
                    self.errors.push("a barrier names a destroyed texture".into());
                    continue;
                };
                let src = convert::texture_access(b.from, t.desc.format, mesh);
                let dst = convert::texture_access(b.to, t.desc.format, mesh);
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src.stages)
                        .src_access_mask(convert::write_only(src.access))
                        .dst_stage_mask(dst.stages)
                        .dst_access_mask(dst.access)
                        .old_layout(src.layout)
                        .new_layout(dst.layout)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(t.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: convert::aspect(t.desc.format),
                            base_mip_level: 0,
                            level_count: t.desc.mips,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                );
            }
            for b in buffers {
                let Ok(res) = st.buffers.get(b.buffer) else {
                    self.errors.push("a barrier names a destroyed buffer".into());
                    continue;
                };
                let (src_stage, src_access) = convert::buffer_access(b.from, mesh);
                let (dst_stage, dst_access) = convert::buffer_access(b.to, mesh);
                buffer_barriers.push(
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(src_stage)
                        .src_access_mask(convert::write_only(src_access))
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(res.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE),
                );
            }
        }
        if image_barriers.is_empty() && buffer_barriers.is_empty() {
            return;
        }
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&image_barriers).buffer_memory_barriers(&buffer_barriers);
        unsafe { self.inner.device.cmd_pipeline_barrier2(self.cb, &dependency) };
    }

    fn begin_rendering(&mut self, info: &RenderingInfo) {
        let mut colors = Vec::new();
        let mut depth = None;
        {
            let st = self.inner.lock();
            for c in &info.colors {
                let Ok(t) = st.textures.get(c.texture) else {
                    self.errors.push("a colour target is a destroyed texture".into());
                    return;
                };
                colors.push(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(t.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(if c.clear { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD })
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: c.clear_color } }),
                );
            }
            if let Some(d) = &info.depth {
                let Ok(t) = st.textures.get(d.texture) else {
                    self.errors.push("the depth target is a destroyed texture".into());
                    return;
                };
                depth = Some(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(t.view)
                        .image_layout(if d.read_only { vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL } else { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL })
                        .load_op(if d.clear { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD })
                        .store_op(if d.read_only { vk::AttachmentStoreOp::NONE } else { vk::AttachmentStoreOp::STORE })
                        .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: d.clear_depth, stencil: 0 } }),
                );
            }
        }
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: info.width, height: info.height } };
        let mut rendering = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors);
        if let Some(d) = &depth {
            rendering = rendering.depth_attachment(d);
        }
        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: info.width as f32, height: info.height as f32, min_depth: 0.0, max_depth: 1.0 };
        unsafe {
            self.inner.device.cmd_begin_rendering(self.cb, &rendering);
            self.inner.device.cmd_set_viewport(self.cb, 0, &[viewport]);
            self.inner.device.cmd_set_scissor(self.cb, 0, &[area]);
        }
    }

    fn end_rendering(&mut self) {
        unsafe { self.inner.device.cmd_end_rendering(self.cb) };
    }

    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let current = {
            let st = self.inner.lock();
            st.pipelines.get(pipeline).map(|p| (p.pipeline, CurrentPipeline { layout: p.layout, bind_point: p.bind_point, set_bindings: p.set_bindings.clone(), set_layouts: p.set_layouts.clone() }))
        };
        let Ok((vk_pipeline, current)) = current else {
            self.error("bind_pipeline: a destroyed pipeline");
            return;
        };
        unsafe { self.inner.device.cmd_bind_pipeline(self.cb, current.bind_point, vk_pipeline) };
        self.pipeline = Some(current);
        self.dirty.iter_mut().for_each(|d| *d = true);
    }

    fn bind_vertex_buffer(&mut self, buffer: BufferHandle, offset: u64) {
        if let Some(b) = self.buffer(buffer) {
            unsafe { self.inner.device.cmd_bind_vertex_buffers(self.cb, 0, &[b], &[offset]) };
        }
    }

    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64) {
        if let Some(b) = self.buffer(buffer) {
            unsafe { self.inner.device.cmd_bind_index_buffer(self.cb, b, offset, vk::IndexType::UINT32) };
        }
    }

    fn bind_uniform_buffer(&mut self, set: u32, binding: u32, buffer: BufferHandle) {
        self.bind(set, binding, Bound::Uniform(buffer));
    }

    fn bind_storage_buffer(&mut self, set: u32, binding: u32, buffer: BufferHandle) {
        self.bind(set, binding, Bound::Storage(buffer));
    }

    fn bind_sampled_texture(&mut self, set: u32, binding: u32, texture: TextureHandle) {
        self.bind(set, binding, Bound::Texture(texture));
    }

    fn bind_bindless_table(&mut self, set: u32) {
        let Some(p) = &self.pipeline else {
            self.error("bind_bindless_table: no pipeline is bound");
            return;
        };
        let (layout, bind_point) = (p.layout, p.bind_point);
        let table = self.inner.lock().bindless.as_ref().map(|t| t.set);
        match table {
            Some(table) => unsafe { self.inner.device.cmd_bind_descriptor_sets(self.cb, bind_point, layout, set, &[table], &[]) },
            None => self.error("bind_bindless_table requires Feature::BindlessDescriptors"),
        }
    }

    fn push_constants(&mut self, stages: ShaderStages, offset: u32, data: &[u8]) {
        let Some(p) = &self.pipeline else {
            self.error("push_constants: no pipeline is bound");
            return;
        };
        unsafe { self.inner.device.cmd_push_constants(self.cb, p.layout, convert::shader_stages(stages), offset, data) };
    }

    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32) {
        self.flush_descriptors();
        unsafe { self.inner.device.cmd_draw(self.cb, vertex_count, instance_count, first_vertex, 0) };
    }

    fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        self.flush_descriptors();
        unsafe { self.inner.device.cmd_draw_indexed(self.cb, index_count, instance_count, first_index, vertex_offset, first_instance) };
    }

    fn draw_indexed_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        self.flush_descriptors();
        if let Some(b) = self.buffer(buffer) {
            unsafe { self.inner.device.cmd_draw_indexed_indirect(self.cb, b, offset, draw_count, stride) };
        }
    }

    fn draw_indexed_indirect_count(&mut self, buffer: BufferHandle, offset: u64, count_buffer: BufferHandle, count_offset: u64, max_draw_count: u32, stride: u32) {
        if !self.require(Feature::DrawIndirectCount, "draw_indexed_indirect_count") {
            return;
        }
        self.flush_descriptors();
        if let (Some(b), Some(c)) = (self.buffer(buffer), self.buffer(count_buffer)) {
            unsafe { self.inner.device.cmd_draw_indexed_indirect_count(self.cb, b, offset, c, count_offset, max_draw_count, stride) };
        }
    }

    fn draw_mesh_tasks(&mut self, x: u32, y: u32, z: u32) {
        if !self.require(Feature::MeshShader, "draw_mesh_tasks") {
            return;
        }
        self.flush_descriptors();
        if let Some(loader) = &self.inner.mesh_loader {
            unsafe { loader.cmd_draw_mesh_tasks(self.cb, x, y, z) };
        }
    }

    fn draw_mesh_tasks_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        if !self.require(Feature::MeshShader, "draw_mesh_tasks_indirect") {
            return;
        }
        self.flush_descriptors();
        if let (Some(b), Some(loader)) = (self.buffer(buffer), &self.inner.mesh_loader) {
            unsafe { loader.cmd_draw_mesh_tasks_indirect(self.cb, b, offset, draw_count, stride) };
        }
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.flush_descriptors();
        unsafe { self.inner.device.cmd_dispatch(self.cb, x, y, z) };
    }

    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        self.flush_descriptors();
        if let Some(b) = self.buffer(buffer) {
            unsafe { self.inner.device.cmd_dispatch_indirect(self.cb, b, offset) };
        }
    }

    fn copy_buffer(&mut self, src: BufferHandle, src_offset: u64, dst: BufferHandle, dst_offset: u64, size: u64) {
        if let (Some(s), Some(d)) = (self.buffer(src), self.buffer(dst)) {
            let region = vk::BufferCopy { src_offset, dst_offset, size };
            unsafe { self.inner.device.cmd_copy_buffer(self.cb, s, d, &[region]) };
        }
    }

    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32) {
        if let Some(b) = self.buffer(buffer) {
            unsafe { self.inner.device.cmd_fill_buffer(self.cb, b, offset, size, value) };
        }
    }

    fn copy_buffer_to_texture(&mut self, src: BufferHandle, src_offset: u64, dst: TextureHandle, mip: u32) {
        let Some(s) = self.buffer(src) else { return };
        let target = self.inner.lock().textures.get(dst).map(|t| (t.image, t.desc.clone()));
        let Ok((image, desc)) = target else {
            self.error("copy_buffer_to_texture: a destroyed texture");
            return;
        };
        let region = vk::BufferImageCopy {
            buffer_offset: src_offset,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: convert::aspect(desc.format), mip_level: mip, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: (desc.width >> mip).max(1), height: (desc.height >> mip).max(1), depth: 1 },
        };
        unsafe { self.inner.device.cmd_copy_buffer_to_image(self.cb, s, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]) };
    }

    fn copy_texture_to_buffer(&mut self, src: TextureHandle, dst: BufferHandle) {
        let Some(d) = self.buffer(dst) else { return };
        let source = self.inner.lock().textures.get(src).map(|t| (t.image, t.desc.clone()));
        let Ok((image, desc)) = source else {
            self.error("copy_texture_to_buffer: a destroyed texture");
            return;
        };
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: convert::aspect(desc.format), mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: desc.width, height: desc.height, depth: 1 },
        };
        unsafe { self.inner.device.cmd_copy_image_to_buffer(self.cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, d, &[region]) };
    }

    fn write_timestamp(&mut self, name: &str) {
        if !self.require(Feature::Timestamps, "write_timestamp") {
            return;
        }
        let index = self.timestamps.len() as u32;
        let Some(query_pool) = self.entry.as_ref().and_then(|e| e.query_pool) else { return };
        if index >= TIMESTAMP_CAPACITY {
            self.error("too many timestamps in one command list");
            return;
        }
        unsafe { self.inner.device.cmd_write_timestamp2(self.cb, vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, index) };
        self.timestamps.push(name.to_string());
    }

    fn begin_debug_label(&mut self, name: &str) {
        if let (Some(loader), Ok(c_name)) = (&self.inner.debug_utils, std::ffi::CString::new(name)) {
            let label = vk::DebugUtilsLabelEXT::default().label_name(&c_name);
            unsafe { loader.cmd_begin_debug_utils_label(self.cb, &label) };
        }
    }

    fn end_debug_label(&mut self) {
        if let Some(loader) = &self.inner.debug_utils {
            unsafe { loader.cmd_end_debug_utils_label(self.cb) };
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}
