//! A device that owns handles and descriptions but never touches a GPU, and a command list that records the
//! name of every call. For testing graph logic (and anything else that needs a [`Device`]).

use phox_rhi::pool::Pool;
use phox_rhi::*;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

/// Records the name of every call. `begin_debug_label` is recorded as `beginDebugLabel:<label>`.
pub struct NullCommandList {
    caps: Capabilities,
    recording: Vec<String>,
    timestamps: Vec<String>,
    errors: Vec<String>,
}

impl NullCommandList {
    fn push(&mut self, name: &str) {
        self.recording.push(name.to_string());
    }

    fn require(&mut self, feature: Feature, call: &str) {
        if !self.caps.has(feature) {
            self.errors.push(format!("{call} requires Feature::{feature}"));
        }
    }
}

impl CommandList for NullCommandList {
    fn barrier(&mut self, _: &[TextureBarrier], _: &[BufferBarrier]) {
        self.push("barrier");
    }
    fn begin_rendering(&mut self, _: &RenderingInfo) {
        self.push("beginRendering");
    }
    fn end_rendering(&mut self) {
        self.push("endRendering");
    }
    fn bind_pipeline(&mut self, _: PipelineHandle) {
        self.push("bindPipeline");
    }
    fn bind_vertex_buffer(&mut self, _: BufferHandle, _: u64) {
        self.push("bindVertexBuffer");
    }
    fn bind_index_buffer(&mut self, _: BufferHandle, _: u64) {
        self.push("bindIndexBuffer");
    }
    fn bind_uniform_buffer(&mut self, _: u32, _: u32, _: BufferHandle) {
        self.push("bindUniformBuffer");
    }
    fn bind_storage_buffer(&mut self, _: u32, _: u32, _: BufferHandle) {
        self.push("bindStorageBuffer");
    }
    fn bind_sampled_texture(&mut self, _: u32, _: u32, _: TextureHandle) {
        self.push("bindSampledTexture");
    }
    fn bind_bindless_table(&mut self, _: u32) {
        self.require(Feature::BindlessDescriptors, "bind_bindless_table");
        self.push("bindBindlessTable");
    }
    fn push_constants(&mut self, _: ShaderStages, _: u32, _: &[u8]) {
        self.push("pushConstants");
    }
    fn draw(&mut self, _: u32, _: u32, _: u32) {
        self.push("draw");
    }
    fn draw_indexed(&mut self, _: u32, _: u32, _: u32, _: i32, _: u32) {
        self.push("drawIndexed");
    }
    fn draw_indexed_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.push("drawIndexedIndirect");
    }
    fn draw_indexed_indirect_count(&mut self, _: BufferHandle, _: u64, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.require(Feature::DrawIndirectCount, "draw_indexed_indirect_count");
        self.push("drawIndexedIndirectCount");
    }
    fn draw_mesh_tasks(&mut self, _: u32, _: u32, _: u32) {
        self.require(Feature::MeshShader, "draw_mesh_tasks");
        self.push("drawMeshTasks");
    }
    fn draw_mesh_tasks_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.require(Feature::MeshShader, "draw_mesh_tasks_indirect");
        self.push("drawMeshTasksIndirect");
    }
    fn dispatch(&mut self, _: u32, _: u32, _: u32) {
        self.push("dispatch");
    }
    fn dispatch_indirect(&mut self, _: BufferHandle, _: u64) {
        self.push("dispatchIndirect");
    }
    fn copy_buffer(&mut self, _: BufferHandle, _: u64, _: BufferHandle, _: u64, _: u64) {
        self.push("copyBuffer");
    }
    fn fill_buffer(&mut self, _: BufferHandle, _: u64, _: u64, _: u32) {
        self.push("fillBuffer");
    }
    fn copy_buffer_to_texture(&mut self, _: BufferHandle, _: u64, _: TextureHandle, _: u32) {
        self.push("copyBufferToTexture");
    }
    fn copy_texture_to_buffer(&mut self, _: TextureHandle, _: BufferHandle) {
        self.push("copyTextureToBuffer");
    }
    fn write_timestamp(&mut self, name: &str) {
        self.require(Feature::Timestamps, "write_timestamp");
        self.timestamps.push(name.to_string());
        self.push("writeTimestamp");
    }
    fn begin_debug_label(&mut self, name: &str) {
        self.recording.push(format!("beginDebugLabel:{name}"));
    }
    fn end_debug_label(&mut self) {
        self.push("endDebugLabel");
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

struct BufferData {
    desc: BufferDesc,
    memory: Vec<u8>, // zero-filled; only for Upload / Readback buffers
}

struct State {
    caps: Capabilities,
    buffers: Pool<BufferTag, BufferData>,
    textures: Pool<TextureTag, TextureDesc>,
    shaders: Pool<ShaderTag, ShaderStage>,
    pipelines: Pool<PipelineTag, bool>, // is compute
    last_recording: Vec<String>,
    ticket: u64,
    timestamps: VecDeque<(u64, Vec<Timestamp>)>,
    bindless_next: u32,
    bindless_free: Vec<u32>,
}

/// A device that never touches a GPU. Reports no features until [`NullDevice::set_features`].
pub struct NullDevice {
    state: Mutex<State>,
}

impl Default for NullDevice {
    fn default() -> Self {
        Self::new()
    }
}

fn invalid(message: &str) -> RhiError {
    RhiError::InvalidDesc(message.to_string())
}

impl NullDevice {
    pub fn new() -> Self {
        let caps = Capabilities {
            device_name: "Null".into(),
            max_texture_size_2d: 16384,
            max_push_constant_bytes: 128,
            max_draw_indirect_count: 1 << 20,
            timestamp_period_ns: 1.0,
            bindless_texture_capacity: 4096,
            ..Capabilities::default()
        };
        Self {
            state: Mutex::new(State {
                caps,
                buffers: Pool::new("buffer"),
                textures: Pool::new("texture"),
                shaders: Pool::new("shader"),
                pipelines: Pool::new("pipeline"),
                last_recording: Vec::new(),
                ticket: 0,
                timestamps: VecDeque::new(),
                bindless_next: 0,
                bindless_free: Vec::new(),
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Overrides the reported feature set (default: none).
    pub fn set_features(&self, features: impl Into<FeatureSet>) {
        self.state().caps.features = features.into();
    }

    /// Call names recorded by the most recently submitted command list; empty before the first submit.
    pub fn last_recording(&self) -> Vec<String> {
        self.state().last_recording.clone()
    }
}

impl Device for NullDevice {
    fn capabilities(&self) -> Capabilities {
        self.state().caps.clone()
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferHandle> {
        if desc.size == 0 {
            return Err(invalid("create_buffer: size must be greater than zero"));
        }
        let memory = if desc.memory == MemoryKind::GpuOnly { Vec::new() } else { vec![0; desc.size as usize] };
        Ok(self.state().buffers.create(BufferData { desc: desc.clone(), memory }))
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureHandle> {
        if desc.width == 0 || desc.height == 0 || desc.mips == 0 {
            return Err(invalid("create_texture: width, height and mips must be greater than zero"));
        }
        if desc.format == Format::Undefined {
            return Err(invalid("create_texture: format must not be Undefined"));
        }
        Ok(self.state().textures.create(desc.clone()))
    }

    fn create_shader(&self, desc: &ShaderDesc) -> Result<ShaderHandle> {
        if desc.spirv.is_empty() {
            return Err(invalid("create_shader: spirv must not be empty"));
        }
        Ok(self.state().shaders.create(desc.stage))
    }

    fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle> {
        let mut state = self.state();
        let (mut vertex, mut mesh, mut fragment) = (false, false, false);
        let mut seen: Vec<ShaderStage> = Vec::new();
        for &shader in &desc.shaders {
            let stage = *state.shaders.get(shader)?;
            if seen.contains(&stage) {
                return Err(invalid("create_graphics_pipeline: duplicate shader stage"));
            }
            seen.push(stage);
            vertex |= stage == ShaderStage::Vertex;
            mesh |= stage == ShaderStage::Mesh;
            fragment |= stage == ShaderStage::Fragment;
            if stage == ShaderStage::Compute {
                return Err(invalid("create_graphics_pipeline: compute shader in a graphics pipeline"));
            }
        }
        if !fragment || (!vertex && !mesh) {
            return Err(invalid("create_graphics_pipeline: needs a Fragment shader and a Vertex or Mesh shader"));
        }
        Ok(state.pipelines.create(false))
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle> {
        let mut state = self.state();
        if *state.shaders.get(desc.shader)? != ShaderStage::Compute {
            return Err(invalid("create_compute_pipeline: shader is not a Compute shader"));
        }
        Ok(state.pipelines.create(true))
    }

    fn destroy_buffer(&self, handle: BufferHandle) {
        self.state().buffers.destroy(handle);
    }
    fn destroy_texture(&self, handle: TextureHandle) {
        self.state().textures.destroy(handle);
    }
    fn destroy_shader(&self, handle: ShaderHandle) {
        self.state().shaders.destroy(handle);
    }
    fn destroy_pipeline(&self, handle: PipelineHandle) {
        self.state().pipelines.destroy(handle);
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let mut state = self.state();
        let b = state.buffers.get_mut(buffer)?;
        if b.desc.memory == MemoryKind::GpuOnly {
            return Err(RhiError::Misuse("write_buffer: buffer memory is GpuOnly".into()));
        }
        let end = offset.checked_add(data.len() as u64).filter(|&e| e <= b.desc.size);
        let end = end.ok_or_else(|| RhiError::Misuse("write_buffer: write is out of bounds".into()))?;
        b.memory[offset as usize..end as usize].copy_from_slice(data);
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, out: &mut [u8]) -> Result<()> {
        let state = self.state();
        let b = state.buffers.get(buffer)?;
        if b.desc.memory == MemoryKind::GpuOnly {
            return Err(RhiError::Misuse("read_buffer: buffer memory is GpuOnly".into()));
        }
        let end = offset.checked_add(out.len() as u64).filter(|&e| e <= b.desc.size);
        let end = end.ok_or_else(|| RhiError::Misuse("read_buffer: read is out of bounds".into()))?;
        out.copy_from_slice(&b.memory[offset as usize..end as usize]);
        Ok(())
    }

    fn describe_texture(&self, handle: TextureHandle) -> Result<TextureDesc> {
        self.state().textures.get(handle).cloned()
    }

    fn describe_buffer(&self, handle: BufferHandle) -> Result<BufferDesc> {
        self.state().buffers.get(handle).map(|b| b.desc.clone())
    }

    fn register_bindless_texture(&self, texture: TextureHandle, _sampler: &SamplerDesc) -> Result<BindlessIndex> {
        let mut state = self.state();
        if !state.caps.has(Feature::BindlessDescriptors) {
            return Err(RhiError::Unsupported("BindlessDescriptors".into()));
        }
        state.textures.get(texture)?;
        if let Some(index) = state.bindless_free.pop() {
            return Ok(BindlessIndex(index));
        }
        if state.bindless_next >= state.caps.bindless_texture_capacity {
            return Err(RhiError::Misuse("the bindless texture table is full".into()));
        }
        state.bindless_next += 1;
        Ok(BindlessIndex(state.bindless_next - 1))
    }

    fn unregister_bindless_texture(&self, index: BindlessIndex) {
        let mut state = self.state();
        if index.0 < state.bindless_next && !state.bindless_free.contains(&index.0) {
            state.bindless_free.push(index.0);
        }
    }

    fn begin_commands(&self) -> Result<Box<dyn CommandList>> {
        Ok(Box::new(NullCommandList {
            caps: self.state().caps.clone(),
            recording: Vec::new(),
            timestamps: Vec::new(),
            errors: Vec::new(),
        }))
    }

    fn submit(&self, commands: Box<dyn CommandList>) -> Result<SubmitTicket> {
        let list = commands
            .into_any()
            .downcast::<NullCommandList>()
            .map_err(|_| RhiError::Misuse("submit: the command list was not made by this device".into()))?;
        if let Some(error) = list.errors.first() {
            return Err(RhiError::Unsupported(error.clone()));
        }
        let mut state = self.state();
        state.ticket += 1;
        let ticket = state.ticket;
        let stamps = list
            .timestamps
            .iter()
            .enumerate()
            .map(|(i, name)| Timestamp { name: name.clone(), nanoseconds: 1000.0 * i as f64 })
            .collect();
        state.timestamps.push_back((ticket, stamps));
        if state.timestamps.len() > 16 {
            state.timestamps.pop_front();
        }
        state.last_recording = list.recording;
        Ok(SubmitTicket { value: ticket })
    }

    fn is_complete(&self, ticket: SubmitTicket) -> bool {
        ticket.value <= self.state().ticket
    }

    fn wait(&self, _ticket: SubmitTicket) -> Result<()> {
        Ok(())
    }

    fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    fn read_timestamps(&self, ticket: SubmitTicket) -> Result<Vec<Timestamp>> {
        let state = self.state();
        Ok(state.timestamps.iter().find(|(t, _)| *t == ticket.value).map(|(_, s)| s.clone()).unwrap_or_default())
    }
}
