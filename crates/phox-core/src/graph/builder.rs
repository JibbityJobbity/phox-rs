use super::*;
use phox_rhi::{BufferDesc, CommandList, TextureDesc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureAccess {
    ShaderRead,
    ShaderWrite,
    TransferSrc,
    TransferDst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    Vertex,
    Index,
    Uniform,
    ShaderRead,
    ShaderWrite,
    /// Indirect draw or dispatch arguments (a read; shaders may read them too).
    Indirect,
    TransferSrc,
    TransferDst,
    HostWrite,
}

impl TextureAccess {
    pub(crate) fn state(self) -> ResourceState {
        match self {
            TextureAccess::ShaderRead => ResourceState::ShaderRead,
            TextureAccess::ShaderWrite => ResourceState::ShaderWrite,
            TextureAccess::TransferSrc => ResourceState::TransferSrc,
            TextureAccess::TransferDst => ResourceState::TransferDst,
        }
    }
    pub(crate) fn usage(self) -> TextureUsage {
        match self {
            TextureAccess::ShaderRead => TextureUsage::SAMPLED,
            TextureAccess::ShaderWrite => TextureUsage::STORAGE,
            TextureAccess::TransferSrc => TextureUsage::TRANSFER_SRC,
            TextureAccess::TransferDst => TextureUsage::TRANSFER_DST,
        }
    }
}

impl BufferAccess {
    pub(crate) fn state(self) -> ResourceState {
        match self {
            BufferAccess::Vertex | BufferAccess::Index => ResourceState::VertexOrIndex,
            BufferAccess::Uniform => ResourceState::Uniform,
            BufferAccess::ShaderRead => ResourceState::ShaderRead,
            BufferAccess::ShaderWrite => ResourceState::ShaderWrite,
            BufferAccess::Indirect => ResourceState::IndirectArgs,
            BufferAccess::TransferSrc => ResourceState::TransferSrc,
            BufferAccess::TransferDst => ResourceState::TransferDst,
            BufferAccess::HostWrite => ResourceState::Undefined, // no barrier
        }
    }
    pub(crate) fn usage(self) -> BufferUsage {
        match self {
            BufferAccess::Vertex => BufferUsage::VERTEX,
            BufferAccess::Index => BufferUsage::INDEX,
            BufferAccess::Uniform => BufferUsage::UNIFORM,
            BufferAccess::ShaderRead | BufferAccess::ShaderWrite => BufferUsage::STORAGE,
            BufferAccess::Indirect => BufferUsage::INDIRECT,
            BufferAccess::TransferSrc => BufferUsage::TRANSFER_SRC,
            BufferAccess::TransferDst => BufferUsage::TRANSFER_DST,
            BufferAccess::HostWrite => BufferUsage::NONE,
        }
    }
    fn is_write(self) -> bool {
        matches!(self, BufferAccess::ShaderWrite | BufferAccess::TransferDst | BufferAccess::HostWrite)
    }
}

/// Declares one frame's graph.
pub struct GraphBuilder<'g> {
    pub(crate) graph: &'g mut RenderGraph,
    pub(crate) data: GraphData<'g>,
    pub(crate) timestamps: bool,
}

impl<'g> GraphBuilder<'g> {
    pub(crate) fn new(graph: &'g mut RenderGraph) -> Self {
        Self { graph, data: GraphData::default(), timestamps: false }
    }

    /// Writes a GPU timestamp before and after every pass (when the device has `Feature::Timestamps`), for
    /// [`Device::read_timestamps`].
    pub fn set_timestamps(&mut self, enabled: bool) {
        self.timestamps = enabled;
    }

    /// A transient texture. `desc.usage` is IGNORED: it is derived from the accesses at `compile()`.
    pub fn create_texture(&mut self, desc: &TextureDesc) -> GraphTexture {
        let name = desc.debug_name.clone();
        self.data.textures.push(TextureNode {
            desc: TextureDesc { debug_name: String::new(), ..desc.clone() },
            name,
            imported: false,
            handle: TextureHandle::INVALID,
            before: ResourceState::Undefined,
            after: ResourceState::Undefined,
        });
        GraphTexture { id: self.data.textures.len() as u32 - 1 }
    }

    /// A transient buffer; the same rule for `usage`.
    pub fn create_buffer(&mut self, desc: &BufferDesc) -> GraphBuffer {
        let name = desc.debug_name.clone();
        self.data.buffers.push(BufferNode {
            desc: BufferDesc { debug_name: String::new(), ..desc.clone() },
            name,
            imported: false,
            handle: BufferHandle::INVALID,
            before: ResourceState::Undefined,
            after: ResourceState::Undefined,
        });
        GraphBuffer { id: self.data.buffers.len() as u32 - 1 }
    }

    /// A texture the caller owns, in state `before` when the graph starts and `after` when it ends
    /// (`Undefined`: leave it wherever it ends up).
    pub fn import_texture(&mut self, handle: TextureHandle, before: ResourceState, after: ResourceState) -> GraphResult<GraphTexture> {
        if !handle.is_valid() {
            return Err(GraphError::Declaration("import_texture: invalid texture handle".into()));
        }
        let desc = self.graph.device.describe_texture(handle)?;
        self.data.textures.push(TextureNode {
            name: desc.debug_name.clone(),
            desc: TextureDesc { debug_name: String::new(), ..desc },
            imported: true,
            handle,
            before,
            after,
        });
        Ok(GraphTexture { id: self.data.textures.len() as u32 - 1 })
    }

    pub fn import_buffer(&mut self, handle: BufferHandle, before: ResourceState, after: ResourceState) -> GraphResult<GraphBuffer> {
        if !handle.is_valid() {
            return Err(GraphError::Declaration("import_buffer: invalid buffer handle".into()));
        }
        let desc = self.graph.device.describe_buffer(handle)?;
        self.data.buffers.push(BufferNode {
            name: desc.debug_name.clone(),
            desc: BufferDesc { debug_name: String::new(), ..desc },
            imported: true,
            handle,
            before,
            after,
        });
        Ok(GraphBuffer { id: self.data.buffers.len() as u32 - 1 })
    }

    /// The declared description (size, format, ...). For a transient the `usage` is not final: it is derived
    /// from the accesses at `compile()`. Components use this to size resources they derive from an input.
    pub fn describe_texture(&self, texture: GraphTexture) -> GraphResult<&TextureDesc> {
        self.data
            .textures
            .get(texture.id as usize)
            .map(|t| &t.desc)
            .ok_or_else(|| GraphError::Declaration("describe: invalid texture handle".into()))
    }

    pub fn describe_buffer(&self, buffer: GraphBuffer) -> GraphResult<&BufferDesc> {
        self.data
            .buffers
            .get(buffer.id as usize)
            .map(|b| &b.desc)
            .ok_or_else(|| GraphError::Declaration("describe: invalid buffer handle".into()))
    }

    /// Declares that the graph needs `feature`; `compile()` fails, naming it, when the device lacks it.
    pub fn require(&mut self, feature: Feature) {
        if !self.data.required.contains(&feature) {
            self.data.required.push(feature);
        }
    }

    /// Adds a pass. `setup` runs immediately and declares what the pass touches; `execute` runs during
    /// [`GraphBuilder::execute`] while the pass's commands are recorded. A failing `setup` leaves the graph
    /// as it was.
    pub fn add_pass<S, E>(&mut self, name: &str, setup: S, execute: E) -> GraphResult<()>
    where
        S: FnOnce(&mut PassBuilder<'_>) -> GraphResult<()>,
        E: FnOnce(&mut PassContext<'_>) -> GraphResult<()> + 'g,
    {
        let mut decl = PassDecl { name: name.to_string(), ..PassDecl::default() };
        let mut builder = PassBuilder { pass: &mut decl, textures: self.data.textures.len(), buffers: self.data.buffers.len() };
        setup(&mut builder)?;
        self.data.passes.push(PassNode {
            decl,
            execute: Some(Box::new(execute)),
            live: false,
            texture_barriers: Vec::new(),
            buffer_barriers: Vec::new(),
        });
        Ok(())
    }

    /// The handle of a resource after `compile()`: the imported handle, the pooled transient, or an invalid
    /// handle when only culled passes use it.
    pub fn resolve_texture(&self, texture: GraphTexture) -> GraphResult<TextureHandle> {
        self.data
            .textures
            .get(texture.id as usize)
            .map(|t| t.handle)
            .ok_or_else(|| GraphError::Declaration("resolve: invalid texture handle".into()))
    }

    pub fn resolve_buffer(&self, buffer: GraphBuffer) -> GraphResult<BufferHandle> {
        self.data
            .buffers
            .get(buffer.id as usize)
            .map(|b| b.handle)
            .ok_or_else(|| GraphError::Declaration("resolve: invalid buffer handle".into()))
    }
}

/// Declares what one pass touches. Handed to the `setup` callback of [`GraphBuilder::add_pass`].
pub struct PassBuilder<'a> {
    pass: &'a mut PassDecl,
    textures: usize,
    buffers: usize,
}

impl PassBuilder<'_> {
    fn check_texture(&self, texture: GraphTexture) -> GraphResult<()> {
        if (texture.id as usize) < self.textures {
            Ok(())
        } else {
            Err(GraphError::Declaration(format!("pass '{}' uses an invalid texture handle", self.pass.name)))
        }
    }

    fn check_buffer(&self, buffer: GraphBuffer) -> GraphResult<()> {
        if (buffer.id as usize) < self.buffers {
            Ok(())
        } else {
            Err(GraphError::Declaration(format!("pass '{}' uses an invalid buffer handle", self.pass.name)))
        }
    }

    /// Renders to `texture`, clearing it to black first.
    pub fn color_target(&mut self, texture: GraphTexture) -> GraphResult<()> {
        self.color_target_with(texture, true, [0.0, 0.0, 0.0, 1.0])
    }

    /// Renders to `texture`, keeping what it holds.
    pub fn color_target_load(&mut self, texture: GraphTexture) -> GraphResult<()> {
        self.color_target_with(texture, false, [0.0; 4])
    }

    pub fn color_target_with(&mut self, texture: GraphTexture, clear: bool, clear_color: [f32; 4]) -> GraphResult<()> {
        self.check_texture(texture)?;
        self.pass.colors.push(ColorDecl { id: texture.id, clear, value: clear_color });
        self.pass.textures.push(TextureUse { id: texture.id, state: ResourceState::ColorAttachment, usage: TextureUsage::COLOR_ATTACHMENT, write: true });
        Ok(())
    }

    /// Tests and writes depth in `texture`, clearing it to 1 first.
    pub fn depth_target(&mut self, texture: GraphTexture) -> GraphResult<()> {
        self.depth_target_with(texture, true, 1.0)
    }

    pub fn depth_target_with(&mut self, texture: GraphTexture, clear: bool, clear_depth: f32) -> GraphResult<()> {
        self.check_texture(texture)?;
        if self.pass.depth.is_some() {
            return Err(GraphError::Declaration(format!("pass '{}' declares more than one depth target", self.pass.name)));
        }
        self.pass.depth = Some(DepthDecl { id: texture.id, clear, value: clear_depth, read_only: false });
        self.pass.textures.push(TextureUse { id: texture.id, state: ResourceState::DepthWrite, usage: TextureUsage::DEPTH_ATTACHMENT, write: true });
        Ok(())
    }

    /// Tests depth against `texture` without writing it (and lets the pass's shaders sample it).
    pub fn depth_read(&mut self, texture: GraphTexture) -> GraphResult<()> {
        self.check_texture(texture)?;
        if self.pass.depth.is_some() {
            return Err(GraphError::Declaration(format!("pass '{}' declares more than one depth target", self.pass.name)));
        }
        self.pass.depth = Some(DepthDecl { id: texture.id, clear: false, value: 1.0, read_only: true });
        self.pass.textures.push(TextureUse {
            id: texture.id,
            state: ResourceState::DepthRead,
            usage: TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
            write: false,
        });
        Ok(())
    }

    pub fn read_texture(&mut self, texture: GraphTexture, access: TextureAccess) -> GraphResult<()> {
        self.check_texture(texture)?;
        if !matches!(access, TextureAccess::ShaderRead | TextureAccess::TransferSrc) {
            return Err(self.wrong_access("read()", "ShaderRead or TransferSrc", "write()"));
        }
        self.pass.textures.push(TextureUse { id: texture.id, state: access.state(), usage: access.usage(), write: false });
        Ok(())
    }

    pub fn write_texture(&mut self, texture: GraphTexture, access: TextureAccess) -> GraphResult<()> {
        self.check_texture(texture)?;
        if !matches!(access, TextureAccess::ShaderWrite | TextureAccess::TransferDst) {
            return Err(self.wrong_access("write()", "ShaderWrite or TransferDst", "read()"));
        }
        self.pass.textures.push(TextureUse { id: texture.id, state: access.state(), usage: access.usage(), write: true });
        Ok(())
    }

    pub fn read_buffer(&mut self, buffer: GraphBuffer, access: BufferAccess) -> GraphResult<()> {
        self.check_buffer(buffer)?;
        if access.is_write() {
            return Err(GraphError::Declaration(format!("pass '{}': read() was given a write access; use write()", self.pass.name)));
        }
        self.pass.buffers.push(BufferUse { id: buffer.id, state: access.state(), usage: access.usage(), write: false, host: false });
        Ok(())
    }

    pub fn write_buffer(&mut self, buffer: GraphBuffer, access: BufferAccess) -> GraphResult<()> {
        self.check_buffer(buffer)?;
        if !access.is_write() {
            return Err(GraphError::Declaration(format!("pass '{}': write() was given a read access; use read()", self.pass.name)));
        }
        self.pass.buffers.push(BufferUse {
            id: buffer.id,
            state: access.state(),
            usage: access.usage(),
            write: true,
            host: access == BufferAccess::HostWrite,
        });
        Ok(())
    }

    /// The pass is never culled (readback, debug output).
    pub fn side_effect(&mut self) {
        self.pass.side_effect = true;
    }

    fn wrong_access(&self, call: &str, takes: &str, other: &str) -> GraphError {
        GraphError::Declaration(format!("pass '{}': {call} takes {takes}; use {other} for the other kind", self.pass.name))
    }
}

/// Handed to the `execute` callback of a pass while its commands are recorded.
pub struct PassContext<'a> {
    pub cmd: &'a mut dyn CommandList,
    pub device: &'a dyn Device,
    pub(crate) pass: &'a str,
    pub(crate) textures: &'a [TextureNode],
    pub(crate) buffers: &'a [BufferNode],
    pub(crate) used_textures: &'a [u32],
    pub(crate) used_buffers: &'a [u32],
}

impl PassContext<'_> {
    /// The device texture behind `texture`. The pass must have declared it.
    pub fn texture(&self, texture: GraphTexture) -> GraphResult<TextureHandle> {
        if !self.used_textures.contains(&texture.id) {
            return Err(GraphError::Declaration(format!("pass '{}' used a texture it did not declare", self.pass)));
        }
        Ok(self.textures[texture.id as usize].handle)
    }

    /// The device buffer behind `buffer`. The pass must have declared it.
    pub fn buffer(&self, buffer: GraphBuffer) -> GraphResult<BufferHandle> {
        if !self.used_buffers.contains(&buffer.id) {
            return Err(GraphError::Declaration(format!("pass '{}' used a buffer it did not declare", self.pass)));
        }
        Ok(self.buffers[buffer.id as usize].handle)
    }
}
