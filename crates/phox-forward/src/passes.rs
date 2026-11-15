//! The forward technique's passes, each a graph [`Component`]: forward (and blended forward), tonemap and
//! readback. They are written once, against `phox_rhi`, and run on any backend.

use crate::error::Result;
use crate::gpu_types::*;
use phox_core::graph::*;
use phox_core::resources::Extent2D;
use phox_core::scene::{ToneMapOperator, ToneMapping};
use phox_core::shader_interface::TonemapPushConstants;
use phox_rhi::*;

/// The HDR scene colour; the tonemap pass maps it to the display format.
pub const HDR_FORMAT: Format = Format::RGBA16Float;
pub const DEPTH_FORMAT: Format = Format::D32Float;
/// Headless frames are tonemapped into this and read back.
pub const HEADLESS_FORMAT: Format = Format::RGBA8Unorm;

fn shader(device: &dyn Device, stage: ShaderStage, name: &str) -> Result<ShaderHandle> {
    let code = crate::shaders::spirv(name).ok_or_else(|| crate::error::ForwardError::Other(format!("shader {name} is not embedded")))?;
    Ok(device.create_shader(&ShaderDesc::new(stage, code))?)
}

/// A contiguous run of indirect commands drawn with one pipeline.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandRange {
    pub first: u32,
    pub count: u32,
}

/// A blended run: consecutive transparent items that share a cull mode, drawn in this order.
#[derive(Clone, Copy, Debug)]
pub struct BlendRun {
    pub cull_back_faces: bool,
    pub range: CommandRange,
}

/// The buffers and draw ranges one frame's forward passes use (made by the renderer each frame).
#[derive(Clone, Debug)]
pub struct ForwardIn {
    pub extent: Extent2D,
    pub clear_color: [f32; 4],
    pub uniforms: BufferHandle,
    pub materials: BufferHandle,
    pub draws: BufferHandle,
    pub draws_address: u64,
    pub commands: BufferHandle,
    pub index_pool: BufferHandle,
    /// Opaque and masked items by cull mode: `[both faces, back faces culled]`.
    pub opaque: [CommandRange; 2],
    pub blended: Vec<BlendRun>,
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardOut {
    pub hdr: GraphTexture,
}

/// Draws the scene's objects with indexed indirect draws, vertex shaders pulling their vertices through
/// device addresses, and shades them with the PBR material table.
#[derive(Default)]
pub struct ForwardPass {
    vertex: ShaderHandle,
    fragment: ShaderHandle,
    /// Opaque, by `[both faces, back faces culled]`.
    opaque: [PipelineHandle; 2],
    /// Blended, depth tested without writes.
    blend: [PipelineHandle; 2],
}

fn pipeline_desc(vertex: ShaderHandle, fragment: ShaderHandle, cull: CullMode, blend: bool, name: &str) -> GraphicsPipelineDesc {
    GraphicsPipelineDesc {
        shaders: vec![vertex, fragment],
        cull,
        color_formats: vec![HDR_FORMAT],
        depth_format: DEPTH_FORMAT,
        depth_test: true,
        depth_write: !blend,
        blend: blend.then_some(BlendDesc { enabled: true }),
        bindings: vec![
            Binding::new(0, 0, BindingType::UniformBuffer, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::new(0, 1, BindingType::StorageBuffer, ShaderStages::FRAGMENT),
            Binding { count: 1, ..Binding::new(1, 0, BindingType::BindlessTextures, ShaderStages::FRAGMENT) },
        ],
        push_constants: vec![PushConstantRange { stages: ShaderStages::VERTEX, offset: 0, size: size_of::<VertexDrawPushConstants>() as u32 }],
        debug_name: name.to_string(),
        ..GraphicsPipelineDesc::default()
    }
}

impl ForwardPass {
    /// Creates the pipelines. Components create their device objects here, never in `setup`.
    pub fn init(&mut self, device: &dyn Device) -> Result<()> {
        self.vertex = shader(device, ShaderStage::Vertex, "forward.vert")?;
        self.fragment = shader(device, ShaderStage::Fragment, "forward.frag")?;
        for (i, cull) in [CullMode::None, CullMode::Back].into_iter().enumerate() {
            self.opaque[i] = device.create_graphics_pipeline(&pipeline_desc(self.vertex, self.fragment, cull, false, "forward"))?;
            self.blend[i] = device.create_graphics_pipeline(&pipeline_desc(self.vertex, self.fragment, cull, true, "forward blend"))?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self, device: &dyn Device) {
        for p in self.opaque.iter().chain(&self.blend) {
            device.destroy_pipeline(*p);
        }
        device.destroy_shader(self.vertex);
        device.destroy_shader(self.fragment);
        *self = Self::default();
    }
}

/// Binds what every forward draw shares: the frame set, the texture table and the draw list's address.
fn bind_frame(c: &mut PassContext<'_>, input: &ForwardIn, uniforms: GraphBuffer, materials: GraphBuffer, pipeline: PipelineHandle) -> GraphResult<()> {
    c.cmd.bind_pipeline(pipeline);
    c.cmd.bind_uniform_buffer(0, 0, c.buffer(uniforms)?);
    c.cmd.bind_storage_buffer(0, 1, c.buffer(materials)?);
    c.cmd.bind_bindless_table(1);
    let push = VertexDrawPushConstants { draws: split_address(input.draws_address) };
    c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&push));
    Ok(())
}

fn draw_range(c: &mut PassContext<'_>, commands: GraphBuffer, range: CommandRange) -> GraphResult<()> {
    let buffer = c.buffer(commands)?;
    let limit = c.device.capabilities().max_draw_indirect_count.max(1);
    let (mut first, mut left) = (range.first, range.count);
    while left > 0 {
        let n = left.min(limit);
        c.cmd.draw_indexed_indirect(buffer, u64::from(first) * size_of::<DrawCommand>() as u64, n, size_of::<DrawCommand>() as u32);
        first += n;
        left -= n;
    }
    Ok(())
}

impl Component for ForwardPass {
    type In = ForwardIn;
    type Out = ForwardOut;

    fn name(&self) -> &str {
        "forward"
    }

    fn setup<'g>(&mut self, b: &mut GraphBuilder<'g>, input: &ForwardIn) -> GraphResult<ForwardOut> {
        b.require(Feature::BindlessDescriptors);
        let (w, h) = (input.extent.width, input.extent.height);
        let hdr = b.create_texture(&TextureDesc::new(w, h, HDR_FORMAT, TextureUsage::NONE, "hdr"));
        let depth = b.create_texture(&TextureDesc::new(w, h, DEPTH_FORMAT, TextureUsage::NONE, "depth"));
        let uniforms = b.import_buffer(input.uniforms, ResourceState::Undefined, ResourceState::Undefined)?;
        let materials = b.import_buffer(input.materials, ResourceState::Undefined, ResourceState::Undefined)?;
        let draws = b.import_buffer(input.draws, ResourceState::Undefined, ResourceState::Undefined)?;
        let commands = b.import_buffer(input.commands, ResourceState::Undefined, ResourceState::Undefined)?;

        let (clear, pipelines, blend_pipelines) = (input.clear_color, self.opaque, self.blend);
        let opaque_input = input.clone();
        b.add_pass(
            "forward",
            move |p| {
                p.color_target_with(hdr, true, clear)?;
                p.depth_target(depth)?;
                p.read_buffer(uniforms, BufferAccess::Uniform)?;
                p.read_buffer(materials, BufferAccess::ShaderRead)?;
                p.read_buffer(draws, BufferAccess::ShaderRead)?;
                p.read_buffer(commands, BufferAccess::Indirect)
            },
            move |c| {
                c.cmd.bind_index_buffer(opaque_input.index_pool, 0);
                for (bucket, range) in opaque_input.opaque.iter().enumerate() {
                    if range.count > 0 {
                        bind_frame(c, &opaque_input, uniforms, materials, pipelines[bucket])?;
                        draw_range(c, commands, *range)?;
                    }
                }
                Ok(())
            },
        )?;

        if !input.blended.is_empty() {
            // Blended items follow, back to front: depth tested but not written, over the opaque result.
            let blended_input = input.clone();
            b.add_pass(
                "forward blend",
                move |p| {
                    p.color_target_load(hdr)?;
                    p.depth_read(depth)?;
                    p.read_buffer(uniforms, BufferAccess::Uniform)?;
                    p.read_buffer(materials, BufferAccess::ShaderRead)?;
                    p.read_buffer(draws, BufferAccess::ShaderRead)?;
                    p.read_buffer(commands, BufferAccess::Indirect)
                },
                move |c| {
                    c.cmd.bind_index_buffer(blended_input.index_pool, 0);
                    for run in &blended_input.blended {
                        bind_frame(c, &blended_input, uniforms, materials, blend_pipelines[usize::from(run.cull_back_faces)])?;
                        draw_range(c, commands, run.range)?;
                    }
                    Ok(())
                },
            )?;
        }
        Ok(ForwardOut { hdr })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TonemapIn {
    pub hdr: GraphTexture,
    pub extent: Extent2D,
    pub mapping: ToneMapping,
    pub target_format: Format,
}

#[derive(Clone, Copy, Debug)]
pub struct TonemapOut {
    pub display: GraphTexture,
}

/// A full-screen triangle that reads the HDR image with `texelFetch`, maps it to the display range and
/// encodes it with the sRGB transfer function (unless the target's format does).
#[derive(Default)]
pub struct TonemapPass {
    vertex: ShaderHandle,
    fragment: ShaderHandle,
    pipelines: Vec<(Format, PipelineHandle)>,
}

impl TonemapPass {
    pub fn init(&mut self, device: &dyn Device) -> Result<()> {
        self.vertex = shader(device, ShaderStage::Vertex, "fullscreen.vert")?;
        self.fragment = shader(device, ShaderStage::Fragment, "tonemap.frag")?;
        Ok(())
    }

    /// The pipeline writing `format`, made on first use.
    pub fn pipeline(&mut self, device: &dyn Device, format: Format) -> Result<PipelineHandle> {
        if let Some((_, p)) = self.pipelines.iter().find(|(f, _)| *f == format) {
            return Ok(*p);
        }
        let desc = GraphicsPipelineDesc {
            shaders: vec![self.vertex, self.fragment],
            color_formats: vec![format],
            bindings: vec![Binding {
                sampler: SamplerDesc { wrap_u: WrapMode::ClampToEdge, wrap_v: WrapMode::ClampToEdge, ..SamplerDesc::default() },
                ..Binding::new(0, 0, BindingType::SampledTexture, ShaderStages::FRAGMENT)
            }],
            push_constants: vec![PushConstantRange { stages: ShaderStages::FRAGMENT, offset: 0, size: size_of::<TonemapPushConstants>() as u32 }],
            debug_name: "tonemap".into(),
            ..GraphicsPipelineDesc::default()
        };
        let pipeline = device.create_graphics_pipeline(&desc)?;
        self.pipelines.push((format, pipeline));
        Ok(pipeline)
    }

    pub fn shutdown(&mut self, device: &dyn Device) {
        for (_, p) in self.pipelines.drain(..) {
            device.destroy_pipeline(p);
        }
        device.destroy_shader(self.vertex);
        device.destroy_shader(self.fragment);
    }

    /// Adds the pass to `b`, which must be for the device `init` ran on.
    pub fn add<'g>(&mut self, b: &mut GraphBuilder<'g>, device: &dyn Device, input: &TonemapIn) -> GraphResult<TonemapOut> {
        let pipeline = self.pipeline(device, input.target_format)?;
        let display = b.create_texture(&TextureDesc::new(input.extent.width, input.extent.height, input.target_format, TextureUsage::NONE, "display"));
        let hdr = input.hdr;
        let constants = TonemapPushConstants {
            exposure: input.mapping.exposure,
            op: match input.mapping.op {
                ToneMapOperator::Clamp => 0,
                ToneMapOperator::Reinhard => 1,
                ToneMapOperator::Aces => 2,
            },
            // An sRGB format encodes on write; anything else gets the transfer function from the shader.
            encode_srgb: u32::from(!matches!(input.target_format, Format::RGBA8Srgb | Format::BGRA8Srgb)),
            padding: 0,
        };
        b.add_pass(
            "tonemap",
            move |p| {
                p.read_texture(hdr, TextureAccess::ShaderRead)?;
                p.color_target(display)
            },
            move |c| {
                c.cmd.bind_pipeline(pipeline);
                c.cmd.bind_sampled_texture(0, 0, c.texture(hdr)?);
                c.cmd.push_constants(ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&constants));
                c.cmd.draw(3, 1, 0); // one oversized triangle, no vertex buffer
                Ok(())
            },
        )?;
        Ok(TonemapOut { display })
    }
}

impl From<crate::error::ForwardError> for GraphError {
    fn from(e: crate::error::ForwardError) -> Self {
        match e {
            crate::error::ForwardError::Graph(g) => g,
            crate::error::ForwardError::Device(d) => GraphError::Device(d),
            other => GraphError::Declaration(other.to_string()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReadbackIn {
    pub source: GraphTexture,
    pub buffer: BufferHandle,
}

/// Copies an 8-bit image into a host-readable buffer. The buffer ends in `HostRead`.
#[derive(Default)]
pub struct ReadbackPass;

impl Component for ReadbackPass {
    type In = ReadbackIn;
    type Out = ();

    fn name(&self) -> &str {
        "readback"
    }

    fn setup<'g>(&mut self, b: &mut GraphBuilder<'g>, input: &ReadbackIn) -> GraphResult<()> {
        let buffer = b.import_buffer(input.buffer, ResourceState::Undefined, ResourceState::HostRead)?;
        let source = input.source;
        b.add_pass(
            "readback",
            move |p| {
                p.read_texture(source, TextureAccess::TransferSrc)?;
                p.write_buffer(buffer, BufferAccess::TransferDst)
            },
            move |c| {
                let (src, dst) = (c.texture(source)?, c.buffer(buffer)?);
                c.cmd.copy_texture_to_buffer(src, dst);
                Ok(())
            },
        )
    }
}
