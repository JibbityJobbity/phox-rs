//! The forward technique's passes, each a graph [`Component`]: forward (and blended forward), tonemap and
//! readback. They are written once, against `phox_rhi`, and run on any backend.

use crate::error::Result;
use crate::gpu_types::*;
use phox_core::graph::*;
use phox_core::meshlets::{depth_pyramid_extent, depth_pyramid_levels};
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

/// Mesh shading: the work lists and counters the task shaders read, built by the renderer each frame.
#[derive(Clone, Debug)]
pub struct MeshletIn {
    pub work: BufferHandle,
    pub work_address: u64,
    pub counters: BufferHandle,
    pub counters_address: u64,
    pub visibility: BufferHandle,
    pub visibility_address: u64,
    /// Work items per bucket and list.
    pub work_capacity: u32,
    /// `MESHLET_FRUSTUM_CULLING | MESHLET_CONE_CULLING`.
    pub flags: u32,
    pub lod_threshold: f32,
    /// Two phases around a depth pyramid: draw what was visible last frame, then test the rest.
    pub occlusion: bool,
    /// Work items by list (plain, early, late) and bucket.
    pub counts: [[u32; 2]; 3],
}

/// The lists of work a meshlet pass can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Frustum and cone culling only.
    Plain = 0,
    /// Occlusion culling, first phase: the meshlets that were visible last frame.
    Early = 1,
    /// Occlusion culling, second phase: every meshlet, tested against the depth pyramid.
    Late = 2,
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
    /// Vertex path: opaque and masked items by cull mode, `[both faces, back faces culled]`.
    pub opaque: [CommandRange; 2],
    /// Blended items draw back to front with vertex shaders, whichever path draws the rest.
    pub blended: Vec<BlendRun>,
    /// Set when the opaque items are drawn by task and mesh shaders.
    pub meshlets: Option<MeshletIn>,
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardOut {
    pub hdr: GraphTexture,
}

/// What the forward pass is built for.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardOptions {
    /// Create the task and mesh pipelines (the device has mesh shaders and they are wanted).
    pub meshlets: bool,
    /// Also create the occlusion culling variants.
    pub occlusion: bool,
}

/// Draws the scene's objects, shades them with the PBR material table and draws blended objects last.
/// Opaque objects use indexed indirect draws with vertex shaders, or task and mesh shaders over meshlets
/// (with frustum, cone and occlusion culling); both read their vertices through device addresses.
#[derive(Default)]
pub struct ForwardPass {
    vertex: ShaderHandle,
    fragment: ShaderHandle,
    tasks: [ShaderHandle; 3],
    mesh: ShaderHandle,
    /// Opaque, by `[both faces, back faces culled]`.
    opaque: [PipelineHandle; 2],
    /// Blended, depth tested without writes.
    blend: [PipelineHandle; 2],
    /// Task and mesh pipelines by phase and cull mode.
    meshlet: [[PipelineHandle; 2]; 3],
    pyramid: DepthPyramidPass,
    options: ForwardOptions,
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

/// A nearest sampler that clamps, for the depth pyramid (depth and R32F formats do not filter everywhere).
fn nearest_clamp() -> SamplerDesc {
    SamplerDesc { mag_filter: Filter::Nearest, min_filter: Filter::Nearest, mip_filter: Filter::Nearest, wrap_u: WrapMode::ClampToEdge, wrap_v: WrapMode::ClampToEdge }
}

fn meshlet_pipeline_desc(shaders: [ShaderHandle; 3], cull: CullMode, phase: Phase, name: &str) -> GraphicsPipelineDesc {
    let task_mesh = ShaderStages::TASK | ShaderStages::MESH;
    let mut bindings = vec![
        Binding::new(0, 0, BindingType::UniformBuffer, task_mesh | ShaderStages::FRAGMENT),
        Binding::new(0, 1, BindingType::StorageBuffer, ShaderStages::FRAGMENT),
        Binding { count: 1, ..Binding::new(1, 0, BindingType::BindlessTextures, ShaderStages::FRAGMENT) },
    ];
    if phase == Phase::Late {
        // The depth pyramid the early phase's depth made.
        bindings.push(Binding { sampler: nearest_clamp(), ..Binding::new(2, 0, BindingType::SampledTexture, ShaderStages::TASK) });
    }
    GraphicsPipelineDesc {
        shaders: shaders.to_vec(),
        cull,
        color_formats: vec![HDR_FORMAT],
        depth_format: DEPTH_FORMAT,
        depth_test: true,
        depth_write: true,
        bindings,
        push_constants: vec![PushConstantRange { stages: task_mesh, offset: 0, size: size_of::<MeshletDrawPushConstants>() as u32 }],
        debug_name: name.to_string(),
        ..GraphicsPipelineDesc::default()
    }
}

impl ForwardPass {
    /// Creates the pipelines. Components create their device objects here, never in `setup`.
    pub fn init(&mut self, device: &dyn Device, options: ForwardOptions) -> Result<()> {
        self.options = options;
        self.vertex = shader(device, ShaderStage::Vertex, "forward.vert")?;
        self.fragment = shader(device, ShaderStage::Fragment, "forward.frag")?;
        for (i, cull) in [CullMode::None, CullMode::Back].into_iter().enumerate() {
            self.opaque[i] = device.create_graphics_pipeline(&pipeline_desc(self.vertex, self.fragment, cull, false, "forward"))?;
            self.blend[i] = device.create_graphics_pipeline(&pipeline_desc(self.vertex, self.fragment, cull, true, "forward blend"))?;
        }
        if options.meshlets {
            self.mesh = shader(device, ShaderStage::Mesh, "forward.mesh")?;
            let phases = [(Phase::Plain, "forward.task"), (Phase::Early, "forward_early.task"), (Phase::Late, "forward_late.task")];
            for (phase, file) in phases {
                if phase != Phase::Plain && !options.occlusion {
                    continue;
                }
                let task = shader(device, ShaderStage::Task, file)?;
                self.tasks[phase as usize] = task;
                for (i, cull) in [CullMode::None, CullMode::Back].into_iter().enumerate() {
                    let desc = meshlet_pipeline_desc([task, self.mesh, self.fragment], cull, phase, &format!("forward meshlets {phase:?}"));
                    self.meshlet[phase as usize][i] = device.create_graphics_pipeline(&desc)?;
                }
            }
            if options.occlusion {
                self.pyramid.init(device)?;
            }
        }
        Ok(())
    }

    pub fn shutdown(&mut self, device: &dyn Device) {
        for p in self.opaque.iter().chain(&self.blend).chain(self.meshlet.iter().flatten()) {
            device.destroy_pipeline(*p);
        }
        for s in [self.vertex, self.fragment, self.mesh].into_iter().chain(self.tasks) {
            device.destroy_shader(s);
        }
        self.pyramid.shutdown(device);
        *self = Self::default();
    }
}

/// Binds what every forward draw shares: the frame set and the texture table.
fn bind_frame(c: &mut PassContext<'_>, uniforms: GraphBuffer, materials: GraphBuffer, pipeline: PipelineHandle) -> GraphResult<()> {
    c.cmd.bind_pipeline(pipeline);
    c.cmd.bind_uniform_buffer(0, 0, c.buffer(uniforms)?);
    c.cmd.bind_storage_buffer(0, 1, c.buffer(materials)?);
    c.cmd.bind_bindless_table(1);
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

/// The graph resources a forward pass declares.
#[derive(Clone, Copy)]
struct FrameBuffers {
    uniforms: GraphBuffer,
    materials: GraphBuffer,
    draws: GraphBuffer,
    commands: GraphBuffer,
}

fn declare_frame(p: &mut PassBuilder<'_>, f: FrameBuffers) -> GraphResult<()> {
    p.read_buffer(f.uniforms, BufferAccess::Uniform)?;
    p.read_buffer(f.materials, BufferAccess::ShaderRead)?;
    p.read_buffer(f.draws, BufferAccess::ShaderRead)?;
    p.read_buffer(f.commands, BufferAccess::Indirect)
}

/// The meshlet resources a pass declares: its counters (also the indirect arguments), its work, the
/// visibility bits (`write`: the late phase records them) and, late, the depth pyramid.
#[derive(Clone, Copy)]
struct MeshletBuffers {
    work: GraphBuffer,
    counters: GraphBuffer,
    /// Only with occlusion culling.
    visibility: Option<GraphBuffer>,
}

fn declare_meshlets(p: &mut PassBuilder<'_>, m: MeshletBuffers, phase: Phase, pyramid: Option<GraphTexture>) -> GraphResult<()> {
    p.read_buffer(m.counters, BufferAccess::Indirect)?;
    p.read_buffer(m.work, BufferAccess::ShaderRead)?;
    if let Some(visibility) = m.visibility {
        match phase {
            Phase::Plain => {}
            Phase::Early => p.read_buffer(visibility, BufferAccess::ShaderRead)?,
            Phase::Late => p.write_buffer(visibility, BufferAccess::ShaderWrite)?,
        }
    }
    if let Some(pyramid) = pyramid {
        p.read_texture(pyramid, TextureAccess::ShaderRead)?;
    }
    Ok(())
}

/// Draws one phase's task shader dispatches: a pipeline per cull bucket, run over that bucket's work items.
#[allow(clippy::too_many_arguments)]
fn draw_meshlets(
    c: &mut PassContext<'_>,
    input: &ForwardIn,
    m: &MeshletIn,
    phase: Phase,
    pipelines: [PipelineHandle; 2],
    f: FrameBuffers,
    counters: GraphBuffer,
    pyramid: Option<GraphTexture>,
) -> GraphResult<()> {
    let counters_buffer = c.buffer(counters)?;
    let phase_index = phase as usize;
    for bucket in 0..2u32 {
        if m.counts[phase_index][bucket as usize] == 0 {
            continue;
        }
        bind_frame(c, f.uniforms, f.materials, pipelines[bucket as usize])?;
        if let Some(pyramid) = pyramid {
            c.cmd.bind_sampled_texture(2, 0, c.texture(pyramid)?);
        }
        let work_bytes = u64::from(m.work_capacity) * 2 * 8; // items of two words, two buckets
        let push = MeshletDrawPushConstants {
            draws: split_address(input.draws_address),
            work: split_address(m.work_address + phase_index as u64 * work_bytes),
            counters: split_address(m.counters_address + (phase_index * size_of::<DrawCounters>()) as u64),
            visibility: split_address(m.visibility_address),
            bucket,
            work_capacity: m.work_capacity,
            flags: m.flags,
            lod_threshold: m.lod_threshold,
        };
        c.cmd.push_constants(ShaderStages::TASK | ShaderStages::MESH, 0, bytemuck::bytes_of(&push));
        let offset = (phase_index * size_of::<DrawCounters>()) as u64 + counters_args_offset(bucket);
        c.cmd.draw_mesh_tasks_indirect(counters_buffer, offset, 1, 16);
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
        if input.meshlets.is_some() {
            b.require(Feature::MeshShader);
        }
        let (w, h) = (input.extent.width, input.extent.height);
        let hdr = b.create_texture(&TextureDesc::new(w, h, HDR_FORMAT, TextureUsage::NONE, "hdr"));
        let depth = b.create_texture(&TextureDesc::new(w, h, DEPTH_FORMAT, TextureUsage::NONE, "depth"));
        let frame = FrameBuffers {
            uniforms: b.import_buffer(input.uniforms, ResourceState::Undefined, ResourceState::Undefined)?,
            materials: b.import_buffer(input.materials, ResourceState::Undefined, ResourceState::Undefined)?,
            draws: b.import_buffer(input.draws, ResourceState::Undefined, ResourceState::Undefined)?,
            commands: b.import_buffer(input.commands, ResourceState::Undefined, ResourceState::Undefined)?,
        };
        let clear = input.clear_color;

        if let Some(m) = &input.meshlets {
            let buffers = MeshletBuffers {
                work: b.import_buffer(m.work, ResourceState::Undefined, ResourceState::Undefined)?,
                counters: b.import_buffer(m.counters, ResourceState::Undefined, ResourceState::Undefined)?,
                visibility: if m.occlusion { Some(b.import_buffer(m.visibility, ResourceState::Undefined, ResourceState::Undefined)?) } else { None },
            };
            let pipelines = self.meshlet;
            let early_input = input.clone();
            if m.occlusion {
                // Draw what was visible last frame, build a depth pyramid from it, then test everything
                // else against the pyramid and draw what is newly visible.
                b.add_pass(
                    "forward early",
                    move |p| {
                        p.color_target_with(hdr, true, clear)?;
                        p.depth_target(depth)?;
                        declare_frame(p, frame)?;
                        declare_meshlets(p, buffers, Phase::Early, None)
                    },
                    move |c| {
                        let m = early_input.meshlets.as_ref().expect("meshlet input");
                        draw_meshlets(c, &early_input, m, Phase::Early, pipelines[Phase::Early as usize], frame, buffers.counters, None)
                    },
                )?;
                let pyramid = self.pyramid.add(b, depth, input.extent)?;
                let late_input = input.clone();
                b.add_pass(
                    "forward late",
                    move |p| {
                        p.color_target_load(hdr)?;
                        p.depth_target_with(depth, false, 1.0)?;
                        declare_frame(p, frame)?;
                        declare_meshlets(p, buffers, Phase::Late, Some(pyramid))
                    },
                    move |c| {
                        let m = late_input.meshlets.as_ref().expect("meshlet input");
                        draw_meshlets(c, &late_input, m, Phase::Late, pipelines[Phase::Late as usize], frame, buffers.counters, Some(pyramid))
                    },
                )?;
            } else {
                b.add_pass(
                    "forward",
                    move |p| {
                        p.color_target_with(hdr, true, clear)?;
                        p.depth_target(depth)?;
                        declare_frame(p, frame)?;
                        declare_meshlets(p, buffers, Phase::Plain, None)
                    },
                    move |c| {
                        let m = early_input.meshlets.as_ref().expect("meshlet input");
                        draw_meshlets(c, &early_input, m, Phase::Plain, pipelines[Phase::Plain as usize], frame, buffers.counters, None)
                    },
                )?;
            }
        } else {
            let pipelines = self.opaque;
            let opaque_input = input.clone();
            b.add_pass(
                "forward",
                move |p| {
                    p.color_target_with(hdr, true, clear)?;
                    p.depth_target(depth)?;
                    declare_frame(p, frame)
                },
                move |c| {
                    c.cmd.bind_index_buffer(opaque_input.index_pool, 0);
                    for (bucket, range) in opaque_input.opaque.iter().enumerate() {
                        if range.count > 0 {
                            bind_frame(c, frame.uniforms, frame.materials, pipelines[bucket])?;
                            let push = VertexDrawPushConstants { draws: split_address(opaque_input.draws_address) };
                            c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&push));
                            draw_range(c, frame.commands, *range)?;
                        }
                    }
                    Ok(())
                },
            )?;
        }

        if !input.blended.is_empty() {
            // Blended items follow, back to front: depth tested but not written, over the opaque result.
            let blended_input = input.clone();
            let blend_pipelines = self.blend;
            b.add_pass(
                "forward blend",
                move |p| {
                    p.color_target_load(hdr)?;
                    p.depth_read(depth)?;
                    declare_frame(p, frame)
                },
                move |c| {
                    c.cmd.bind_index_buffer(blended_input.index_pool, 0);
                    for run in &blended_input.blended {
                        bind_frame(c, frame.uniforms, frame.materials, blend_pipelines[usize::from(run.cull_back_faces)])?;
                        let push = VertexDrawPushConstants { draws: split_address(blended_input.draws_address) };
                        c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&push));
                        draw_range(c, frame.commands, run.range)?;
                    }
                    Ok(())
                },
            )?;
        }
        Ok(ForwardOut { hdr })
    }
}

/// Builds a hierarchical depth buffer from the depth the early phase drew: mip 0 is the largest
/// power-of-two size not above the target's, each texel the farthest depth of every depth pixel it overlaps;
/// each further mip keeps the farthest of 2 x 2, down to 1x1.
#[derive(Default)]
pub struct DepthPyramidPass {
    shader: ShaderHandle,
    pipeline: PipelineHandle,
}

impl DepthPyramidPass {
    pub fn init(&mut self, device: &dyn Device) -> Result<()> {
        self.shader = shader(device, ShaderStage::Compute, "depthpyramid.comp")?;
        let compute = ShaderStages::COMPUTE;
        self.pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: self.shader,
            bindings: vec![
                Binding { sampler: nearest_clamp(), ..Binding::new(0, 0, BindingType::SampledTexture, compute) },
                Binding::new(0, 1, BindingType::StorageTexture, compute),
                Binding::new(0, 2, BindingType::StorageTexture, compute),
            ],
            push_constants: vec![PushConstantRange { stages: compute, offset: 0, size: 4 }],
            debug_name: "depth pyramid".into(),
        })?;
        Ok(())
    }

    pub fn shutdown(&mut self, device: &dyn Device) {
        device.destroy_pipeline(self.pipeline);
        device.destroy_shader(self.shader);
    }

    /// Adds the pass: reads `depth` (D32Float) and returns the pyramid (R32Float, all mips).
    pub fn add<'g>(&mut self, b: &mut GraphBuilder<'g>, depth: GraphTexture, extent: Extent2D) -> GraphResult<GraphTexture> {
        let size = depth_pyramid_extent(extent);
        let levels = depth_pyramid_levels(size);
        let mut desc = TextureDesc::new(size.width, size.height, Format::R32Float, TextureUsage::NONE, "depth pyramid");
        desc.mips = levels;
        let pyramid = b.create_texture(&desc);
        let pipeline = self.pipeline;
        b.add_pass(
            "depth pyramid",
            move |p| {
                p.read_texture(depth, TextureAccess::ShaderRead)?;
                p.write_texture(pyramid, TextureAccess::ShaderWrite)
            },
            move |c| {
                let (depth_texture, pyramid_texture) = (c.texture(depth)?, c.texture(pyramid)?);
                c.cmd.bind_pipeline(pipeline);
                for level in 0..levels {
                    if level > 0 {
                        // Each mip reads the one before; the pyramid stays in GENERAL throughout.
                        c.cmd.barrier(&[TextureBarrier { texture: pyramid_texture, from: ResourceState::ShaderWrite, to: ResourceState::ShaderWrite }], &[]);
                    }
                    // Mip 0 reads depth only; its source binding just needs a valid view of another mip.
                    let source = if level > 0 { level - 1 } else { 1.min(levels - 1) };
                    c.cmd.bind_sampled_texture(0, 0, depth_texture);
                    c.cmd.bind_storage_texture(0, 1, pyramid_texture, source);
                    c.cmd.bind_storage_texture(0, 2, pyramid_texture, level);
                    c.cmd.push_constants(ShaderStages::COMPUTE, 0, &level.to_ne_bytes());
                    let (width, height) = ((size.width >> level).max(1), (size.height >> level).max(1));
                    c.cmd.dispatch(width.div_ceil(8), height.div_ceil(8), 1);
                }
                Ok(())
            },
        )?;
        Ok(pyramid)
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
