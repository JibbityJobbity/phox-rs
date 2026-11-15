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

/// The lists of work a meshlet pass can run, and which cull pass made them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Frustum and cone culling only.
    Plain = 0,
    /// Occlusion culling, first phase: the meshlets that were visible last frame.
    Early = 1,
    /// Occlusion culling, second phase: every meshlet, tested against the depth pyramid.
    Late = 2,
}

/// Mesh shading settings of a frame.
#[derive(Clone, Copy, Debug)]
pub struct MeshletSettings {
    /// `MESHLET_FRUSTUM_CULLING | MESHLET_CONE_CULLING`, or 0.
    pub flags: u32,
    /// Cluster LOD: the error, in pixels, a meshlet may show.
    pub lod_threshold: f32,
    /// Two phases around a depth pyramid: draw what was visible last frame, then test the rest.
    pub occlusion: bool,
}

/// The buffers and counts one frame's forward passes use (made by the renderer each frame).
#[derive(Clone, Debug)]
pub struct ForwardIn {
    pub extent: Extent2D,
    pub clear_color: [f32; 4],
    /// The frame's `FrameUniforms`.
    pub uniforms: BufferHandle,
    /// The frame's `GpuMesh` table.
    pub mesh_table: BufferHandle,
    pub mesh_table_address: u64,
    /// What changed in the scene since last frame (empty: nothing).
    pub instance_updates: BufferHandle,
    pub instance_updates_address: u64,
    pub instance_update_count: u32,
    pub material_updates: BufferHandle,
    pub material_updates_address: u64,
    pub material_update_count: u32,
    /// The scene, kept on the GPU: a `GpuInstance` and a `MaterialData` per slot.
    pub instances: BufferHandle,
    pub instances_address: u64,
    pub materials: BufferHandle,
    pub materials_address: u64,
    /// Occlusion culling: per slot visible last frame, and the meshlet visibility bits.
    pub history: BufferHandle,
    pub history_address: u64,
    pub visibility: BufferHandle,
    pub visibility_address: u64,
    pub instance_count: u32,
    /// Most draws, and most work per bucket, a cull pass can append (both at least 1).
    pub draw_capacity: u32,
    pub work_capacity: u32,
    /// Vertex path: draws take the cull pass's count. Otherwise the command buffer is cleared first and drawn
    /// whole, its unused commands drawing nothing.
    pub draw_count: bool,
    pub index_pool: BufferHandle,
    /// The device's largest task shader dispatch in x.
    pub max_groups_x: u32,
    /// Blended objects, drawn back to front with vertex shaders: their draws, commands and runs.
    pub blend_draws: BufferHandle,
    pub blend_draws_address: u64,
    pub blend_commands: BufferHandle,
    pub blended: Vec<BlendRun>,
    /// Set when the opaque objects are drawn by task and mesh shaders.
    pub meshlets: Option<MeshletSettings>,
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

/// Which instance cull pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CullVariant {
    Meshlets,
    Early,
    Late,
    Vertex,
}

/// Draws the scene's objects, shades them with the PBR material table and draws blended objects last. The
/// GPU does the rest: a scene update pass writes what changed, cull passes pick the visible objects and
/// build the draw lists, and the opaque ones are drawn by indirect calls: indexed draws with vertex shaders,
/// or task and mesh shaders over meshlets (with frustum, cone and occlusion culling). Everything reads its
/// vertices through device addresses.
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
    compute_shaders: Vec<ShaderHandle>,
    scene_update: PipelineHandle,
    cull: [PipelineHandle; 4],
    args: PipelineHandle,
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

fn compute_pipeline(device: &dyn Device, file: &str, bindings: Vec<Binding>, push_size: usize, shaders: &mut Vec<ShaderHandle>) -> Result<PipelineHandle> {
    let shader = shader(device, ShaderStage::Compute, file)?;
    shaders.push(shader);
    Ok(device.create_compute_pipeline(&ComputePipelineDesc {
        shader,
        bindings,
        push_constants: vec![PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: push_size as u32 }],
        debug_name: file.to_string(),
    })?)
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
        let compute = ShaderStages::COMPUTE;
        let frame_set = || Binding::new(0, 0, BindingType::UniformBuffer, compute);
        let cull_size = size_of::<CullPushConstants>();
        self.scene_update = compute_pipeline(device, "scene_update.comp", vec![], size_of::<SceneUpdatePushConstants>(), &mut self.compute_shaders)?;
        self.cull[CullVariant::Vertex as usize] = compute_pipeline(device, "cull_vertex.comp", vec![frame_set()], cull_size, &mut self.compute_shaders)?;
        if options.meshlets {
            self.args = compute_pipeline(device, "draw_args.comp", vec![], size_of::<ArgsPushConstants>(), &mut self.compute_shaders)?;
            self.cull[CullVariant::Meshlets as usize] = compute_pipeline(device, "cull.comp", vec![frame_set()], cull_size, &mut self.compute_shaders)?;
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
                self.cull[CullVariant::Early as usize] = compute_pipeline(device, "cull_early.comp", vec![frame_set()], cull_size, &mut self.compute_shaders)?;
                let pyramid_set = Binding { sampler: nearest_clamp(), ..Binding::new(1, 0, BindingType::SampledTexture, compute) };
                self.cull[CullVariant::Late as usize] = compute_pipeline(device, "cull_late.comp", vec![frame_set(), pyramid_set], cull_size, &mut self.compute_shaders)?;
                self.pyramid.init(device)?;
            }
        }
        Ok(())
    }

    pub fn shutdown(&mut self, device: &dyn Device) {
        let pipelines = self.opaque.iter().chain(&self.blend).chain(self.meshlet.iter().flatten()).chain(&self.cull).chain([&self.scene_update, &self.args]);
        for p in pipelines {
            device.destroy_pipeline(*p);
        }
        for s in [self.vertex, self.fragment, self.mesh].into_iter().chain(self.tasks).chain(self.compute_shaders.iter().copied()) {
            device.destroy_shader(s);
        }
        if self.options.occlusion {
            self.pyramid.shutdown(device);
        }
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

/// The graph resources of the scene and the frame.
#[derive(Clone, Copy)]
struct SceneBuffers {
    uniforms: GraphBuffer,
    mesh_table: GraphBuffer,
    instances: GraphBuffer,
    materials: GraphBuffer,
    /// Occlusion culling only.
    history: Option<GraphBuffer>,
    visibility: Option<GraphBuffer>,
}

/// One cull pass's output: what to draw, the work per bucket, and the counts (and dispatch arguments).
#[derive(Clone, Copy)]
struct DrawList {
    draws: GraphBuffer,
    work: GraphBuffer,
    counters: GraphBuffer,
}

fn create_draw_list(b: &mut GraphBuilder<'_>, name: &str, input: &ForwardIn, meshlets: bool) -> DrawList {
    // A work item is (draw, meshlet group); a draw command is a VkDrawIndexedIndirectCommand.
    let entry = if meshlets { 8 } else { size_of::<DrawCommand>() as u64 };
    let buffer = |size: u64, label: String| BufferDesc { size, usage: BufferUsage::NONE, memory: MemoryKind::GpuOnly, debug_name: label, device_address: true };
    DrawList {
        draws: b.create_buffer(&buffer(u64::from(input.draw_capacity) * size_of::<GpuDraw>() as u64, format!("{name} draws"))),
        work: b.create_buffer(&buffer(2 * u64::from(input.work_capacity) * entry, format!("{name} work"))),
        counters: b.create_buffer(&buffer(size_of::<DrawCounters>() as u64, format!("{name} counters"))),
    }
}

impl ForwardPass {
    /// Writes what changed in the scene into its GPU arrays.
    fn add_scene_update<'g>(&self, b: &mut GraphBuilder<'g>, input: &ForwardIn, scene: SceneBuffers) -> GraphResult<()> {
        if input.instance_update_count == 0 && input.material_update_count == 0 {
            return Ok(());
        }
        let instance_updates = b.import_buffer(input.instance_updates, ResourceState::Undefined, ResourceState::Undefined)?;
        let material_updates = b.import_buffer(input.material_updates, ResourceState::Undefined, ResourceState::Undefined)?;
        let (pipeline, i) = (self.scene_update, input.clone());
        b.add_pass(
            "scene update",
            move |p| {
                p.read_buffer(instance_updates, BufferAccess::ShaderRead)?;
                p.read_buffer(material_updates, BufferAccess::ShaderRead)?;
                p.write_buffer(scene.instances, BufferAccess::ShaderWrite)?;
                p.write_buffer(scene.materials, BufferAccess::ShaderWrite)
            },
            move |c| {
                c.cmd.bind_pipeline(pipeline);
                let push = SceneUpdatePushConstants {
                    instance_updates: split_address(i.instance_updates_address),
                    material_updates: split_address(i.material_updates_address),
                    instances: split_address(i.instances_address),
                    materials: split_address(i.materials_address),
                    instance_update_count: i.instance_update_count,
                    material_update_count: i.material_update_count,
                };
                c.cmd.push_constants(ShaderStages::COMPUTE, 0, bytemuck::bytes_of(&push));
                c.cmd.dispatch(i.instance_update_count.max(i.material_update_count).div_ceil(64), 1, 1);
                Ok(())
            },
        )
    }

    /// Zeroes the counters of `lists` (and, drawing without a count, the vertex path's commands).
    fn add_clear<'g>(&self, b: &mut GraphBuilder<'g>, input: &ForwardIn, lists: &[DrawList], meshlets: bool) -> GraphResult<()> {
        let clear_work = !meshlets && !input.draw_count;
        let lists = lists.to_vec();
        let l = lists.clone();
        b.add_pass(
            "clear draw counters",
            move |p| {
                for list in &l {
                    p.write_buffer(list.counters, BufferAccess::TransferDst)?;
                    if clear_work {
                        p.write_buffer(list.work, BufferAccess::TransferDst)?;
                    }
                }
                Ok(())
            },
            move |c| {
                for list in &lists {
                    let counters = c.buffer(list.counters)?;
                    c.cmd.fill_buffer(counters, 0, size_of::<DrawCounters>() as u64, 0);
                    if clear_work {
                        let work = c.buffer(list.work)?;
                        let size = c.device.describe_buffer(work)?.size;
                        c.cmd.fill_buffer(work, 0, size, 0);
                    }
                }
                Ok(())
            },
        )
    }

    /// The instance cull pass: appends what `variant` keeps to `list`.
    #[allow(clippy::too_many_arguments)]
    fn add_cull<'g>(&self, b: &mut GraphBuilder<'g>, name: &str, variant: CullVariant, input: &ForwardIn, scene: SceneBuffers, list: DrawList, pyramid: Option<GraphTexture>) -> GraphResult<()> {
        let (pipeline, i) = (self.cull[variant as usize], input.clone());
        let frustum = input.meshlets.is_none_or(|m| m.flags & MESHLET_FRUSTUM_CULLING != 0);
        b.add_pass(
            name,
            move |p| {
                p.read_buffer(scene.uniforms, BufferAccess::Uniform)?;
                p.read_buffer(scene.instances, BufferAccess::ShaderRead)?;
                p.read_buffer(scene.mesh_table, BufferAccess::ShaderRead)?;
                p.write_buffer(list.draws, BufferAccess::ShaderWrite)?;
                p.write_buffer(list.work, BufferAccess::ShaderWrite)?;
                p.write_buffer(list.counters, BufferAccess::ShaderWrite)?;
                match (variant, scene.history) {
                    (CullVariant::Early, Some(history)) => p.read_buffer(history, BufferAccess::ShaderRead)?,
                    (CullVariant::Late, Some(history)) => p.write_buffer(history, BufferAccess::ShaderWrite)?,
                    _ => {}
                }
                if let Some(pyramid) = pyramid {
                    p.read_texture(pyramid, TextureAccess::ShaderRead)?;
                }
                Ok(())
            },
            move |c| {
                c.cmd.bind_pipeline(pipeline);
                c.cmd.bind_uniform_buffer(0, 0, c.buffer(scene.uniforms)?);
                if let Some(pyramid) = pyramid {
                    c.cmd.bind_sampled_texture(1, 0, c.texture(pyramid)?);
                }
                let address = |c: &PassContext<'_>, buffer: GraphBuffer| -> GraphResult<[u32; 2]> { Ok(split_address(c.device.buffer_address(c.buffer(buffer)?)?)) };
                let push = CullPushConstants {
                    instances: split_address(i.instances_address),
                    meshes: split_address(i.mesh_table_address),
                    draws: address(c, list.draws)?,
                    work: address(c, list.work)?,
                    counters: address(c, list.counters)?,
                    history: split_address(i.history_address),
                    instance_count: i.instance_count,
                    work_capacity: i.work_capacity,
                    flags: if frustum { CULL_FRUSTUM } else { 0 },
                    padding: 0,
                };
                c.cmd.push_constants(ShaderStages::COMPUTE, 0, bytemuck::bytes_of(&push));
                c.cmd.dispatch(i.instance_count.div_ceil(64).max(1), 1, 1);
                Ok(())
            },
        )
    }

    /// Turns `list`'s work counts into task shader dispatch arguments.
    fn add_args<'g>(&self, b: &mut GraphBuilder<'g>, name: &str, input: &ForwardIn, list: DrawList) -> GraphResult<()> {
        let (pipeline, max_x) = (self.args, input.max_groups_x);
        b.add_pass(
            name,
            move |p| p.write_buffer(list.counters, BufferAccess::ShaderWrite),
            move |c| {
                c.cmd.bind_pipeline(pipeline);
                let counters = c.device.buffer_address(c.buffer(list.counters)?)?;
                let push = ArgsPushConstants { counters: split_address(counters), max_groups_x: max_x, padding: 0 };
                c.cmd.push_constants(ShaderStages::COMPUTE, 0, bytemuck::bytes_of(&push));
                c.cmd.dispatch(1, 1, 1);
                Ok(())
            },
        )
    }

    /// One pass drawing `list`: task and mesh shaders over its work items, or its indexed draws.
    #[allow(clippy::too_many_arguments)]
    fn add_draw<'g>(
        &self,
        b: &mut GraphBuilder<'g>,
        name: &str,
        input: &ForwardIn,
        scene: SceneBuffers,
        list: DrawList,
        phase: Phase,
        target: (GraphTexture, GraphTexture),
        first: bool,
        pyramid: Option<GraphTexture>,
    ) -> GraphResult<()> {
        let (hdr, depth) = target;
        let clear = input.clear_color;
        let (meshlet_settings, i) = (input.meshlets, input.clone());
        let pipelines = if meshlet_settings.is_some() { self.meshlet[phase as usize] } else { self.opaque };
        let meshlets = meshlet_settings.is_some();
        b.add_pass(
            name,
            move |p| {
                if first {
                    p.color_target_with(hdr, true, clear)?;
                    p.depth_target(depth)?;
                } else {
                    p.color_target_load(hdr)?;
                    p.depth_target_with(depth, false, 1.0)?;
                }
                p.read_buffer(scene.uniforms, BufferAccess::Uniform)?;
                p.read_buffer(scene.materials, BufferAccess::ShaderRead)?; // set 0's material table
                p.read_buffer(list.draws, BufferAccess::ShaderRead)?;
                p.read_buffer(list.counters, BufferAccess::Indirect)?;
                // Work items are read by the task shader; draw commands by the draw itself.
                p.read_buffer(list.work, if meshlets { BufferAccess::ShaderRead } else { BufferAccess::Indirect })?;
                if let Some(visibility) = scene.visibility {
                    match phase {
                        Phase::Early => p.read_buffer(visibility, BufferAccess::ShaderRead)?,
                        Phase::Late => p.write_buffer(visibility, BufferAccess::ShaderWrite)?,
                        Phase::Plain => {}
                    }
                }
                if let Some(pyramid) = pyramid {
                    p.read_texture(pyramid, TextureAccess::ShaderRead)?;
                }
                Ok(())
            },
            move |c| {
                let draws = c.device.buffer_address(c.buffer(list.draws)?)?;
                let work = c.buffer(list.work)?;
                let counters = c.buffer(list.counters)?;
                if let Some(m) = meshlet_settings {
                    let (work_address, counters_address) = (c.device.buffer_address(work)?, c.device.buffer_address(counters)?);
                    for bucket in 0..2u32 {
                        bind_frame(c, scene.uniforms, scene.materials, pipelines[bucket as usize])?;
                        if let Some(pyramid) = pyramid {
                            c.cmd.bind_sampled_texture(2, 0, c.texture(pyramid)?);
                        }
                        let push = MeshletDrawPushConstants {
                            draws: split_address(draws),
                            work: split_address(work_address),
                            counters: split_address(counters_address),
                            visibility: split_address(i.visibility_address),
                            bucket,
                            work_capacity: i.work_capacity,
                            flags: m.flags,
                            lod_threshold: m.lod_threshold,
                        };
                        c.cmd.push_constants(ShaderStages::TASK | ShaderStages::MESH, 0, bytemuck::bytes_of(&push));
                        c.cmd.draw_mesh_tasks_indirect(counters, counters_args_offset(bucket), 1, 16);
                    }
                } else {
                    c.cmd.bind_index_buffer(i.index_pool, 0);
                    let stride = size_of::<DrawCommand>() as u32;
                    for bucket in 0..2u32 {
                        bind_frame(c, scene.uniforms, scene.materials, pipelines[bucket as usize])?;
                        let push = VertexDrawPushConstants { draws: split_address(draws) };
                        c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&push));
                        let offset = u64::from(bucket) * u64::from(i.work_capacity) * u64::from(stride);
                        if i.draw_count {
                            c.cmd.draw_indexed_indirect_count(work, offset, counters, 4 + 4 * u64::from(bucket), i.work_capacity, stride);
                        } else {
                            // In calls of at most maxDrawIndirectCount, which may be as low as 2^16 - 1.
                            let limit = c.device.capabilities().max_draw_indirect_count.max(1);
                            let mut at = 0;
                            while at < i.work_capacity {
                                let n = limit.min(i.work_capacity - at);
                                c.cmd.draw_indexed_indirect(work, offset + u64::from(at) * u64::from(stride), n, stride);
                                at += n;
                            }
                        }
                    }
                }
                Ok(())
            },
        )
    }
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
        } else if input.draw_count {
            b.require(Feature::DrawIndirectCount);
        }
        let (w, h) = (input.extent.width, input.extent.height);
        let hdr = b.create_texture(&TextureDesc::new(w, h, HDR_FORMAT, TextureUsage::NONE, "hdr"));
        let depth = b.create_texture(&TextureDesc::new(w, h, DEPTH_FORMAT, TextureUsage::NONE, "depth"));
        let import = |b: &mut GraphBuilder<'g>, buffer: BufferHandle| b.import_buffer(buffer, ResourceState::Undefined, ResourceState::Undefined);
        let occlusion = input.meshlets.is_some_and(|m| m.occlusion);
        let scene = SceneBuffers {
            uniforms: import(b, input.uniforms)?,
            mesh_table: import(b, input.mesh_table)?,
            instances: import(b, input.instances)?,
            materials: import(b, input.materials)?,
            history: if occlusion { Some(import(b, input.history)?) } else { None },
            visibility: if occlusion { Some(import(b, input.visibility)?) } else { None },
        };
        self.add_scene_update(b, input, scene)?;

        match input.meshlets {
            Some(m) if m.occlusion => {
                // Draw what was visible last frame, build a depth pyramid from it, then test everything
                // else against the pyramid and draw what is newly visible.
                let (early, late) = (create_draw_list(b, "early", input, true), create_draw_list(b, "late", input, true));
                self.add_clear(b, input, &[early, late], true)?;
                self.add_cull(b, "cull early", CullVariant::Early, input, scene, early, None)?;
                self.add_args(b, "draw args early", input, early)?;
                self.add_draw(b, "forward early", input, scene, early, Phase::Early, (hdr, depth), true, None)?;
                let pyramid = self.pyramid.add(b, depth, input.extent)?;
                self.add_cull(b, "cull late", CullVariant::Late, input, scene, late, Some(pyramid))?;
                self.add_args(b, "draw args late", input, late)?;
                self.add_draw(b, "forward late", input, scene, late, Phase::Late, (hdr, depth), false, Some(pyramid))?;
            }
            Some(_) => {
                let list = create_draw_list(b, "frame", input, true);
                self.add_clear(b, input, &[list], true)?;
                self.add_cull(b, "cull", CullVariant::Meshlets, input, scene, list, None)?;
                self.add_args(b, "draw args", input, list)?;
                self.add_draw(b, "forward", input, scene, list, Phase::Plain, (hdr, depth), true, None)?;
            }
            None => {
                let list = create_draw_list(b, "frame", input, false);
                self.add_clear(b, input, &[list], false)?;
                self.add_cull(b, "cull", CullVariant::Vertex, input, scene, list, None)?;
                self.add_draw(b, "forward", input, scene, list, Phase::Plain, (hdr, depth), true, None)?;
            }
        }

        if !input.blended.is_empty() {
            // Blended items follow, back to front: depth tested but not written, over the opaque result.
            let blended_input = input.clone();
            let blend_pipelines = self.blend;
            let blend_draws = import(b, input.blend_draws)?;
            let blend_commands = import(b, input.blend_commands)?;
            b.add_pass(
                "forward blend",
                move |p| {
                    p.color_target_load(hdr)?;
                    p.depth_read(depth)?;
                    p.read_buffer(scene.uniforms, BufferAccess::Uniform)?;
                    p.read_buffer(scene.materials, BufferAccess::ShaderRead)?;
                    p.read_buffer(blend_draws, BufferAccess::ShaderRead)?;
                    p.read_buffer(blend_commands, BufferAccess::Indirect)
                },
                move |c| {
                    c.cmd.bind_index_buffer(blended_input.index_pool, 0);
                    for run in &blended_input.blended {
                        bind_frame(c, scene.uniforms, scene.materials, blend_pipelines[usize::from(run.cull_back_faces)])?;
                        let push = VertexDrawPushConstants { draws: split_address(blended_input.blend_draws_address) };
                        c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&push));
                        draw_range(c, blend_commands, run.range)?;
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
