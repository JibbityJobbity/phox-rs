//! The headless renderer: a scene in, an image out, through the render graph on any [`Device`].

use crate::error::Result;
use crate::frame::FrameSlot;
use crate::gpu_types::*;
use crate::mesh_cache::MeshCache;
use crate::passes::*;
use crate::technique::{ForwardTechnique, FrameInputs};
use crate::texture_cache::TextureCache;
use crate::gpu_scene::{GpuScene, SceneFrame, StateBuffer};
use phox_core::cpu_image::CpuImage;
use phox_core::graph::*;
use phox_core::render_view::{RenderView, extract_camera};
use phox_core::resources::Extent2D;
use phox_core::scene::Scene;
use phox_core::shader_interface::FrameUniforms;
use phox_rhi::*;
use std::sync::Arc;

/// Frames in flight: the CPU records the next while the GPU draws the last.
pub const FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone, Debug)]
pub struct RendererConfig {
    /// Size of headless frames.
    pub extent: Extent2D,
    /// Keep each frame's render graph as Graphviz DOT ([`ForwardRenderer::frame_graph_dot`]).
    pub capture_graph: bool,
    /// Write GPU timestamps around each pass when the device can ([`ForwardRenderer::gpu_timings`]).
    pub timestamps: bool,
    /// Draw with task and mesh shaders where the device has them, as meshlets the task shader culls.
    /// Otherwise, or when off, meshes are drawn with vertex shaders.
    pub mesh_shaders: bool,
    /// With mesh shaders: cull meshlets outside the view frustum and, for single-sided materials, those
    /// facing away from the camera. Off only to measure what culling gains.
    pub meshlet_culling: bool,
    /// With meshlet culling: also cull meshlets hidden behind others, drawing in two phases around a depth
    /// pyramid.
    pub occlusion_culling: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            extent: Extent2D::new(640, 480),
            capture_graph: false,
            timestamps: true,
            mesh_shaders: true,
            meshlet_culling: true,
            occlusion_culling: true,
        }
    }
}

/// GPU time of one render graph pass.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTiming {
    pub pass: String,
    pub milliseconds: f64,
}

pub struct ForwardRenderer {
    device: Arc<dyn Device>,
    config: RendererConfig,
    graph: RenderGraph,
    technique: ForwardTechnique,
    frames: Vec<FrameSlot>,
    meshes: MeshCache,
    textures: TextureCache,
    scene: GpuScene,
    meshlets: bool,
    occlusion: bool,
    meshlet_culling: bool,
    readback: Option<(BufferHandle, Extent2D)>,
    frame_count: u64,
    last_frame: CpuImage,
    timings: Vec<PassTiming>,
    dot: String,
    cpu_milliseconds: f64,
}

fn byte_len<T>(items: &[T]) -> u64 {
    std::mem::size_of_val(items) as u64
}

impl ForwardRenderer {
    pub fn new(device: Arc<dyn Device>, config: RendererConfig) -> Result<Self> {
        let meshlets = config.mesh_shaders && device.capabilities().has(Feature::MeshShader);
        // Occlusion culling needs meshlet culling (and the D32 depth buffer every device has).
        let occlusion = meshlets && config.meshlet_culling && config.occlusion_culling;
        let mut technique = ForwardTechnique::new(device.clone(), ForwardOptions { meshlets, occlusion });
        technique.init(device.as_ref())?;
        let frames = (0..FRAMES_IN_FLIGHT).map(|_| FrameSlot::new(&device)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            graph: RenderGraph::new(device.clone()),
            meshes: MeshCache::new(device.clone(), meshlets)?,
            scene: GpuScene::new(device.clone(), meshlets, occlusion),
            meshlets,
            occlusion,
            meshlet_culling: config.meshlet_culling,
            textures: TextureCache::new(device.clone()),
            technique,
            frames,
            readback: None,
            frame_count: 0,
            last_frame: CpuImage::default(),
            timings: Vec::new(),
            dot: String::new(),
            cpu_milliseconds: 0.0,
            config,
            device,
        })
    }

    pub fn device(&self) -> &Arc<dyn Device> {
        &self.device
    }

    /// Opaque meshes are drawn as meshlets by task and mesh shaders.
    pub fn uses_mesh_shaders(&self) -> bool {
        self.meshlets
    }

    /// Meshlets hidden behind others are culled too, in two phases around a depth pyramid.
    pub fn uses_occlusion_culling(&self) -> bool {
        self.occlusion
    }

    /// The most recent headless frame.
    pub fn last_frame(&self) -> &CpuImage {
        &self.last_frame
    }

    /// GPU time per pass of the most recent frame; empty without timestamps.
    pub fn gpu_timings(&self) -> &[PassTiming] {
        &self.timings
    }

    /// CPU time the most recent frame took, from the end of the wait for its frame slot to the submit.
    pub fn cpu_milliseconds(&self) -> f64 {
        self.cpu_milliseconds
    }

    /// The most recent frame's compiled render graph in Graphviz DOT; empty unless captured.
    pub fn frame_graph_dot(&self) -> &str {
        &self.dot
    }

    fn readback_buffer(&mut self, extent: Extent2D) -> Result<BufferHandle> {
        if let Some((buffer, e)) = self.readback {
            if e == extent {
                return Ok(buffer);
            }
            self.device.destroy_buffer(buffer);
        }
        let size = u64::from(extent.width) * u64::from(extent.height) * 4;
        let buffer = self.device.create_buffer(&BufferDesc::new(size, BufferUsage::TRANSFER_DST, MemoryKind::Readback, "readback"))?;
        self.readback = Some((buffer, extent));
        Ok(buffer)
    }

    /// Renders `scene` at the configured size and reads the image back: [`ForwardRenderer::last_frame`].
    pub fn render(&mut self, scene: &Scene) -> Result<()> {
        let extent = self.config.extent;
        let slot_index = (self.frame_count % FRAMES_IN_FLIGHT as u64) as usize;
        // The frame slot's buffers are rewritten, so the frame that last used them must be done.
        if let Some(ticket) = self.frames[slot_index].ticket.take() {
            self.device.wait(ticket)?;
        }

        let cpu_start = std::time::Instant::now();
        self.meshes.begin_frame();
        self.textures.begin_frame();
        let view = extract_camera(scene, extent);
        // What changed since last frame goes to the GPU; the meshes and textures in use are uploaded.
        let frame = self.scene.update(scene, &view, &mut self.meshes, &mut self.textures)?;

        let readback = self.readback_buffer(extent)?;
        let inputs = self.upload_frame(slot_index, &view, &frame, extent, readback)?;

        self.technique.set_frame(inputs);
        let mut builder = self.graph.begin();
        let timestamps = self.config.timestamps && self.device.capabilities().has(Feature::Timestamps);
        builder.set_timestamps(timestamps);
        self.technique.build(&mut builder)?;
        builder.compile()?;
        if self.config.capture_graph {
            self.dot = builder.to_dot()?;
        }
        let ticket = builder.execute()?;
        drop(builder);
        self.cpu_milliseconds = cpu_start.elapsed().as_secs_f64() * 1000.0;
        self.frames[slot_index].ticket = Some(ticket);
        self.frame_count += 1;

        // Headless: the frame is finished when render() returns.
        self.device.wait(ticket)?;
        let mut rgba = vec![0u8; extent.width as usize * extent.height as usize * 4];
        self.device.read_buffer(readback, 0, &mut rgba)?;
        self.last_frame = CpuImage { width: extent.width, height: extent.height, rgba };
        self.timings = if timestamps { pass_timings(&self.device.read_timestamps(ticket)?) } else { Vec::new() };
        Ok(())
    }

    /// Fills frame slot `slot`'s buffers for `view` and `frame`: the uniforms, the mesh table, what changed
    /// in the scene and the blended draws.
    fn upload_frame(&mut self, slot: usize, view: &RenderView<'_>, frame: &SceneFrame, extent: Extent2D, readback: BufferHandle) -> Result<FrameInputs> {
        let (w, h) = (extent.width as f32, extent.height as f32);
        let uniforms = FrameUniforms {
            view: view.view,
            proj: view.proj,
            view_proj: view.view_proj,
            camera_position: view.camera_position.extend(0.0),
            light_direction: view.light_direction.extend(view.ambient),
            light_color: view.light_color.extend(0.0),
            resolution: phox_core::math::Vec4::new(w, h, 1.0 / w, 1.0 / h),
            frustum_planes: view.frustum_planes,
        };
        let buffers = &mut self.frames[slot];
        buffers.uniforms.ensure(size_of::<FrameUniforms>() as u64)?;
        buffers.mesh_table.ensure(byte_len(&frame.mesh_table))?;
        buffers.instance_updates.ensure(byte_len(&frame.instance_updates))?;
        buffers.material_updates.ensure(byte_len(&frame.material_updates))?;
        buffers.blend_draws.ensure(byte_len(&frame.blend_draws))?;
        buffers.blend_commands.ensure(byte_len(&frame.blend_commands))?;
        buffers.uniforms.write(bytemuck::bytes_of(&uniforms))?;
        buffers.mesh_table.write(bytemuck::cast_slice(&frame.mesh_table))?;
        buffers.instance_updates.write(bytemuck::cast_slice(&frame.instance_updates))?;
        buffers.material_updates.write(bytemuck::cast_slice(&frame.material_updates))?;
        buffers.blend_draws.write(bytemuck::cast_slice(&frame.blend_draws))?;
        buffers.blend_commands.write(bytemuck::cast_slice(&frame.blend_commands))?;

        let caps = self.device.capabilities();
        let state = |b: &Option<StateBuffer>| b.as_ref().map_or((BufferHandle::INVALID, 0), |b| (b.handle, b.address));
        let (instances, instances_address) = state(&self.scene.instances);
        let (materials, materials_address) = state(&self.scene.materials);
        let (history, history_address) = state(&self.scene.history);
        let (visibility, visibility_address) = state(&self.scene.visibility);
        Ok(FrameInputs {
            forward: ForwardIn {
                extent,
                clear_color: view.clear_color.to_array(),
                uniforms: buffers.uniforms.handle(),
                mesh_table: buffers.mesh_table.handle(),
                mesh_table_address: buffers.mesh_table.address()?,
                instance_updates: buffers.instance_updates.handle(),
                instance_updates_address: buffers.instance_updates.address()?,
                instance_update_count: frame.instance_updates.len() as u32,
                material_updates: buffers.material_updates.handle(),
                material_updates_address: buffers.material_updates.address()?,
                material_update_count: frame.material_updates.len() as u32,
                instances,
                instances_address,
                materials,
                materials_address,
                history,
                history_address,
                visibility,
                visibility_address,
                instance_count: frame.instance_count,
                draw_capacity: frame.draw_capacity,
                work_capacity: frame.work_capacity,
                draw_count: caps.has(Feature::DrawIndirectCount),
                index_pool: self.meshes.index_pool(),
                max_groups_x: caps.max_task_workgroup_count_x.max(1),
                blend_draws: buffers.blend_draws.handle(),
                blend_draws_address: buffers.blend_draws.address()?,
                blend_commands: buffers.blend_commands.handle(),
                blended: frame.blend_runs.clone(),
                meshlets: self.meshlets.then_some(MeshletSettings {
                    flags: if self.meshlet_culling { MESHLET_FRUSTUM_CULLING | MESHLET_CONE_CULLING } else { 0 },
                    lod_threshold: 1.0,
                    occlusion: self.occlusion,
                }),
            },
            mapping: view.tone_mapping,
            target_format: HEADLESS_FORMAT,
            readback: Some(readback),
        })
    }
}

/// The GPU time of each pass from the timestamps `begin` and `end` the graph writes around it.
fn pass_timings(stamps: &[Timestamp]) -> Vec<PassTiming> {
    stamps
        .iter()
        .filter_map(|begin| {
            let end = stamps.iter().find(|s| s.name.strip_suffix("/end") == Some(begin.name.as_str()))?;
            Some(PassTiming { pass: begin.name.clone(), milliseconds: (end.nanoseconds - begin.nanoseconds) / 1.0e6 })
        })
        .collect()
}

impl Drop for ForwardRenderer {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();
        self.technique.shutdown(self.device.as_ref());
        if let Some((buffer, _)) = self.readback.take() {
            self.device.destroy_buffer(buffer);
        }
    }
}

