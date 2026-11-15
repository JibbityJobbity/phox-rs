//! The headless renderer: a scene in, an image out, through the render graph on any [`Device`].

use crate::error::{ForwardError, Result};
use crate::frame::FrameSlot;
use crate::gpu_types::*;
use crate::mesh_cache::MeshCache;
use crate::passes::*;
use crate::technique::{ForwardTechnique, FrameInputs};
use crate::texture_cache::TextureCache;
use phox_core::cpu_image::CpuImage;
use phox_core::graph::*;
use phox_core::render_view::{DrawItem, RenderView, extract_view};
use phox_core::resources::Extent2D;
use phox_core::scene::Scene;
use phox_core::shader_interface::{FrameUniforms, MaterialData, material_data};
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
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self { extent: Extent2D::new(640, 480), capture_graph: false, timestamps: true }
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
    readback: Option<(BufferHandle, Extent2D)>,
    frame_count: u64,
    last_frame: CpuImage,
    timings: Vec<PassTiming>,
    dot: String,
}

fn byte_len<T>(items: &[T]) -> u64 {
    std::mem::size_of_val(items) as u64
}

impl ForwardRenderer {
    pub fn new(device: Arc<dyn Device>, config: RendererConfig) -> Result<Self> {
        let mut technique = ForwardTechnique::new(device.clone());
        technique.init(device.as_ref())?;
        let frames = (0..FRAMES_IN_FLIGHT).map(|_| FrameSlot::new(&device)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            graph: RenderGraph::new(device.clone()),
            meshes: MeshCache::new(device.clone())?,
            textures: TextureCache::new(device.clone()),
            technique,
            frames,
            readback: None,
            frame_count: 0,
            last_frame: CpuImage::default(),
            timings: Vec::new(),
            dot: String::new(),
            config,
            device,
        })
    }

    pub fn device(&self) -> &Arc<dyn Device> {
        &self.device
    }

    /// The most recent headless frame.
    pub fn last_frame(&self) -> &CpuImage {
        &self.last_frame
    }

    /// GPU time per pass of the most recent frame; empty without timestamps.
    pub fn gpu_timings(&self) -> &[PassTiming] {
        &self.timings
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

        self.meshes.begin_frame();
        let view = extract_view(scene, extent, true);
        let items: Vec<&DrawItem<'_>> = view.draws.iter().chain(view.transparent.iter()).collect();
        self.textures.begin_frame();
        for item in &items {
            self.meshes.require(item.mesh);
        }
        // Each item's material table entry names its textures by slot in the bindless table.
        let mut materials: Vec<MaterialData> = Vec::with_capacity(items.len());
        for item in &items {
            let mut data = material_data(item.material);
            let [base, metallic_roughness, normal, occlusion, emissive] = item.material.textures();
            let mut slot = |t: Option<&phox_core::texture::Texture>, field: &mut u32| -> Result<()> {
                if let Some(t) = t {
                    *field = self.textures.require(t)?;
                }
                Ok(())
            };
            slot(base, &mut data.base_color_texture)?;
            slot(metallic_roughness, &mut data.metallic_roughness_texture)?;
            slot(normal, &mut data.normal_texture)?;
            slot(occlusion, &mut data.occlusion_texture)?;
            slot(emissive, &mut data.emissive_texture)?;
            materials.push(data);
        }
        self.meshes.flush()?;
        self.textures.flush()?;

        let readback = self.readback_buffer(extent)?;
        let inputs = self.upload_frame(slot_index, &view, &items, &materials, extent, readback)?;

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

    /// Fills frame slot `slot`'s buffers for `view`: the uniforms, one material and one draw per item, and
    /// the indirect commands (opaque by cull mode, then blended runs in order).
    fn upload_frame(
        &mut self,
        slot: usize,
        view: &RenderView<'_>,
        items: &[&DrawItem<'_>],
        materials: &[MaterialData],
        extent: Extent2D,
        readback: BufferHandle,
    ) -> Result<FrameInputs> {
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
        let mut draws: Vec<GpuDraw> = Vec::with_capacity(items.len());
        for (d, item) in items.iter().enumerate() {
            let info = self.meshes.get(item.mesh).ok_or_else(|| ForwardError::Other("a mesh was not uploaded".into()))?;
            let camera = (item.model.inverse() * view.camera_position.extend(1.0)).truncate();
            draws.push(GpuDraw {
                model: item.model,
                camera_position: camera.extend(1.0),
                vertices: split_address(info.vertex_address),
                primitives: [0, 0],
                count: info.index_count,
                material: d as u32,
                visibility_offset: 0,
                flags: 0,
            });
        }
        let command = |d: usize, item: &DrawItem<'_>| {
            let info = self.meshes.get(item.mesh).expect("uploaded above");
            DrawCommand { index_count: info.index_count, instance_count: 1, first_index: info.first_index, vertex_offset: 0, first_instance: d as u32 }
        };
        let mut commands: Vec<DrawCommand> = Vec::with_capacity(items.len());
        let mut opaque = [CommandRange::default(); 2];
        for (bucket, range) in opaque.iter_mut().enumerate() {
            range.first = commands.len() as u32;
            for (d, item) in view.draws.iter().enumerate().filter(|(_, i)| usize::from(i.cull_back_faces) == bucket) {
                commands.push(command(d, item));
            }
            range.count = commands.len() as u32 - range.first;
        }
        // Blended items draw back to front, so they cannot be regrouped: runs of one cull mode, in order.
        let mut blended: Vec<BlendRun> = Vec::new();
        for (i, item) in view.transparent.iter().enumerate() {
            let first = commands.len() as u32;
            commands.push(command(view.draws.len() + i, item));
            match blended.last_mut() {
                Some(run) if run.cull_back_faces == item.cull_back_faces => run.range.count += 1,
                _ => blended.push(BlendRun { cull_back_faces: item.cull_back_faces, range: CommandRange { first, count: 1 } }),
            }
        }

        let frame = &mut self.frames[slot];
        frame.uniforms.ensure(size_of::<FrameUniforms>() as u64)?;
        frame.materials.ensure(byte_len(materials))?;
        frame.draws.ensure(byte_len(&draws))?;
        frame.commands.ensure(byte_len(&commands))?;
        frame.uniforms.write(bytemuck::bytes_of(&uniforms))?;
        frame.materials.write(bytemuck::cast_slice(materials))?;
        frame.draws.write(bytemuck::cast_slice(&draws))?;
        frame.commands.write(bytemuck::cast_slice(&commands))?;

        Ok(FrameInputs {
            forward: ForwardIn {
                extent,
                clear_color: view.clear_color.to_array(),
                uniforms: frame.uniforms.handle(),
                materials: frame.materials.handle(),
                draws: frame.draws.handle(),
                draws_address: frame.draws.address()?,
                commands: frame.commands.handle(),
                index_pool: self.meshes.index_pool(),
                opaque,
                blended,
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

