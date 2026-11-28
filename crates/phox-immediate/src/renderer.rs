//! Draws a [`DrawList`] through the render graph: to a window's swapchain image, or to an image read back to
//! the CPU.

use std::sync::Arc;

use phox_core::graph::{BufferAccess, RenderGraph, TextureAccess};
use phox_rhi::{
    BlendDesc, BufferDesc, BufferHandle, BufferUsage, CommandList, Device, Format, GraphicsPipelineDesc, MemoryKind, PipelineHandle,
    PushConstantRange, ResourceState, ShaderDesc, ShaderHandle, ShaderStage, ShaderStages, SubmitTicket, TextureDesc,
    TextureUsage, VertexAttribute, VertexFormat, VertexLayout,
};

use crate::draw::{Batch, Color, DrawList, Vertex};
use crate::error::{ImmediateError, Result};

const FRAMES_IN_FLIGHT: usize = 2;

const VERTEX_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/immediate.vert.spv"));
const FRAGMENT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/immediate.frag.spv"));

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    viewport: [f32; 2],
    linear_output: u32,
    padding: u32,
}

/// The vertex buffer one frame in flight writes, and that frame's submission.
#[derive(Default)]
struct FrameSlot {
    vertices: Option<(BufferHandle, u64)>,
    ticket: Option<SubmitTicket>,
}

pub struct ImmediateRenderer {
    device: Arc<dyn Device>,
    graph: RenderGraph,
    vertex: ShaderHandle,
    fragment: ShaderHandle,
    pipelines: Vec<(Format, PipelineHandle)>,
    slots: [FrameSlot; FRAMES_IN_FLIGHT],
    frame: u64,
}

fn spirv(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect()
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

const fn is_srgb(format: Format) -> bool {
    matches!(format, Format::RGBA8Srgb | Format::BGRA8Srgb)
}

impl ImmediateRenderer {
    pub fn new(device: Arc<dyn Device>) -> Result<Self> {
        let vertex = device.create_shader(&ShaderDesc::new(ShaderStage::Vertex, spirv(VERTEX_SPIRV)))?;
        let fragment = device.create_shader(&ShaderDesc::new(ShaderStage::Fragment, spirv(FRAGMENT_SPIRV)))?;
        Ok(Self {
            graph: RenderGraph::new(device.clone()),
            device,
            vertex,
            fragment,
            pipelines: Vec::new(),
            slots: Default::default(),
            frame: 0,
        })
    }

    pub fn device(&self) -> &Arc<dyn Device> {
        &self.device
    }

    fn pipeline(&mut self, format: Format) -> Result<PipelineHandle> {
        if let Some((_, p)) = self.pipelines.iter().find(|(f, _)| *f == format) {
            return Ok(*p);
        }
        let desc = GraphicsPipelineDesc {
            shaders: vec![self.vertex, self.fragment],
            vertex_layout: VertexLayout {
                stride: size_of::<Vertex>() as u32,
                attributes: vec![
                    VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 },
                    VertexAttribute { location: 1, format: VertexFormat::Float32x4, offset: 8 },
                ],
            },
            color_formats: vec![format],
            blend: Some(BlendDesc { enabled: true }),
            push_constants: vec![PushConstantRange { stages: ShaderStages::VERTEX, offset: 0, size: size_of::<PushConstants>() as u32 }],
            debug_name: "immediate".into(),
            ..GraphicsPipelineDesc::default()
        };
        let pipeline = self.device.create_graphics_pipeline(&desc)?;
        self.pipelines.push((format, pipeline));
        Ok(pipeline)
    }

    /// Waits for the frame that last used the next slot, then uploads `list` into that slot's buffer.
    fn begin_slot(&mut self, list: &DrawList) -> Result<(usize, Option<BufferHandle>)> {
        let index = (self.frame % FRAMES_IN_FLIGHT as u64) as usize;
        self.frame += 1;
        if let Some(ticket) = self.slots[index].ticket.take() {
            self.device.wait(ticket)?;
        }
        let bytes: &[u8] = bytemuck::cast_slice(list.vertices());
        if bytes.is_empty() {
            return Ok((index, None));
        }
        let needed = bytes.len() as u64;
        let slot = &mut self.slots[index];
        if slot.vertices.is_none_or(|(_, capacity)| capacity < needed) {
            if let Some((old, _)) = slot.vertices.take() {
                self.device.destroy_buffer(old);
            }
            let capacity = needed.next_power_of_two().max(64 * 1024);
            let buffer = self.device.create_buffer(&BufferDesc::new(capacity, BufferUsage::VERTEX, MemoryKind::Upload, "immediate vertices"))?;
            slot.vertices = Some((buffer, capacity));
        }
        let (buffer, _) = slot.vertices.expect("allocated above");
        self.device.write_buffer(buffer, 0, bytes)?;
        Ok((index, Some(buffer)))
    }

    fn clear_value(clear: Color, format: Format) -> [f32; 4] {
        if is_srgb(format) {
            [srgb_to_linear(clear.r), srgb_to_linear(clear.g), srgb_to_linear(clear.b), clear.a]
        } else {
            [clear.r, clear.g, clear.b, clear.a]
        }
    }

    /// Draws `list` into the window the device presents to, `width` x `height` physical pixels, after clearing
    /// it to `clear`. Returns false when there is nothing to draw into (a minimised window).
    pub fn render_window(&mut self, list: &DrawList, width: u32, height: u32, clear: Color) -> Result<bool> {
        let Some(backbuffer) = self.device.acquire_backbuffer(width, height)? else { return Ok(false) };
        let pipeline = self.pipeline(backbuffer.format)?;
        let (slot, vertices) = self.begin_slot(list)?;
        let constants = push_constants(list, backbuffer.format);
        let batches = list.batches();
        let (scale, target_size) = (list.scale(), (backbuffer.width, backbuffer.height));
        let clear = Self::clear_value(clear, backbuffer.format);

        let mut b = self.graph.begin();
        let target = b.import_texture(backbuffer.texture, ResourceState::Undefined, ResourceState::Present)?;
        let vertices = vertices.map(|v| b.import_buffer(v, ResourceState::Undefined, ResourceState::Undefined)).transpose()?;
        b.add_pass(
            "immediate",
            move |p| {
                if let Some(v) = vertices {
                    p.read_buffer(v, BufferAccess::Vertex)?;
                }
                p.color_target_with(target, true, clear)
            },
            move |c| {
                if let Some(v) = vertices {
                    c.cmd.bind_pipeline(pipeline);
                    c.cmd.bind_vertex_buffer(c.buffer(v)?, 0);
                    c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&constants));
                    record_batches(c.cmd, &batches, scale, target_size);
                }
                Ok(())
            },
        )?;
        b.compile()?;
        let commands = b.record()?;
        drop(b);
        self.slots[slot].ticket = Some(self.device.submit_for_present(commands, &backbuffer)?);
        Ok(true)
    }

    /// Draws `list` into a new `width` x `height` RGBA8 (sRGB-encoded) image and returns its pixels, row by
    /// row from the top. For tests and previews; waits for the device.
    pub fn render_image(&mut self, list: &DrawList, width: u32, height: u32, clear: Color) -> Result<Vec<u8>> {
        let format = Format::RGBA8Unorm;
        let pipeline = self.pipeline(format)?;
        let (slot, vertices) = self.begin_slot(list)?;
        let constants = push_constants(list, format);
        let batches = list.batches();
        let (scale, target_size) = (list.scale(), (width, height));
        let clear = Self::clear_value(clear, format);
        let size = u64::from(width) * u64::from(height) * 4;
        let readback = self.device.create_buffer(&BufferDesc::new(size, BufferUsage::TRANSFER_DST, MemoryKind::Readback, "immediate readback"))?;

        let result = (|| -> Result<Vec<u8>> {
            let mut b = self.graph.begin();
            let image = b.create_texture(&TextureDesc::new(width, height, format, TextureUsage::NONE, "immediate image"));
            let vertices = vertices.map(|v| b.import_buffer(v, ResourceState::Undefined, ResourceState::Undefined)).transpose()?;
            let out = b.import_buffer(readback, ResourceState::Undefined, ResourceState::HostRead)?;
            b.add_pass(
                "immediate",
                move |p| {
                    if let Some(v) = vertices {
                        p.read_buffer(v, BufferAccess::Vertex)?;
                    }
                    p.color_target_with(image, true, clear)
                },
                move |c| {
                    if let Some(v) = vertices {
                        c.cmd.bind_pipeline(pipeline);
                        c.cmd.bind_vertex_buffer(c.buffer(v)?, 0);
                        c.cmd.push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&constants));
                        record_batches(c.cmd, &batches, scale, target_size);
                    }
                    Ok(())
                },
            )?;
            b.add_pass(
                "readback",
                move |p| {
                    p.read_texture(image, TextureAccess::TransferSrc)?;
                    p.write_buffer(out, BufferAccess::TransferDst)
                },
                move |c| {
                    let (src, dst) = (c.texture(image)?, c.buffer(out)?);
                    c.cmd.copy_texture_to_buffer(src, dst);
                    Ok(())
                },
            )?;
            b.compile()?;
            let commands = b.record()?;
            drop(b);
            let ticket = self.device.submit(commands)?;
            self.slots[slot].ticket = Some(ticket);
            self.device.wait(ticket)?;
            let mut pixels = vec![0u8; size as usize];
            self.device.read_buffer(readback, 0, &mut pixels)?;
            Ok(pixels)
        })();
        self.device.destroy_buffer(readback);
        result
    }
}

/// One draw per batch, each with its clip as a scissor in physical pixels (rounded outwards).
fn record_batches(cmd: &mut dyn CommandList, batches: &[Batch], scale: f32, (width, height): (u32, u32)) {
    for batch in batches {
        match batch.clip {
            Some(clip) => {
                let x0 = (clip.x * scale).floor().max(0.0) as u32;
                let y0 = (clip.y * scale).floor().max(0.0) as u32;
                let x1 = ((clip.right() * scale).ceil().max(0.0) as u32).min(width);
                let y1 = ((clip.bottom() * scale).ceil().max(0.0) as u32).min(height);
                cmd.set_scissor(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0));
            }
            None => cmd.set_scissor(0, 0, width, height),
        }
        cmd.draw(batch.vertex_count, 1, batch.first_vertex);
    }
}

fn push_constants(list: &DrawList, format: Format) -> PushConstants {
    let (w, h) = list.size();
    PushConstants { viewport: [w, h], linear_output: u32::from(is_srgb(format)), padding: 0 }
}

impl Drop for ImmediateRenderer {
    fn drop(&mut self) {
        if let Err(e) = self.device.wait_idle() {
            // Destroying objects the device may still use is worse than leaking them.
            eprintln!("phox-immediate: wait_idle failed ({e}); leaking GPU objects");
            return;
        }
        for slot in &mut self.slots {
            if let Some((buffer, _)) = slot.vertices.take() {
                self.device.destroy_buffer(buffer);
            }
        }
        for (_, p) in self.pipelines.drain(..) {
            self.device.destroy_pipeline(p);
        }
        self.device.destroy_shader(self.vertex);
        self.device.destroy_shader(self.fragment);
    }
}

impl From<ImmediateError> for phox_core::graph::GraphError {
    fn from(e: ImmediateError) -> Self {
        match e {
            ImmediateError::Graph(g) => g,
            ImmediateError::Device(d) => phox_core::graph::GraphError::Device(d),
        }
    }
}
