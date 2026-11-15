//! Device copies of [`Texture`]s, each in a slot of the bindless texture table.

use crate::error::Result;
use phox_core::texture::{Texture, TextureEncoding};
use phox_rhi::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Textures not used for this many frames are evicted (no frame in flight can still be reading them).
const EVICT_AFTER_FRAMES: u64 = 120;

struct Entry {
    handle: TextureHandle,
    slot: BindlessIndex,
    last_used: u64,
}

struct Pending {
    handle: TextureHandle,
    /// Every mip's texels, tightly packed one after the other, and where each starts.
    bytes: Vec<u8>,
    offsets: Vec<u64>,
}

pub struct TextureCache {
    device: Arc<dyn Device>,
    entries: HashMap<u64, Entry>,
    pending: Vec<Pending>,
    frame: u64,
}

fn format_of(encoding: TextureEncoding) -> Format {
    // sRGB textures have sRGB formats, so shader samples come back linear.
    if encoding == TextureEncoding::Srgb { Format::RGBA8Srgb } else { Format::RGBA8Unorm }
}

impl TextureCache {
    pub fn new(device: Arc<dyn Device>) -> Self {
        Self { device, entries: HashMap::new(), pending: Vec::new(), frame: 0 }
    }

    /// Starts a frame: textures unused for a long while give their slots back.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        let stale: Vec<u64> = self.entries.iter().filter(|(_, e)| frame - e.last_used > EVICT_AFTER_FRAMES).map(|(&id, _)| id).collect();
        for id in stale {
            if let Some(entry) = self.entries.remove(&id) {
                self.device.unregister_bindless_texture(entry.slot);
                self.device.destroy_texture(entry.handle);
            }
        }
    }

    /// Marks `texture` used this frame and returns its slot in the texture table at once; its texels are
    /// copied by [`TextureCache::flush`], before the frame that samples it.
    pub fn require(&mut self, texture: &Texture) -> Result<u32> {
        if let Some(entry) = self.entries.get_mut(&texture.id()) {
            entry.last_used = self.frame;
            return Ok(entry.slot.0);
        }
        let handle = self.device.create_texture(&TextureDesc {
            width: texture.width(),
            height: texture.height(),
            mips: texture.mip_count(),
            format: format_of(texture.encoding()),
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            debug_name: texture.name().to_string(),
        })?;
        let slot = self.device.register_bindless_texture(handle, texture.sampler())?;
        let mut bytes = Vec::with_capacity(texture.byte_size());
        let mut offsets = Vec::new();
        for level in 0..texture.mip_count() {
            offsets.push(bytes.len() as u64);
            bytes.extend_from_slice(&texture.mip(level).rgba);
        }
        self.pending.push(Pending { handle, bytes, offsets });
        self.entries.insert(texture.id(), Entry { handle, slot, last_used: self.frame });
        Ok(slot.0)
    }

    /// Copies the queued textures' mips and puts them in the state shaders sample them in.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut cmd = self.device.begin_commands()?;
        let mut done = Vec::new();
        for p in std::mem::take(&mut self.pending) {
            let staging = self.device.create_buffer(&BufferDesc::new(p.bytes.len() as u64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "texture staging"))?;
            self.device.write_buffer(staging, 0, &p.bytes)?;
            cmd.barrier(&[TextureBarrier { texture: p.handle, from: ResourceState::Undefined, to: ResourceState::TransferDst }], &[]);
            for (mip, &offset) in p.offsets.iter().enumerate() {
                cmd.copy_buffer_to_texture(staging, offset, p.handle, mip as u32);
            }
            done.push(TextureBarrier { texture: p.handle, from: ResourceState::TransferDst, to: ResourceState::ShaderRead });
            // Freed once the GPU is done with it.
            self.device.destroy_buffer(staging);
        }
        cmd.barrier(&done, &[]);
        self.device.submit(cmd)?;
        Ok(())
    }
}

impl Drop for TextureCache {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            self.device.unregister_bindless_texture(entry.slot);
            self.device.destroy_texture(entry.handle);
        }
    }
}
