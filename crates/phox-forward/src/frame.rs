//! Per-frame upload memory: host-visible buffers the CPU fills each frame, one set per frame in flight.

use crate::error::Result;
use phox_rhi::*;
use std::sync::Arc;

/// A host-visible buffer that grows (to the next power of two) when a frame needs more room.
pub struct GrowBuffer {
    device: Arc<dyn Device>,
    handle: BufferHandle,
    capacity: u64,
    usage: BufferUsage,
    device_address: bool,
    name: &'static str,
}

impl GrowBuffer {
    pub fn new(device: Arc<dyn Device>, name: &'static str, usage: BufferUsage, device_address: bool) -> Result<Self> {
        let mut buffer = Self { device, handle: BufferHandle::INVALID, capacity: 0, usage, device_address, name };
        buffer.ensure(256)?;
        Ok(buffer)
    }

    /// Makes room for `bytes`. The previous contents are not kept: a frame rewrites the buffer.
    pub fn ensure(&mut self, bytes: u64) -> Result<()> {
        if bytes <= self.capacity {
            return Ok(());
        }
        let capacity = bytes.next_power_of_two().max(256);
        let desc = BufferDesc { size: capacity, usage: self.usage, memory: MemoryKind::Upload, debug_name: self.name.to_string(), device_address: self.device_address };
        let handle = self.device.create_buffer(&desc)?;
        if self.handle.is_valid() {
            self.device.destroy_buffer(self.handle);
        }
        self.handle = handle;
        self.capacity = capacity;
        Ok(())
    }

    pub fn handle(&self) -> BufferHandle {
        self.handle
    }

    pub fn address(&self) -> Result<u64> {
        Ok(self.device.buffer_address(self.handle)?)
    }

    pub fn write(&self, data: &[u8]) -> Result<()> {
        if !data.is_empty() {
            self.device.write_buffer(self.handle, 0, data)?;
        }
        Ok(())
    }
}

impl Drop for GrowBuffer {
    fn drop(&mut self) {
        if self.handle.is_valid() {
            self.device.destroy_buffer(self.handle);
        }
    }
}

/// What one frame in flight owns: its buffers, and the submission that last used them.
pub struct FrameSlot {
    /// The frame's `FrameUniforms`.
    pub uniforms: GrowBuffer,
    /// The frame's `GpuMesh` table.
    pub mesh_table: GrowBuffer,
    /// What changed in the scene this frame: `InstanceUpdate`s and `MaterialUpdate`s.
    pub instance_updates: GrowBuffer,
    pub material_updates: GrowBuffer,
    /// Blended objects' `GpuDraw`s and indirect draw commands, farthest first.
    pub blend_draws: GrowBuffer,
    pub blend_commands: GrowBuffer,
    pub ticket: Option<SubmitTicket>,
}

impl FrameSlot {
    pub fn new(device: &Arc<dyn Device>) -> Result<Self> {
        Ok(Self {
            uniforms: GrowBuffer::new(device.clone(), "frame uniforms", BufferUsage::UNIFORM, false)?,
            mesh_table: GrowBuffer::new(device.clone(), "mesh table", BufferUsage::STORAGE, true)?,
            instance_updates: GrowBuffer::new(device.clone(), "instance updates", BufferUsage::STORAGE, true)?,
            material_updates: GrowBuffer::new(device.clone(), "material updates", BufferUsage::STORAGE, true)?,
            blend_draws: GrowBuffer::new(device.clone(), "blend draws", BufferUsage::STORAGE, true)?,
            blend_commands: GrowBuffer::new(device.clone(), "blend commands", BufferUsage::INDIRECT, false)?,
            ticket: None,
        })
    }
}
