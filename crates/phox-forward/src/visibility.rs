//! Which meshlets each object had visible at the end of the last frame (one bit per meshlet), kept across
//! frames for occlusion culling. The early pass draws what was visible; the late pass tests everything.

use crate::error::Result;
use phox_rhi::*;
use std::collections::HashMap;
use std::sync::Arc;

struct Range {
    mesh: u64,
    offset: u32,
    words: u32,
}

/// Visibility bits in a device buffer, in whole 32-bit words per object, so each task shader workgroup reads
/// and writes its own word without atomics. Ranges come from a bump allocator; an object keeps its range
/// while it draws the same mesh. When the buffer is full the store starts over with every bit cleared,
/// growing to twice what the frame needs. The bits are hints: starting over only costs a frame of culling.
pub struct VisibilityStore {
    device: Arc<dyn Device>,
    buffer: BufferHandle,
    capacity_words: u32,
    next: u32,
    ranges: HashMap<u64, Range>,
}

/// Where one object's bits are, and whether they carry last frame's result.
#[derive(Clone, Copy, Debug)]
pub struct Visibility {
    pub offset: u32,
    /// The bits were written by an earlier frame for this mesh.
    pub valid: bool,
}

impl VisibilityStore {
    pub fn new(device: Arc<dyn Device>) -> Self {
        Self { device, buffer: BufferHandle::INVALID, capacity_words: 0, next: 0, ranges: HashMap::new() }
    }

    pub fn buffer(&self) -> BufferHandle {
        self.buffer
    }

    /// Finds or allocates the range of each `(object, mesh, meshlets)`, in order.
    pub fn assign(&mut self, objects: &[(u64, u64, u32)]) -> Result<Vec<Visibility>> {
        let words = |meshlets: u32| meshlets.div_ceil(32).max(1);
        let needed: u32 = objects.iter().map(|&(_, _, m)| words(m)).sum();
        let fresh_needed: u32 = objects
            .iter()
            .filter(|(id, mesh, m)| self.ranges.get(id).is_none_or(|r| r.mesh != *mesh || r.words != words(*m)))
            .map(|&(_, _, m)| words(m))
            .sum();
        if !self.buffer.is_valid() || self.next + fresh_needed > self.capacity_words {
            self.restart(needed)?;
        }
        let mut out = Vec::with_capacity(objects.len());
        for &(id, mesh, meshlets) in objects {
            let w = words(meshlets);
            match self.ranges.get(&id) {
                Some(r) if r.mesh == mesh && r.words == w => out.push(Visibility { offset: r.offset, valid: true }),
                _ => {
                    let offset = self.next;
                    self.next += w;
                    self.ranges.insert(id, Range { mesh, offset, words: w });
                    out.push(Visibility { offset, valid: false });
                }
            }
        }
        Ok(out)
    }

    /// A new, cleared buffer with room for twice what the frame needs; every object loses its range.
    fn restart(&mut self, needed: u32) -> Result<()> {
        let capacity = (needed * 2).max(1024);
        let buffer = self.device.create_buffer(&BufferDesc {
            size: u64::from(capacity) * 4,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            memory: MemoryKind::GpuOnly,
            debug_name: "meshlet visibility".into(),
            device_address: true,
        })?;
        let mut cmd = self.device.begin_commands()?;
        cmd.barrier(&[], &[BufferBarrier { buffer, from: ResourceState::Undefined, to: ResourceState::TransferDst }]);
        cmd.fill_buffer(buffer, 0, u64::from(capacity) * 4, 0);
        // The frame's graph imports the buffer from `Undefined`, which orders it after this clear.
        self.device.submit(cmd)?;
        if self.buffer.is_valid() {
            self.device.destroy_buffer(self.buffer);
        }
        self.buffer = buffer;
        self.capacity_words = capacity;
        self.next = 0;
        self.ranges.clear();
        Ok(())
    }

    pub fn address(&self) -> Result<u64> {
        Ok(self.device.buffer_address(self.buffer)?)
    }
}

impl Drop for VisibilityStore {
    fn drop(&mut self) {
        if self.buffer.is_valid() {
            self.device.destroy_buffer(self.buffer);
        }
    }
}
