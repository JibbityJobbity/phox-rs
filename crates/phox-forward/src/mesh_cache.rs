//! Device-local copies of meshes: one vertex buffer per mesh (read by the shaders through its address) and
//! one index pool every mesh shares, so a single indirect call can draw any of them.

use crate::error::Result;
use phox_core::meshlets::optimize_index_order;
use phox_core::scene::Mesh;
use phox_rhi::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Meshes not drawn for this many frames are evicted.
const EVICT_AFTER_FRAMES: u64 = 120;
const INITIAL_POOL_INDICES: u64 = 1 << 20;

/// Where a mesh lives on the device.
#[derive(Clone, Copy, Debug)]
pub struct GpuMeshInfo {
    pub vertex_address: u64,
    /// Where its indices start in the index pool, in indices.
    pub first_index: u32,
    pub index_count: u32,
}

struct Entry {
    version: u64,
    vertices: BufferHandle,
    info: GpuMeshInfo,
    last_used: u64,
}

struct Pending {
    mesh_id: u64,
    version: u64,
    vertex_bytes: Vec<u8>,
    indices: Vec<u32>,
}

/// Free ranges of the index pool, in indices.
#[derive(Default)]
struct RangeAllocator {
    free: Vec<(u64, u64)>, // offset, length; sorted, coalesced
    end: u64,              // everything from here on is unused
}

impl RangeAllocator {
    fn allocate(&mut self, len: u64, capacity: u64) -> Option<u64> {
        if let Some(i) = self.free.iter().position(|&(_, l)| l >= len) {
            let (offset, l) = self.free[i];
            if l == len {
                self.free.remove(i);
            } else {
                self.free[i] = (offset + len, l - len);
            }
            return Some(offset);
        }
        (self.end + len <= capacity).then(|| {
            let offset = self.end;
            self.end += len;
            offset
        })
    }

    fn release(&mut self, offset: u64, len: u64) {
        let at = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(at, (offset, len));
        if at + 1 < self.free.len() && self.free[at].0 + self.free[at].1 == self.free[at + 1].0 {
            self.free[at].1 += self.free[at + 1].1;
            self.free.remove(at + 1);
        }
        if at > 0 && self.free[at - 1].0 + self.free[at - 1].1 == self.free[at].0 {
            self.free[at - 1].1 += self.free[at].1;
            self.free.remove(at);
        }
        if let Some(&(o, l)) = self.free.last() {
            if o + l == self.end {
                self.end = o;
                self.free.pop();
            }
        }
    }
}

pub struct MeshCache {
    device: Arc<dyn Device>,
    entries: HashMap<u64, Entry>,
    pending: Vec<Pending>,
    pool: BufferHandle,
    pool_capacity: u64,
    pool_state: ResourceState,
    ranges: RangeAllocator,
    frame: u64,
}

impl MeshCache {
    pub fn new(device: Arc<dyn Device>) -> Result<Self> {
        let mut cache = Self {
            device,
            entries: HashMap::new(),
            pending: Vec::new(),
            pool: BufferHandle::INVALID,
            pool_capacity: 0,
            pool_state: ResourceState::Undefined,
            ranges: RangeAllocator::default(),
            frame: 0,
        };
        cache.pool = cache.create_pool(INITIAL_POOL_INDICES)?;
        cache.pool_capacity = INITIAL_POOL_INDICES;
        Ok(cache)
    }

    fn create_pool(&self, indices: u64) -> Result<BufferHandle> {
        let usage = BufferUsage::INDEX | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST;
        Ok(self.device.create_buffer(&BufferDesc::new(indices * 4, usage, MemoryKind::GpuOnly, "index pool"))?)
    }

    /// The buffer holding every mesh's indices; bind it as the index buffer. Changes when the pool grows.
    pub fn index_pool(&self) -> BufferHandle {
        self.pool
    }

    /// Starts a frame: meshes not required since [`EVICT_AFTER_FRAMES`] frames ago are dropped.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        let stale: Vec<u64> = self.entries.iter().filter(|(_, e)| frame - e.last_used > EVICT_AFTER_FRAMES).map(|(&id, _)| id).collect();
        for id in stale {
            if let Some(entry) = self.entries.remove(&id) {
                self.device.destroy_buffer(entry.vertices);
                self.ranges.release(u64::from(entry.info.first_index), u64::from(entry.info.index_count));
            }
        }
    }

    /// Marks `mesh` used this frame and queues its upload if it is new or its version changed. Its
    /// [`GpuMeshInfo`] is available from [`MeshCache::get`] after [`MeshCache::flush`].
    pub fn require(&mut self, mesh: &Mesh) {
        if let Some(entry) = self.entries.get_mut(&mesh.id()) {
            entry.last_used = self.frame;
            if entry.version == mesh.version() {
                return;
            }
        }
        if self.pending.iter().any(|p| p.mesh_id == mesh.id() && p.version == mesh.version()) {
            return;
        }
        self.pending.push(Pending {
            mesh_id: mesh.id(),
            version: mesh.version(),
            vertex_bytes: bytemuck::cast_slice(mesh.vertices()).to_vec(),
            // The vertex path draws indices in vertex-cache order.
            indices: optimize_index_order(mesh),
        });
    }

    pub fn get(&self, mesh: &Mesh) -> Option<GpuMeshInfo> {
        self.entries.get(&mesh.id()).map(|e| e.info)
    }

    /// Uploads what was queued: staging buffers, copies into device-local memory and the barriers that make
    /// them visible, in one submission ahead of the frame that draws them.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut self.pending);
        // Replaced meshes give their old buffers back first.
        for p in &pending {
            if let Some(old) = self.entries.remove(&p.mesh_id) {
                self.device.destroy_buffer(old.vertices);
                self.ranges.release(u64::from(old.info.first_index), u64::from(old.info.index_count));
            }
        }
        let needed: u64 = pending.iter().map(|p| p.indices.len() as u64).sum();
        let mut cmd = self.device.begin_commands()?;
        let buffer_barrier = |buffer, from, to| BufferBarrier { buffer, from, to };

        // Grow the pool when the new indices do not fit (a recorded copy; the old one is freed once idle).
        let mut capacity = self.pool_capacity;
        while self.ranges.end + needed > capacity {
            capacity *= 2;
        }
        if capacity != self.pool_capacity {
            let grown = self.create_pool(capacity)?;
            cmd.barrier(
                &[],
                &[buffer_barrier(self.pool, self.pool_state, ResourceState::TransferSrc), buffer_barrier(grown, ResourceState::Undefined, ResourceState::TransferDst)],
            );
            if self.ranges.end > 0 {
                cmd.copy_buffer(self.pool, 0, grown, 0, self.ranges.end * 4);
            }
            self.device.destroy_buffer(self.pool);
            self.pool = grown;
            self.pool_capacity = capacity;
            self.pool_state = ResourceState::TransferDst;
        } else {
            cmd.barrier(&[], &[buffer_barrier(self.pool, self.pool_state, ResourceState::TransferDst)]);
            self.pool_state = ResourceState::TransferDst;
        }

        let mut barriers = Vec::new();
        for p in &pending {
            let vertices = self.device.create_buffer(&BufferDesc {
                size: p.vertex_bytes.len() as u64,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                memory: MemoryKind::GpuOnly,
                debug_name: format!("mesh {} vertices", p.mesh_id),
                device_address: true,
            })?;
            let staging_vertices = self.device.create_buffer(&BufferDesc::new(p.vertex_bytes.len() as u64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "vertex staging"))?;
            self.device.write_buffer(staging_vertices, 0, &p.vertex_bytes)?;
            cmd.barrier(&[], &[buffer_barrier(vertices, ResourceState::Undefined, ResourceState::TransferDst)]);
            cmd.copy_buffer(staging_vertices, 0, vertices, 0, p.vertex_bytes.len() as u64);
            barriers.push(buffer_barrier(vertices, ResourceState::TransferDst, ResourceState::ShaderRead));

            let index_bytes: &[u8] = bytemuck::cast_slice(&p.indices);
            let staging_indices = self.device.create_buffer(&BufferDesc::new(index_bytes.len() as u64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "index staging"))?;
            self.device.write_buffer(staging_indices, 0, index_bytes)?;
            let first = self.ranges.allocate(p.indices.len() as u64, self.pool_capacity).expect("the pool was grown to fit");
            cmd.copy_buffer(staging_indices, 0, self.pool, first * 4, index_bytes.len() as u64);
            // The staging buffers are freed once the GPU is done with them.
            self.device.destroy_buffer(staging_vertices);
            self.device.destroy_buffer(staging_indices);

            self.entries.insert(
                p.mesh_id,
                Entry {
                    version: p.version,
                    vertices,
                    info: GpuMeshInfo { vertex_address: self.device.buffer_address(vertices)?, first_index: first as u32, index_count: p.indices.len() as u32 },
                    last_used: self.frame,
                },
            );
        }
        barriers.push(buffer_barrier(self.pool, ResourceState::TransferDst, ResourceState::VertexOrIndex));
        self.pool_state = ResourceState::VertexOrIndex;
        cmd.barrier(&[], &barriers);
        self.device.submit(cmd)?;
        Ok(())
    }
}

impl Drop for MeshCache {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            self.device.destroy_buffer(entry.vertices);
        }
        self.device.destroy_buffer(self.pool);
    }
}
