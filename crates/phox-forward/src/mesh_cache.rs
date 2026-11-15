//! Device-local copies of meshes: one vertex buffer per mesh (read by the shaders through its address) and
//! one index pool every mesh shares, so a single indirect call can draw any of them.

use crate::error::Result;
use phox_core::meshlets::{MeshletMesh, build_meshlet_lod, build_meshlets, optimize_index_order};
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
    /// Mesh shading: the address of its meshlets, followed by their data (0 without mesh shading).
    pub meshlet_address: u64,
    /// Meshlets of every LOD level.
    pub meshlet_count: u32,
}

struct Entry {
    version: u64,
    vertices: BufferHandle,
    meshlets: BufferHandle,
    cpu_meshlets: Option<Arc<MeshletMesh>>,
    info: GpuMeshInfo,
    last_used: u64,
}

struct Pending {
    mesh_id: u64,
    version: u64,
    vertex_bytes: Vec<u8>,
    indices: Vec<u32>,
    /// The meshlets then their data, as the shaders read them, and how many meshlets there are.
    meshlets: Option<(Vec<u8>, u32)>,
    cpu_meshlets: Option<Arc<MeshletMesh>>,
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
    meshlets: bool,
    lod: bool,
    keep_cpu: bool,
}

impl MeshCache {
    /// With `meshlets`, every mesh also gets a meshlet buffer for the task and mesh shaders.
    /// With `lod`, meshlets form a cluster LOD hierarchy (`build_meshlet_lod`); `keep_cpu` also keeps the
    /// meshlets in host memory, for statistics.
    pub fn new(device: Arc<dyn Device>, meshlets: bool, lod: bool, keep_cpu: bool) -> Result<Self> {
        let mut cache = Self {
            device,
            entries: HashMap::new(),
            pending: Vec::new(),
            pool: BufferHandle::INVALID,
            pool_capacity: 0,
            pool_state: ResourceState::Undefined,
            ranges: RangeAllocator::default(),
            frame: 0,
            meshlets,
            lod,
            keep_cpu,
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
                self.free_entry(&entry);
            }
        }
    }

    fn free_entry(&mut self, entry: &Entry) {
        self.device.destroy_buffer(entry.vertices);
        if entry.meshlets.is_valid() {
            self.device.destroy_buffer(entry.meshlets);
        }
        self.ranges.release(u64::from(entry.info.first_index), u64::from(entry.info.index_count));
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
        // The vertex path draws indices in vertex-cache order, and the meshlets are built from that order.
        let indices = optimize_index_order(mesh);
        let mut cpu_meshlets = None;
        let meshlets = self.meshlets.then(|| {
            let mut built = if self.lod { build_meshlet_lod(mesh, &indices) } else { build_meshlets(mesh, &indices) };
            if self.keep_cpu {
                cpu_meshlets = Some(Arc::new(built.clone()));
            }
            // Offsets count from the start of the buffer, which holds the meshlets first.
            let shift = (built.meshlets.len() * (size_of::<phox_core::meshlets::Meshlet>() / 4)) as u32;
            for m in &mut built.meshlets {
                m.vertex_offset += shift;
                m.triangle_offset += shift;
            }
            let mut bytes: Vec<u8> = bytemuck::cast_slice(&built.meshlets).to_vec();
            bytes.extend_from_slice(bytemuck::cast_slice(&built.data));
            (bytes, built.meshlets.len() as u32)
        });
        self.pending.push(Pending { mesh_id: mesh.id(), version: mesh.version(), vertex_bytes: bytemuck::cast_slice(mesh.vertices()).to_vec(), indices, meshlets, cpu_meshlets });
    }

    /// The mesh's meshlets in host memory (only with `keep_cpu`).
    pub fn cpu_meshlets(&self, mesh: &Mesh) -> Option<Arc<MeshletMesh>> {
        self.entries.get(&mesh.id()).and_then(|e| e.cpu_meshlets.clone())
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
                self.free_entry(&old);
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
            let vertices = self.upload_storage(cmd.as_mut(), &p.vertex_bytes, &format!("mesh {} vertices", p.mesh_id), &mut barriers)?;
            let (meshlets, meshlet_count) = match &p.meshlets {
                Some((bytes, count)) => (self.upload_storage(cmd.as_mut(), bytes, &format!("mesh {} meshlets", p.mesh_id), &mut barriers)?, *count),
                None => (BufferHandle::INVALID, 0),
            };

            let index_bytes: &[u8] = bytemuck::cast_slice(&p.indices);
            let staging_indices = self.device.create_buffer(&BufferDesc::new(index_bytes.len() as u64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "index staging"))?;
            self.device.write_buffer(staging_indices, 0, index_bytes)?;
            let first = self.ranges.allocate(p.indices.len() as u64, self.pool_capacity).expect("the pool was grown to fit");
            cmd.copy_buffer(staging_indices, 0, self.pool, first * 4, index_bytes.len() as u64);
            // The staging buffers are freed once the GPU is done with them.
            self.device.destroy_buffer(staging_indices);

            self.entries.insert(
                p.mesh_id,
                Entry {
                    version: p.version,
                    vertices,
                    meshlets,
                    cpu_meshlets: p.cpu_meshlets.clone(),
                    info: GpuMeshInfo {
                        vertex_address: self.device.buffer_address(vertices)?,
                        first_index: first as u32,
                        index_count: p.indices.len() as u32,
                        meshlet_address: if meshlets.is_valid() { self.device.buffer_address(meshlets)? } else { 0 },
                        meshlet_count,
                    },
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

impl MeshCache {
    /// A device-local buffer shaders read through its address, filled with `bytes` through a staging buffer.
    /// Appends the barrier that makes it readable to `barriers`.
    fn upload_storage(&self, cmd: &mut dyn CommandList, bytes: &[u8], name: &str, barriers: &mut Vec<BufferBarrier>) -> Result<BufferHandle> {
        let buffer = self.device.create_buffer(&BufferDesc {
            size: bytes.len() as u64,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            memory: MemoryKind::GpuOnly,
            debug_name: name.to_string(),
            device_address: true,
        })?;
        let staging = self.device.create_buffer(&BufferDesc::new(bytes.len() as u64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "staging"))?;
        self.device.write_buffer(staging, 0, bytes)?;
        cmd.barrier(&[], &[BufferBarrier { buffer, from: ResourceState::Undefined, to: ResourceState::TransferDst }]);
        cmd.copy_buffer(staging, 0, buffer, 0, bytes.len() as u64);
        barriers.push(BufferBarrier { buffer, from: ResourceState::TransferDst, to: ResourceState::ShaderRead });
        // Freed once the GPU is done with it.
        self.device.destroy_buffer(staging);
        Ok(buffer)
    }
}

impl Drop for MeshCache {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            self.device.destroy_buffer(entry.vertices);
            if entry.meshlets.is_valid() {
                self.device.destroy_buffer(entry.meshlets);
            }
        }
        self.device.destroy_buffer(self.pool);
    }
}
