//! The scene as the GPU-driven forward pass reads it, kept on the GPU across frames: a `GpuInstance` and a
//! `MaterialData` per object, indexed by the object's slot, in device-local buffers. Each frame `update`
//! finds what changed and hands only that to the scene update pass, which scatters it into the buffers.
//!
//! It reads the objects the scene logged ([`Scene::changes`]) and, of each, what changed (the object's
//! transform and material revisions); only when the scene cannot say (the first frame, another scene, a log
//! that started over) does it look at every object. So a still scene costs next to nothing, and a moving
//! one what moves. Meshes whose version changed are found by checking each distinct mesh.
//!
//! An object's slot is its `ObjectId::index`, which stays the same while the object exists, so the per-slot
//! state the GPU keeps across frames follows it: whether it was visible at the end of the last frame (the
//! history), and with meshlets, one visibility bit per meshlet in whole 32-bit words. Both are hints: stale
//! ones cost time, never correctness, since the late phase tests whatever the early phase skipped. When the
//! bits run out they start over, cleared.
//!
//! Blended objects are not culled on the GPU, because they must be drawn back to front: the CPU writes their
//! draws and commands in that order every frame, in runs that share a pipeline.

use crate::error::Result;
use crate::gpu_types::*;
use crate::mesh_cache::MeshCache;
use crate::passes::{BlendRun, CommandRange};
use crate::texture_cache::TextureCache;
use phox_core::math::{Mat3, Vec3};
use phox_core::render_view::RenderView;
use phox_core::scene::{AlphaMode, ChangeCursor, Mesh, Object, Scene};
use phox_core::shader_interface::{MaterialData, material_data};
use phox_core::texture::Texture;
use phox_rhi::*;
use std::collections::HashMap;
use std::sync::Arc;

const INITIAL_SLOTS: u32 = 1 << 10;
const INITIAL_VISIBILITY_WORDS: u32 = 1 << 14;

/// A device-local buffer shaders read and write through its address.
pub struct StateBuffer {
    device: Arc<dyn Device>,
    pub handle: BufferHandle,
    pub address: u64,
    pub size: u64,
}

impl StateBuffer {
    /// A new buffer of `size` bytes, filled with `value` (a recorded fill, submitted at once).
    fn new(device: &Arc<dyn Device>, name: &str, size: u64, value: u32) -> Result<Self> {
        let handle = device.create_buffer(&BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            memory: MemoryKind::GpuOnly,
            debug_name: name.to_string(),
            device_address: true,
        })?;
        let mut cmd = device.begin_commands()?;
        cmd.barrier(&[], &[BufferBarrier { buffer: handle, from: ResourceState::Undefined, to: ResourceState::TransferDst }]);
        cmd.fill_buffer(handle, 0, size, value);
        // The frame's graph imports the buffer from `Undefined`, which orders it after this fill.
        device.submit(cmd)?;
        Ok(Self { device: device.clone(), handle, address: device.buffer_address(handle)?, size })
    }
}

impl Drop for StateBuffer {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.handle);
    }
}

/// What `update` checks for every object every frame.
#[derive(Clone, Copy, Default)]
struct Tracked {
    /// `Object::id` of the object in the slot; 0: empty.
    object: u64,
    transform_revision: u64,
    material_revision: u64,
    /// The last `update` that found the object.
    seen: u64,
}

#[derive(Default)]
struct Slot {
    occupied: bool,
    model: phox_core::math::Mat4,
    /// Index into the mesh entries; `NO_MESH`: the object has none.
    mesh: u32,
    blended: bool,
    double_sided: bool,
    material: Option<MaterialData>,
    textures: [Option<Arc<Texture>>; 5],
    /// The mesh (id and version) the visibility words were handed out for, and where they are.
    visibility_for: (u64, u64),
    visibility_offset: u32,
    visibility_words: u32,
    /// Counted in the cull passes' capacities.
    counted: bool,
    groups: u32,
    instance_dirty: bool,
    material_dirty: bool,
}

struct MeshEntry {
    mesh: Arc<Mesh>,
    users: u32,
    /// The `Mesh::version` the slots using it were last brought up to date with.
    version: u64,
}

/// What `update` hands the frame.
pub struct SceneFrame {
    pub mesh_table: Vec<GpuMesh>,
    pub instance_updates: Vec<InstanceUpdate>,
    pub material_updates: Vec<MaterialUpdate>,
    /// Blended objects' draws and commands (farthest first), and the runs that share a pipeline.
    pub blend_draws: Vec<GpuDraw>,
    pub blend_commands: Vec<DrawCommand>,
    pub blend_runs: Vec<BlendRun>,
    /// Slots, empty ones included.
    pub instance_count: u32,
    /// Most draws a cull pass can append, at least 1.
    pub draw_capacity: u32,
    /// Most work one bucket can take, at least 1: work items (meshlet path) or draw commands (vertex path).
    pub work_capacity: u32,
}

pub struct GpuScene {
    device: Arc<dyn Device>,
    meshlets: bool,
    occlusion: bool,
    cursor: ChangeCursor,
    updates: u64,
    tracked: Vec<Tracked>,
    slots: Vec<Slot>,
    meshes: Vec<Option<MeshEntry>>,
    free_meshes: Vec<u32>,
    mesh_index: HashMap<u64, u32>,
    textures: HashMap<u64, (Arc<Texture>, u32)>,
    capacity: u32,
    pub instances: Option<StateBuffer>,
    pub materials: Option<StateBuffer>,
    pub history: Option<StateBuffer>,
    pub visibility: Option<StateBuffer>,
    visibility_used: u32,
    instance_dirty: Vec<u32>,
    material_dirty: Vec<u32>,
    draw_count: u32,
    work_total: u32,
}

fn words_for(meshlets: u32) -> u32 {
    meshlets.div_ceil(32).max(1)
}

impl GpuScene {
    /// `meshlets`: the mesh cache holds meshlets (task and mesh shaders), otherwise index buffers.
    /// `occlusion`: keep the history and visibility bits for two-phase occlusion culling.
    pub fn new(device: Arc<dyn Device>, meshlets: bool, occlusion: bool) -> Self {
        Self {
            device,
            meshlets,
            occlusion,
            cursor: ChangeCursor::default(),
            updates: 0,
            tracked: Vec::new(),
            slots: Vec::new(),
            meshes: Vec::new(),
            free_meshes: Vec::new(),
            mesh_index: HashMap::new(),
            textures: HashMap::new(),
            capacity: 0,
            instances: None,
            materials: None,
            history: None,
            visibility: None,
            visibility_used: 0,
            instance_dirty: Vec::new(),
            material_dirty: Vec::new(),
            draw_count: 0,
            work_total: 0,
        }
    }

    fn acquire_mesh(&mut self, mesh: &Arc<Mesh>) -> u32 {
        if let Some(&i) = self.mesh_index.get(&mesh.id()) {
            if let Some(entry) = self.meshes[i as usize].as_mut() {
                entry.users += 1;
                return i;
            }
        }
        let entry = MeshEntry { mesh: mesh.clone(), users: 1, version: 0 };
        let index = match self.free_meshes.pop() {
            Some(i) => {
                self.meshes[i as usize] = Some(entry);
                i
            }
            None => {
                self.meshes.push(Some(entry));
                self.meshes.len() as u32 - 1
            }
        };
        self.mesh_index.insert(mesh.id(), index);
        index
    }

    fn release_mesh(&mut self, index: u32) {
        let Some(entry) = self.meshes[index as usize].as_mut() else { return };
        entry.users -= 1;
        if entry.users == 0 {
            self.mesh_index.remove(&entry.mesh.id());
            self.meshes[index as usize] = None;
            self.free_meshes.push(index);
        }
    }

    fn release_textures(&mut self, textures: [Option<Arc<Texture>>; 5]) {
        for texture in textures.into_iter().flatten() {
            if let Some(entry) = self.textures.get_mut(&texture.id()) {
                entry.1 -= 1;
                if entry.1 == 0 {
                    self.textures.remove(&texture.id());
                }
            }
        }
    }

    /// Counts (or stops counting) the slot towards the cull passes' capacities.
    fn set_counted(&mut self, index: usize, counted: bool, groups: u32) {
        let slot = &mut self.slots[index];
        if slot.counted {
            self.draw_count -= 1;
            self.work_total -= slot.groups;
        }
        slot.counted = counted;
        slot.groups = if counted { groups } else { 0 };
        if counted {
            self.draw_count += 1;
            self.work_total += groups;
        }
    }

    /// Empties slot `index`, releasing what it holds; its instance is rewritten as empty.
    fn release(&mut self, index: usize) {
        if !self.slots[index].occupied {
            return;
        }
        let (mesh, textures) = (self.slots[index].mesh, std::mem::take(&mut self.slots[index].textures));
        if mesh != NO_MESH {
            self.release_mesh(mesh);
        }
        self.release_textures(textures);
        self.set_counted(index, false, 0);
        let instance_dirty = self.slots[index].instance_dirty;
        self.slots[index] = Slot { mesh: NO_MESH, instance_dirty, ..Slot::default() };
        self.tracked[index].object = 0;
        self.mark_instance(index);
    }

    fn mark_instance(&mut self, index: usize) {
        if !self.slots[index].instance_dirty {
            self.slots[index].instance_dirty = true;
            self.instance_dirty.push(index as u32);
        }
    }

    fn mark_material(&mut self, index: usize) {
        if !self.slots[index].material_dirty {
            self.slots[index].material_dirty = true;
            self.material_dirty.push(index as u32);
        }
    }

    /// Brings slot `index` up to date with `object`, if anything changed.
    fn track(&mut self, index: usize, object: &Object, textures: &mut TextureCache) -> Result<()> {
        let tracked = self.tracked[index];
        self.tracked[index].seen = self.updates;
        let new_object = tracked.object != object.id();
        let transform_changed = new_object || tracked.transform_revision != object.transform_revision();
        let material_changed = new_object || tracked.material_revision != object.material_revision();
        if !transform_changed && !material_changed {
            return Ok(());
        }
        if new_object {
            self.release(index);
        }
        self.tracked[index] = Tracked {
            object: object.id(),
            transform_revision: object.transform_revision(),
            material_revision: object.material_revision(),
            seen: self.updates,
        };
        self.slots[index].occupied = true;

        if transform_changed {
            let mesh = object.mesh().filter(|m| !m.is_empty());
            let current = self.slots[index].mesh;
            let same = match (mesh, current) {
                (Some(m), c) if c != NO_MESH => self.meshes[c as usize].as_ref().is_some_and(|e| e.mesh.id() == m.id()),
                (None, c) => c == NO_MESH,
                _ => false,
            };
            if !same {
                if current != NO_MESH {
                    self.release_mesh(current);
                }
                self.slots[index].mesh = mesh.map_or(NO_MESH, |m| self.acquire_mesh(m));
                self.slots[index].visibility_for = (0, 0);
            }
            self.slots[index].model = object.transform().matrix();
        }
        if material_changed {
            let material = object.material();
            let mut data = material_data(material);
            let old = std::mem::take(&mut self.slots[index].textures);
            let mut held: [Option<Arc<Texture>>; 5] = Default::default();
            let fields: [&mut u32; 5] = [
                &mut data.base_color_texture,
                &mut data.metallic_roughness_texture,
                &mut data.normal_texture,
                &mut data.occlusion_texture,
                &mut data.emissive_texture,
            ];
            let sources = [
                &material.base_color_texture,
                &material.metallic_roughness_texture,
                &material.normal_texture,
                &material.occlusion_texture,
                &material.emissive_texture,
            ];
            for (i, (field, source)) in fields.into_iter().zip(sources).enumerate() {
                if let Some(texture) = source {
                    *field = textures.require(texture)?;
                    self.textures.entry(texture.id()).or_insert_with(|| (texture.clone(), 0)).1 += 1;
                    held[i] = Some(texture.clone());
                }
            }
            self.release_textures(old);
            let slot = &mut self.slots[index];
            slot.textures = held;
            slot.material = Some(data);
            slot.blended = material.alpha_mode == AlphaMode::Blend;
            slot.double_sided = material.double_sided;
            self.mark_material(index);
        }
        self.mark_instance(index);
        Ok(())
    }

    fn grow(&mut self, slots: u32) -> Result<()> {
        let capacity = slots.next_power_of_two().max(INITIAL_SLOTS);
        self.instances = Some(StateBuffer::new(&self.device, "scene instances", u64::from(capacity) * 80, NO_MESH)?);
        self.materials = Some(StateBuffer::new(&self.device, "scene materials", u64::from(capacity) * 80, 0)?);
        self.capacity = capacity;
        if self.occlusion {
            self.history = Some(StateBuffer::new(&self.device, "scene history", u64::from(capacity) * 4, 0)?);
            self.restart_visibility(0)?;
        }
        // Everything the new buffers hold is lost: write every occupied slot again.
        for i in 0..self.slots.len() {
            if self.slots[i].occupied {
                self.mark_instance(i);
                self.mark_material(i);
            }
        }
        Ok(())
    }

    /// New, cleared visibility bits with room for twice `needed` words; every slot loses its range, and
    /// the history (which is only meaningful with them) starts over too.
    fn restart_visibility(&mut self, needed: u32) -> Result<()> {
        let words = (needed * 2).max(INITIAL_VISIBILITY_WORDS);
        self.visibility = Some(StateBuffer::new(&self.device, "meshlet visibility", u64::from(words) * 4, 0)?);
        self.visibility_used = 0;
        for i in 0..self.slots.len() {
            if self.slots[i].visibility_for != (0, 0) {
                self.slots[i].visibility_for = (0, 0);
                if self.slots[i].occupied {
                    self.mark_instance(i);
                }
            }
        }
        if self.history.is_some() {
            self.history = Some(StateBuffer::new(&self.device, "scene history", u64::from(self.capacity.max(INITIAL_SLOTS)) * 4, 0)?);
        }
        Ok(())
    }

    /// Brings the GPU's copy of `scene` up to date: finds what changed, makes the meshes and textures in use
    /// resident (their uploads are submitted ahead of the frame) and returns this frame's updates, mesh
    /// table and blended draws. `view` supplies the camera, for sorting blended objects.
    pub fn update(&mut self, scene: &Scene, view: &RenderView<'_>, meshes: &mut MeshCache, textures: &mut TextureCache) -> Result<SceneFrame> {
        self.updates += 1;
        let slot_count = scene.slot_count();
        if self.tracked.len() < slot_count {
            self.tracked.resize(slot_count, Tracked::default());
            self.slots.resize_with(slot_count, || Slot { mesh: NO_MESH, ..Slot::default() });
        }
        let mut regrown = false;
        if self.instances.is_none() || slot_count as u32 > self.capacity {
            // Tracking below marks changes; growing first means every occupied slot is rewritten.
            regrown = true;
        }

        match scene.changes(&mut self.cursor) {
            Some(indices) => {
                for index in indices {
                    match scene.object_at(index) {
                        Some(object) => self.track(index as usize, object, textures)?,
                        None => self.release(index as usize),
                    }
                }
            }
            None => {
                for (object, id) in scene.objects().iter().zip(scene.object_ids()) {
                    self.track(id.index as usize, object, textures)?;
                }
                for i in 0..self.slots.len() {
                    if self.slots[i].occupied && self.tracked[i].seen != self.updates {
                        self.release(i);
                    }
                }
            }
        }
        if regrown {
            self.grow((slot_count as u32).max(1))?;
        }

        // Meshes whose geometry changed in place reach every object using them.
        let stale: Vec<u32> = self
            .meshes
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().filter(|e| e.version != e.mesh.version()).map(|_| i as u32))
            .collect();
        for &i in &stale {
            if let Some(entry) = self.meshes[i as usize].as_mut() {
                entry.version = entry.mesh.version();
            }
            for s in 0..self.slots.len() {
                if self.slots[s].mesh == i {
                    self.slots[s].visibility_for = (0, 0);
                    self.mark_instance(s);
                }
            }
        }

        // Everything in use stays resident (the caches evict what goes unrequired for a while).
        for entry in self.meshes.iter().flatten() {
            meshes.require(&entry.mesh);
        }
        for (texture, _) in self.textures.values() {
            textures.require(texture)?;
        }
        meshes.flush()?;
        textures.flush()?;

        // The mesh table: where each distinct mesh is.
        let mut mesh_table = vec![GpuMesh::default(); self.meshes.len()];
        for (i, entry) in self.meshes.iter().enumerate() {
            let Some(entry) = entry else { continue };
            let Some(info) = meshes.get(&entry.mesh) else { continue };
            let bounds = entry.mesh.bounds();
            let center = bounds.center();
            mesh_table[i] = GpuMesh {
                sphere: center.extend((bounds.max - center).length()),
                vertices: split_address(info.vertex_address),
                meshlets: split_address(info.meshlet_address),
                meshlet_count: info.meshlet_count,
                index_count: info.index_count,
                first_index: info.first_index,
                padding: 0,
            };
        }

        // What each culled slot adds to the cull passes' capacities, and (occlusion) its visibility words.
        let mut slot_work = Vec::new();
        for i in 0..self.slots.len() {
            let slot = &self.slots[i];
            if !slot.occupied {
                continue;
            }
            let table = (slot.mesh != NO_MESH).then(|| mesh_table[slot.mesh as usize]);
            let culled = table.is_some() && !slot.blended;
            let groups = match (&table, self.meshlets) {
                (Some(t), true) => t.meshlet_count.div_ceil(32),
                (Some(_), false) => 1,
                _ => 0,
            };
            if culled != slot.counted || groups != slot.groups {
                slot_work.push((i, culled, groups));
            }
        }
        for (i, culled, groups) in slot_work {
            self.set_counted(i, culled, groups);
        }
        if self.occlusion && self.meshlets {
            self.assign_visibility(&mesh_table)?;
        }

        // The updates: only what changed.
        let mut instance_updates = Vec::with_capacity(self.instance_dirty.len());
        for &i in &self.instance_dirty {
            let slot = &mut self.slots[i as usize];
            slot.instance_dirty = false;
            let instance = if slot.occupied && slot.mesh != NO_MESH && !slot.blended {
                let cull_back = !slot.double_sided && Mat3::from_mat4(slot.model).determinant() > 0.0;
                GpuInstance {
                    model: slot.model,
                    mesh: slot.mesh,
                    flags: if cull_back { INSTANCE_CULL_BACK_FACES } else { 0 },
                    visibility_offset: slot.visibility_offset,
                    padding: 0,
                }
            } else {
                GpuInstance::default()
            };
            instance_updates.push(InstanceUpdate { instance, slot: i, padding: [0; 3] });
        }
        self.instance_dirty.clear();
        let mut material_updates = Vec::with_capacity(self.material_dirty.len());
        for &i in &self.material_dirty {
            let slot = &mut self.slots[i as usize];
            slot.material_dirty = false;
            if let Some(material) = slot.material {
                material_updates.push(MaterialUpdate { material, slot: i, padding: [0; 3] });
            }
        }
        self.material_dirty.clear();

        let (blend_draws, blend_commands, blend_runs) = self.blended(view, &mesh_table, scene);
        Ok(SceneFrame {
            mesh_table,
            instance_updates,
            material_updates,
            blend_draws,
            blend_commands,
            blend_runs,
            instance_count: slot_count as u32,
            draw_capacity: self.draw_count.next_power_of_two().max(1),
            work_capacity: if self.meshlets { self.work_total } else { self.draw_count }.next_power_of_two().max(1),
        })
    }

    /// Gives each culled slot visibility words for its mesh's meshlets, starting the bits over when full.
    fn assign_visibility(&mut self, mesh_table: &[GpuMesh]) -> Result<()> {
        let key = |this: &Self, i: usize| -> Option<((u64, u64), u32)> {
            let slot = &this.slots[i];
            if !slot.counted {
                return None;
            }
            let entry = this.meshes[slot.mesh as usize].as_ref()?;
            Some(((entry.mesh.id(), entry.mesh.version()), words_for(mesh_table[slot.mesh as usize].meshlet_count)))
        };
        let wanted: Vec<(usize, (u64, u64), u32)> = (0..self.slots.len())
            .filter_map(|i| key(self, i).filter(|(k, _)| self.slots[i].visibility_for != *k).map(|(k, w)| (i, k, w)))
            .collect();
        let needed: u32 = wanted.iter().map(|&(_, _, w)| w).sum();
        let capacity = (self.visibility.as_ref().map_or(0, |v| v.size) / 4) as u32;
        let mut wanted = wanted;
        if self.visibility.is_none() || self.visibility_used + needed > capacity {
            let in_use: u32 = (0..self.slots.len()).filter_map(|i| key(self, i)).map(|(_, w)| w).sum();
            self.restart_visibility(in_use)?;
            wanted = (0..self.slots.len()).filter_map(|i| key(self, i).map(|(k, w)| (i, k, w))).collect();
        }
        for (i, k, words) in wanted {
            let slot = &mut self.slots[i];
            slot.visibility_for = k;
            slot.visibility_offset = self.visibility_used;
            slot.visibility_words = words;
            self.visibility_used += words;
            self.mark_instance(i);
        }
        Ok(())
    }

    /// The blended objects' draws, farthest first, in runs that share a cull mode.
    fn blended(&self, view: &RenderView<'_>, mesh_table: &[GpuMesh], scene: &Scene) -> (Vec<GpuDraw>, Vec<DrawCommand>, Vec<BlendRun>) {
        let forward = scene.camera.forward();
        let mut order: Vec<(f32, usize)> = (0..self.slots.len())
            .filter(|&i| self.slots[i].occupied && self.slots[i].blended && self.slots[i].mesh != NO_MESH)
            .map(|i| {
                let slot = &self.slots[i];
                let bounds = self.meshes[slot.mesh as usize].as_ref().map_or(Vec3::ZERO, |e| e.mesh.bounds().center());
                let center = (slot.model * bounds.extend(1.0)).truncate();
                ((center - scene.camera.position).dot(forward), i)
            })
            .collect();
        // Back to front, so each blended item is composited over everything behind it. Ties keep slot order.
        order.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut draws, mut commands, mut runs) = (Vec::new(), Vec::new(), Vec::<BlendRun>::new());
        for (_, i) in order {
            let slot = &self.slots[i];
            let mesh = mesh_table[slot.mesh as usize];
            let cull_back = !slot.double_sided && Mat3::from_mat4(slot.model).determinant() > 0.0;
            let camera = (slot.model.inverse() * view.camera_position.extend(1.0)).truncate();
            let draw = draws.len() as u32;
            draws.push(GpuDraw {
                model: slot.model,
                camera_position: camera.extend(1.0),
                vertices: mesh.vertices,
                primitives: [0, 0],
                count: mesh.index_count,
                material: i as u32,
                visibility_offset: 0,
                flags: 0,
            });
            let first = commands.len() as u32;
            commands.push(DrawCommand { index_count: mesh.index_count, instance_count: 1, first_index: mesh.first_index, vertex_offset: 0, first_instance: draw });
            match runs.last_mut() {
                Some(run) if run.cull_back_faces == cull_back => run.range.count += 1,
                _ => runs.push(BlendRun { cull_back_faces: cull_back, range: CommandRange { first, count: 1 } }),
            }
        }
        (draws, commands, runs)
    }
}

