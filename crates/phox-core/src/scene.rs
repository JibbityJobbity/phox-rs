//! The scene: meshes, materials, objects and the scene's change log.

use crate::camera::Camera;
use crate::color::linear_to_srgb3;
use crate::error::{Error, Result};
use crate::math::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use crate::texture::Texture;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn next_id(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed)
}
static MESH_IDS: AtomicU64 = AtomicU64::new(1);
static OBJECT_IDS: AtomicU64 = AtomicU64::new(1);
static LOG_IDS: AtomicU64 = AtomicU64::new(1);

/// The vertex format of every Phox mesh (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    /// xyz: the direction of increasing u, perpendicular to the normal; w: +1 or -1, the handedness of the
    /// bitangent (`cross(normal, tangent) * w` points along increasing v). Leave it zero and [`Mesh`]
    /// works it out from the texture coordinates.
    pub tangent: Vec4,
}
const _: () = assert!(size_of::<Vertex>() == 48);

impl Default for Vertex {
    fn default() -> Self {
        Self { position: Vec3::ZERO, normal: Vec3::Z, uv: Vec2::ZERO, tangent: Vec4::ZERO }
    }
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self { position, normal, uv, tangent: Vec4::ZERO }
    }
}

/// An axis-aligned box.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

fn compute_bounds(vertices: &[Vertex]) -> Bounds {
    let Some(first) = vertices.first() else { return Bounds::default() };
    vertices.iter().fold(Bounds { min: first.position, max: first.position }, |b, v| Bounds {
        min: b.min.min(v.position),
        max: b.max.max(v.position),
    })
}

fn validate(vertices: &[Vertex], indices: &[u32]) -> Result<()> {
    if indices.len() % 3 != 0 {
        return Err(Error::IndexCount(indices.len()));
    }
    if let Some(&index) = indices.iter().find(|&&i| i as usize >= vertices.len()) {
        return Err(Error::IndexOutOfRange { index, vertices: vertices.len() });
    }
    Ok(())
}

/// A unit vector perpendicular to `n`.
fn any_perpendicular(n: Vec3) -> Vec3 {
    let axis = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    axis.cross(n).normalize()
}

/// Computes per-vertex tangents for normal mapping from the triangles' texture coordinates (Lengyel's
/// method: per-triangle tangents summed at each vertex, then made perpendicular to the normal). Vertices
/// whose triangles have no usable texture coordinates get an arbitrary tangent perpendicular to their normal.
pub fn generate_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut tangents = vec![Vec3::ZERO; vertices.len()];
    let mut bitangents = vec![Vec3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (vertices[tri[0] as usize], vertices[tri[1] as usize], vertices[tri[2] as usize]);
        let e1 = b.position - a.position;
        let e2 = c.position - a.position;
        let d1 = b.uv - a.uv;
        let d2 = c.uv - a.uv;
        let det = d1.x * d2.y - d2.x * d1.y;
        if det.abs() < 1e-12 {
            continue; // no usable texture mapping on this triangle
        }
        // Unnormalised, so larger triangles weigh more (area weighting).
        let tangent = (e1 * d2.y - e2 * d1.y) / det;
        let bitangent = (e2 * d1.x - e1 * d2.x) / det;
        for &i in tri {
            tangents[i as usize] += tangent;
            bitangents[i as usize] += bitangent;
        }
    }
    for (i, v) in vertices.iter_mut().enumerate() {
        let n = v.normal.normalize();
        // Gram-Schmidt: remove the part along the normal.
        let mut t = tangents[i] - n * n.dot(tangents[i]);
        let length = t.length();
        t = if length > 1e-12 && length.is_finite() { t / length } else { any_perpendicular(n) };
        let w = if n.cross(t).dot(bitangents[i]) < 0.0 { -1.0 } else { 1.0 };
        v.tangent = t.extend(w);
    }
}

/// Indexed triangles. Each Mesh has an id that is never reused and a version that changes whenever its
/// geometry does, so renderers can cache GPU copies. Cloning gives a new id; moving keeps it.
///
/// If any vertex has a zero tangent (w == 0), the mesh computes tangents for all of its vertices from
/// their positions and texture coordinates (see [`generate_tangents`]).
#[derive(Debug)]
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    bounds: Bounds,
    id: u64,
    version: u64,
}

impl Default for Mesh {
    /// An empty mesh with a fresh id (what `std::mem::take` leaves behind).
    fn default() -> Self {
        Self { vertices: Vec::new(), indices: Vec::new(), bounds: Bounds::default(), id: next_id(&MESH_IDS), version: 1 }
    }
}

impl Clone for Mesh {
    fn clone(&self) -> Self {
        Self { vertices: self.vertices.clone(), indices: self.indices.clone(), bounds: self.bounds, id: next_id(&MESH_IDS), version: 1 }
    }
}

impl Mesh {
    /// Fails if the index count is not a multiple of 3 or an index is out of range.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Result<Self> {
        let mut mesh = Self::default();
        mesh.set_geometry(vertices, indices)?;
        mesh.version = 1;
        Ok(mesh)
    }

    /// For the built-in shapes, whose indices are valid by construction.
    fn built(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        debug_assert!(validate(&vertices, &indices).is_ok());
        let mut mesh = Self::default();
        mesh.install(vertices, indices);
        mesh.version = 1;
        mesh
    }

    fn install(&mut self, mut vertices: Vec<Vertex>, indices: Vec<u32>) {
        if vertices.iter().any(|v| v.tangent.w == 0.0) {
            generate_tangents(&mut vertices, &indices);
        }
        self.bounds = compute_bounds(&vertices);
        self.vertices = vertices;
        self.indices = indices;
        self.version += 1;
    }

    /// Replaces the geometry (same checks as [`Mesh::new`]) and bumps the version.
    pub fn set_geometry(&mut self, vertices: Vec<Vertex>, indices: Vec<u32>) -> Result<()> {
        validate(&vertices, &indices)?;
        self.install(vertices, indices);
        Ok(())
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn version(&self) -> u64 {
        self.version
    }

    /// An equilateral triangle centred on the origin, facing +Z, with circumradius `radius`.
    pub fn triangle(radius: f32) -> Self {
        let s = 3f32.sqrt() * 0.5 * radius;
        let n = Vec3::Z;
        // Counter-clockwise seen from +Z.
        Self::built(
            vec![
                Vertex::new(Vec3::new(0.0, radius, 0.0), n, Vec2::new(0.5, 0.0)),
                Vertex::new(Vec3::new(-s, -0.5 * radius, 0.0), n, Vec2::new(0.0, 1.0)),
                Vertex::new(Vec3::new(s, -0.5 * radius, 0.0), n, Vec2::new(1.0, 1.0)),
            ],
            vec![0, 1, 2],
        )
    }

    /// A cube centred on the origin with flat-shaded faces (24 vertices).
    pub fn cube(size: f32) -> Self {
        // u x v == normal, so corners in u/v order wind counter-clockwise from outside.
        let faces: [(Vec3, Vec3, Vec3); 6] = [
            (Vec3::X, Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::X, Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
        ];
        let h = 0.5 * size;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (normal, u, v) in faces {
            let base = vertices.len() as u32;
            for c in [Vec2::new(-1.0, -1.0), Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::new(-1.0, 1.0)] {
                vertices.push(Vertex::new((normal + c.x * u + c.y * v) * h, normal, Vec2::new(0.5 * (c.x + 1.0), 0.5 * (1.0 - c.y))));
            }
            indices.extend([0u32, 1, 2, 0, 2, 3].map(|i| base + i));
        }
        Self::built(vertices, indices)
    }

    /// A square in the XZ plane facing +Y.
    pub fn plane(size: f32) -> Self {
        let h = 0.5 * size;
        let up = Vec3::Y;
        Self::built(
            vec![
                Vertex::new(Vec3::new(-h, 0.0, h), up, Vec2::new(0.0, 1.0)),
                Vertex::new(Vec3::new(h, 0.0, h), up, Vec2::new(1.0, 1.0)),
                Vertex::new(Vec3::new(h, 0.0, -h), up, Vec2::new(1.0, 0.0)),
                Vertex::new(Vec3::new(-h, 0.0, -h), up, Vec2::new(0.0, 0.0)),
            ],
            vec![0, 1, 2, 0, 2, 3],
        )
    }

    /// A torus about the Y axis with `rings` segments around that axis and `sides` around the tube.
    pub fn torus(radius: f32, tube_radius: f32, rings: u32, sides: u32) -> Self {
        let rings = rings.max(3);
        let sides = sides.max(3);
        let mut vertices = Vec::with_capacity((rings as usize + 1) * (sides as usize + 1));
        // The seam vertices are duplicated so texture coordinates can wrap. Their positions repeat the first
        // ring's exactly (angles from the wrapped index, not 2 pi), so the seam is closed for tools that match
        // vertices by position, such as the cluster LOD builder.
        for i in 0..=rings {
            let u = i as f32 / rings as f32;
            let theta = (i % rings) as f32 / rings as f32 * crate::math::TWO_PI;
            let around = Vec3::new(theta.cos(), 0.0, theta.sin());
            for j in 0..=sides {
                let v = j as f32 / sides as f32;
                let phi = (j % sides) as f32 / sides as f32 * crate::math::TWO_PI;
                let normal = phi.cos() * around + Vec3::new(0.0, phi.sin(), 0.0);
                vertices.push(Vertex::new(radius * around + tube_radius * normal, normal, Vec2::new(u, v)));
            }
        }
        let at = |i: u32, j: u32| i * (sides + 1) + j;
        let mut indices = Vec::with_capacity(rings as usize * sides as usize * 6);
        for i in 0..rings {
            for j in 0..sides {
                // Counter-clockwise seen from outside the tube.
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                indices.extend([a, d, c, a, c, b]);
            }
        }
        Self::built(vertices, indices)
    }

    /// A UV sphere centred on the origin, poles on the Y axis; u runs once around, v from top to bottom.
    pub fn sphere(radius: f32, segments: u32, rings: u32) -> Self {
        let segments = segments.max(3);
        let rings = rings.max(2);
        let mut vertices = Vec::with_capacity((segments as usize + 1) * (rings as usize + 1));
        // Seam and pole vertices are duplicated so each gets its own texture coordinate. Their positions are
        // exactly equal (wrapped angles, exact poles), so the surface is closed for tools that match vertices
        // by position.
        for j in 0..=rings {
            let v = j as f32 / rings as f32;
            let phi = v * crate::math::PI; // from +Y down to -Y
            let pole = j == 0 || j == rings;
            let sin_phi = if pole { 0.0 } else { phi.sin() };
            let cos_phi = if j == 0 { 1.0 } else if j == rings { -1.0 } else { phi.cos() };
            for i in 0..=segments {
                let u = i as f32 / segments as f32;
                let theta = (i % segments) as f32 / segments as f32 * crate::math::TWO_PI;
                let normal = Vec3::new(sin_phi * theta.cos(), cos_phi, -sin_phi * theta.sin());
                // The tangent is d(position)/d(theta), normalised. Increasing v points down, against
                // cross(normal, tangent), hence the handedness of -1.
                let tangent = Vec3::new(-theta.sin(), 0.0, -theta.cos());
                vertices.push(Vertex { position: radius * normal, normal, uv: Vec2::new(u, v), tangent: tangent.extend(-1.0) });
            }
        }
        let at = |i: u32, j: u32| j * (segments + 1) + i;
        let mut indices = Vec::with_capacity(segments as usize * rings as usize * 6);
        for j in 0..rings {
            for i in 0..segments {
                // Counter-clockwise seen from outside; the degenerate triangles at the poles are skipped.
                let (a, b, c, d) = (at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j));
                if j != 0 {
                    indices.extend([a, b, d]);
                }
                if j != rings - 1 {
                    indices.extend([d, b, c]);
                }
            }
        }
        Self::built(vertices, indices)
    }
}

/// How a material's alpha is used (glTF's alphaMode).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Alpha is ignored.
    #[default]
    Opaque,
    /// Fragments with alpha below `alpha_cutoff` are discarded, the rest are opaque.
    Mask,
    /// Blended over what is behind, after the opaque objects and back to front; no depth writes.
    Blend,
}

/// A metallic-roughness material, as in glTF 2.0. Each texture is optional; where present it is
/// multiplied by its factor. Colours are linear (see [`crate::color`]).
#[derive(Clone, Debug)]
pub struct Material {
    /// Optional, for debuggers and tools.
    pub name: String,
    /// Base colour factor; rgb linear, a: opacity.
    pub albedo: Vec4,
    /// 0: dielectric, 1: metal.
    pub metallic: f32,
    /// Perceptual roughness: 0 mirror-like, 1 fully rough.
    pub roughness: f32,
    /// Emitted light, linear; added after lighting.
    pub emissive: Vec3,
    /// Scales the normal map's x and y.
    pub normal_scale: f32,
    /// 0: occlusion texture ignored, 1: fully applied.
    pub occlusion_strength: f32,
    pub alpha_mode: AlphaMode,
    /// `AlphaMode::Mask` only.
    pub alpha_cutoff: f32,
    /// Both faces of each triangle are drawn (and lit). Turn it off for closed meshes seen from outside:
    /// their back faces are then culled, and the mesh shading path skips whole meshlets that face away.
    pub double_sided: bool,
    /// rgb: colour (sRGB), a: opacity.
    pub base_color_texture: Option<Arc<Texture>>,
    /// g: roughness, b: metallic.
    pub metallic_roughness_texture: Option<Arc<Texture>>,
    /// Tangent-space normal.
    pub normal_texture: Option<Arc<Texture>>,
    /// r: ambient occlusion.
    pub occlusion_texture: Option<Arc<Texture>>,
    /// rgb: emitted colour (sRGB).
    pub emissive_texture: Option<Arc<Texture>>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            albedo: Vec4::new(1.0, 0.2, 0.4, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3::ZERO,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: true,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
        }
    }
}

impl Material {
    /// Every texture the material uses, base colour first; `None` for those it does not.
    pub fn textures(&self) -> [Option<&Texture>; 5] {
        [
            self.base_color_texture.as_deref(),
            self.metallic_roughness_texture.as_deref(),
            self.normal_texture.as_deref(),
            self.occlusion_texture.as_deref(),
            self.emissive_texture.as_deref(),
        ]
    }
}

/// A Scene's record of which objects changed, by `ObjectId::index`: [`Scene::changes`] reads it. Objects
/// append themselves when a `*_mut()` guard is dropped, at most once until a reader has read past them.
#[derive(Debug)]
pub struct SceneChangeLog {
    id: u64,
    entries: Vec<u32>,
    /// The position of `entries[0]`.
    first: u64,
    /// The furthest position a reader has read to.
    read: u64,
    /// Past this many entries the log starts over (the scene sets it).
    limit: usize,
}

impl SceneChangeLog {
    fn new() -> Self {
        Self { id: next_id(&LOG_IDS), entries: Vec::new(), first: 0, read: 0, limit: 1024 }
    }

    /// Logs object slot `index`, unless it is logged and unread. `logged_at` is the object's own record of
    /// where (one past its last entry; 0: never).
    fn record(&mut self, index: u32, logged_at: &mut u64) {
        if *logged_at > self.read {
            return; // its entry has not been read yet
        }
        if self.entries.len() >= self.limit {
            // Start over: readers behind this point look at every object instead (Scene::changes).
            self.first += self.entries.len() as u64;
            self.entries.clear();
            self.read = self.first;
        }
        self.entries.push(index);
        *logged_at = self.first + self.entries.len() as u64;
    }

    pub fn id(&self) -> u64 {
        self.id
    }
}

type SharedLog = Arc<Mutex<SceneChangeLog>>;

fn lock(log: &SharedLog) -> std::sync::MutexGuard<'_, SceneChangeLog> {
    log.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Scale, then rotate, then translate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self { position: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl Transform {
    pub fn at(position: Vec3) -> Self {
        Self { position, ..Self::default() }
    }

    /// translate * rotate * scale, composed directly: the rotation's columns scaled, then the translation.
    pub fn matrix(&self) -> Mat4 {
        let rotate = Mat3::from_quat(self.rotation);
        Mat4::from_cols(
            (rotate.x_axis * self.scale.x).extend(0.0),
            (rotate.y_axis * self.scale.y).extend(0.0),
            (rotate.z_axis * self.scale.z).extend(0.0),
            self.position.extend(1.0),
        )
    }

    /// The transform whose `matrix()` is `m`, for affine matrices without shear (shear is dropped). A
    /// mirroring matrix gets a negative x scale.
    pub fn from_matrix(m: &Mat4) -> Self {
        let mut basis = Mat3::from_mat4(*m);
        let mut scale = Vec3::new(basis.x_axis.length(), basis.y_axis.length(), basis.z_axis.length());
        if basis.determinant() < 0.0 {
            scale.x = -scale.x; // a rotation cannot mirror, so the scale does
        }
        for (i, axis) in [&mut basis.x_axis, &mut basis.y_axis, &mut basis.z_axis].into_iter().enumerate() {
            if scale[i] != 0.0 {
                *axis /= scale[i];
            }
        }
        Self { position: m.w_axis.truncate(), rotation: Quat::from_mat3(&basis).normalize(), scale }
    }
}

/// Describes an object, for [`Scene::add_object`] and [`Scene::add_objects`]. Name the fields that matter
/// and leave the rest: `ObjectDesc { mesh: Some(cube), ..Default::default() }`.
#[derive(Clone, Debug, Default)]
pub struct ObjectDesc {
    /// `None`: the object draws nothing.
    pub mesh: Option<Arc<Mesh>>,
    pub material: Material,
    pub transform: Transform,
    /// Optional, for debuggers, logs and tools. Nothing looks an object up by it: an object's identity is
    /// the [`ObjectId`] its scene hands out.
    pub name: String,
}

/// A mesh placed in a scene with a material. Each object has an id that is never reused, so renderers can
/// keep per-object state across frames. Cloning gives a new id; moving keeps it.
///
/// Renderers keep copies of what they drew (the GPU renderer keeps every object on the GPU) and find what
/// changed through two revisions: `transform_mut()` and `set_mesh()` bump `transform_revision()`,
/// `material_mut()` bumps `material_revision()`. The `*_mut()` accessors return guards that record the
/// change in the scene's change log when they are dropped, so there is no way to change an object without
/// the renderer hearing of it.
#[derive(Debug)]
pub struct Object {
    id: u64,
    transform_revision: u64,
    material_revision: u64,
    // In a Scene: its change log, the object's ObjectId::index, and where it last logged itself.
    log: Option<SharedLog>,
    index: u32,
    logged_at: u64,
    name: String,
    mesh: Option<Arc<Mesh>>,
    material: Material,
    transform: Transform,
}

impl Clone for Object {
    /// A copy is a different object: it gets its own id and is not in any scene.
    fn clone(&self) -> Self {
        Self {
            id: next_id(&OBJECT_IDS),
            transform_revision: 1,
            material_revision: 1,
            log: None,
            index: 0,
            logged_at: 0,
            name: self.name.clone(),
            mesh: self.mesh.clone(),
            material: self.material.clone(),
            transform: self.transform,
        }
    }
}

/// Changes an object's material; logs the change when dropped.
pub struct MaterialMut<'a>(&'a mut Object);
/// Changes an object's transform; logs the change when dropped.
pub struct TransformMut<'a>(&'a mut Object);

impl Deref for MaterialMut<'_> {
    type Target = Material;
    fn deref(&self) -> &Material {
        &self.0.material
    }
}
impl DerefMut for MaterialMut<'_> {
    fn deref_mut(&mut self) -> &mut Material {
        &mut self.0.material
    }
}
impl Drop for MaterialMut<'_> {
    fn drop(&mut self) {
        self.0.material_revision += 1;
        self.0.log_change();
    }
}
impl Deref for TransformMut<'_> {
    type Target = Transform;
    fn deref(&self) -> &Transform {
        &self.0.transform
    }
}
impl DerefMut for TransformMut<'_> {
    fn deref_mut(&mut self) -> &mut Transform {
        &mut self.0.transform
    }
}
impl Drop for TransformMut<'_> {
    fn drop(&mut self) {
        self.0.transform_revision += 1;
        self.0.log_change();
    }
}

impl Object {
    pub fn new(desc: ObjectDesc) -> Self {
        Self {
            id: next_id(&OBJECT_IDS),
            transform_revision: 1,
            material_revision: 1,
            log: None,
            index: 0,
            logged_at: 0,
            name: desc.name,
            mesh: desc.mesh,
            material: desc.material,
            transform: desc.transform,
        }
    }

    fn log_change(&mut self) {
        if let Some(log) = &self.log {
            lock(log).record(self.index, &mut self.logged_at);
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }
    /// A label for debuggers and tools; may be empty, and need not be unique.
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn mesh(&self) -> Option<&Arc<Mesh>> {
        self.mesh.as_ref()
    }
    pub fn set_mesh(&mut self, mesh: Option<Arc<Mesh>>) {
        self.mesh = mesh;
        self.transform_revision += 1;
        self.log_change();
    }
    pub fn material(&self) -> &Material {
        &self.material
    }
    /// For changing the material: bumps `material_revision()` and logs the change when the guard drops.
    pub fn material_mut(&mut self) -> MaterialMut<'_> {
        MaterialMut(self)
    }
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    /// For changing the transform: bumps `transform_revision()` and logs the change when the guard drops.
    pub fn transform_mut(&mut self) -> TransformMut<'_> {
        TransformMut(self)
    }
    /// Changes whenever the transform or mesh may have.
    pub fn transform_revision(&self) -> u64 {
        self.transform_revision
    }
    /// Changes whenever the material may have.
    pub fn material_revision(&self) -> u64 {
        self.material_revision
    }

    /// Makes this object look like `other` (name, mesh, material, transform) while keeping its own id and
    /// its place in a scene, which is told of the change. The counterpart of C++'s assignment operator.
    pub fn assign_from(&mut self, other: &Object) {
        self.name = other.name.clone();
        self.mesh = other.mesh.clone();
        self.material = other.material.clone();
        self.transform = other.transform;
        self.transform_revision += 1;
        self.material_revision += 1;
        self.log_change();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// The way the light travels (need not be normalised).
    pub direction: Vec3,
    pub color: Vec3,
    /// Light that reaches every surface regardless of direction, as a fraction of the light colour (a
    /// stand-in for environment lighting).
    pub ambient: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self { direction: Vec3::new(-0.4, -1.0, -0.6), color: Vec3::ONE, ambient: 0.25 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ToneMapOperator {
    /// Exposure, then clamp to [0, 1]; colours already in range pass through unchanged.
    #[default]
    Clamp,
    /// c / (1 + c)
    Reinhard,
    /// Narkowicz's fit of the ACES filmic curve.
    Aces,
}

/// How the renderer maps scene colours (which may exceed 1) to the display range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneMapping {
    pub op: ToneMapOperator,
    /// Colours are multiplied by this first.
    pub exposure: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self { op: ToneMapOperator::Clamp, exposure: 1.0 }
    }
}

/// Applies `mapping` to a linear colour and encodes the result for the display with the sRGB transfer
/// function, giving values in [0, 1]. The tonemap shader does the same on the GPU.
pub fn apply_tone_mapping(color: Vec3, mapping: &ToneMapping) -> Vec3 {
    let mut c = color * mapping.exposure;
    match mapping.op {
        ToneMapOperator::Clamp => {}
        ToneMapOperator::Reinhard => c /= Vec3::ONE + c,
        ToneMapOperator::Aces => c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14),
    }
    linear_to_srgb3(c.clamp(Vec3::ZERO, Vec3::ONE))
}

/// Names an object in a Scene. It stays valid until that object is removed; after that the scene rejects it,
/// even once the slot holds a new object (the generation differs). A copy of a scene keeps its ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectId {
    pub index: u32,
    pub generation: u32,
}

impl Default for ObjectId {
    fn default() -> Self {
        Self { index: u32::MAX, generation: 0 }
    }
}

impl ObjectId {
    /// Whether the id was ever handed out (a default id is not). Use [`Scene::contains`] to ask whether
    /// the object still exists.
    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

/// Where a reader of [`Scene::changes`] has got to.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChangeCursor {
    /// `SceneChangeLog::id` of the scene read; 0: none yet.
    pub log: u64,
    pub position: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    /// The object's position in `objects`, while the slot is in use.
    dense: u32,
    /// Bumped on removal, so ids into the old object go stale.
    generation: u32,
}

/// What to draw and how to look at it. Plain data: renderers read it and never keep references past a
/// render call.
///
/// Objects are stored contiguously. Keep `ObjectId`s, not references: those are only valid until the next
/// `add_object` or `remove_object`.
#[derive(Debug)]
pub struct Scene {
    objects: Vec<Object>,
    ids: Vec<ObjectId>, // parallel to objects
    slots: Vec<Slot>,   // indexed by ObjectId::index
    free_slots: Vec<u32>,
    log: SharedLog,
    pub camera: Camera,
    pub light: DirectionalLight,
    /// Linear; sRGB (0.1, 0.2, 0.3).
    pub clear_color: Vec4,
    pub tone_mapping: ToneMapping,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            ids: Vec::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            log: Arc::new(Mutex::new(SceneChangeLog::new())),
            camera: Camera::default(),
            light: DirectionalLight::default(),
            clear_color: Vec4::new(0.0100228, 0.0331048, 0.0732389, 1.0),
            tone_mapping: ToneMapping::default(),
        }
    }
}

impl Clone for Scene {
    /// A copy keeps the ObjectIds; its objects are copies, so they get new `Object::id`s, and it has a
    /// change log of its own.
    fn clone(&self) -> Self {
        let mut copy = Self {
            objects: self.objects.clone(),
            ids: self.ids.clone(),
            slots: self.slots.clone(),
            free_slots: self.free_slots.clone(),
            log: Arc::new(Mutex::new(SceneChangeLog::new())),
            camera: self.camera.clone(),
            light: self.light,
            clear_color: self.clear_color,
            tone_mapping: self.tone_mapping,
        };
        for (object, id) in copy.objects.iter_mut().zip(&copy.ids) {
            object.log = Some(copy.log.clone());
            object.index = id.index;
        }
        lock(&copy.log).limit = 2 * copy.objects.len() + 1024;
        copy
    }
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the object `desc` describes and returns the id that names it from then on.
    pub fn add_object(&mut self, desc: ObjectDesc) -> Result<ObjectId> {
        let index = if let Some(index) = self.free_slots.pop() {
            index
        } else {
            if self.slots.len() >= u32::MAX as usize {
                return Err(Error::TooManyObjects);
            }
            self.slots.push(Slot::default());
            (self.slots.len() - 1) as u32
        };
        let id = ObjectId { index, generation: self.slots[index as usize].generation };
        let mut object = Object::new(desc);
        object.log = Some(self.log.clone());
        object.index = index;
        self.objects.push(object);
        self.ids.push(id);
        self.slots[index as usize].dense = (self.objects.len() - 1) as u32;
        lock(&self.log).limit = 2 * self.objects.len() + 1024;
        self.objects.last_mut().expect("just pushed").log_change();
        Ok(id)
    }

    /// Adds several objects, in order, and returns their ids in the same order, to be taken apart by a
    /// destructuring `let [a, b] = scene.add_objects([..])?;`.
    pub fn add_objects<const N: usize>(&mut self, descs: [ObjectDesc; N]) -> Result<[ObjectId; N]> {
        let mut ids = [ObjectId::default(); N];
        for (slot, desc) in ids.iter_mut().zip(descs) {
            *slot = self.add_object(desc)?;
        }
        Ok(ids)
    }

    /// Removes the object; the order of the remaining objects may change. Returns false if `id` is stale.
    pub fn remove_object(&mut self, id: ObjectId) -> bool {
        if !self.contains(id) {
            return false;
        }
        let dense = self.slots[id.index as usize].dense as usize;
        // Fill the hole with the last object (which keeps its id) and point its slot at the new position.
        self.objects.swap_remove(dense);
        self.ids.swap_remove(dense);
        if dense < self.ids.len() {
            self.slots[self.ids[dense].index as usize].dense = dense as u32;
        }
        let mut logged_at = 0; // the object is gone: log its index regardless
        lock(&self.log).record(id.index, &mut logged_at);
        // A slot whose generation would wrap is retired rather than reused, so no id ever becomes valid again.
        let slot = &mut self.slots[id.index as usize];
        if slot.generation != u32::MAX {
            slot.generation += 1;
            self.free_slots.push(id.index);
        }
        true
    }

    pub fn contains(&self, id: ObjectId) -> bool {
        self.lookup(id).is_some()
    }

    fn lookup(&self, id: ObjectId) -> Option<usize> {
        let slot = self.slots.get(id.index as usize)?;
        // A free slot has already moved on to the next generation, so the check also rejects it.
        let dense = slot.dense as usize;
        (slot.generation == id.generation && self.ids.get(dense) == Some(&id)).then_some(dense)
    }

    /// The object, or `None` if `id` is stale.
    pub fn get(&self, id: ObjectId) -> Option<&Object> {
        self.lookup(id).map(|d| &self.objects[d])
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut Object> {
        self.lookup(id).map(|d| &mut self.objects[d])
    }

    /// The object; fails if `id` is stale.
    pub fn at(&self, id: ObjectId) -> Result<&Object> {
        self.get(id).ok_or(Error::StaleObject { index: id.index, generation: id.generation })
    }

    pub fn at_mut(&mut self, id: ObjectId) -> Result<&mut Object> {
        self.get_mut(id).ok_or(Error::StaleObject { index: id.index, generation: id.generation })
    }

    /// The object whose `ObjectId::index` is `index`, or `None` if there is none.
    pub fn object_at(&self, index: u32) -> Option<&Object> {
        let slot = self.slots.get(index as usize)?;
        let dense = slot.dense as usize;
        (self.ids.get(dense) == Some(&ObjectId { index, generation: slot.generation })).then(|| &self.objects[dense])
    }

    /// The `ObjectId::index` of every object added, removed or changed through its `*_mut()` guards (see
    /// [`Object`]) since `cursor` last read this scene, and moves `cursor` to now. Indices may repeat, and an
    /// index may name an object since removed (`object_at` is `None`) or replaced. `None` when the scene
    /// cannot tell: `cursor` has not read this scene, or the entries it needs were dropped (the log starts
    /// over once it outgrows the scene); then look at every object. For renderers that keep copies of the
    /// objects.
    pub fn changes(&self, cursor: &mut ChangeCursor) -> Option<Vec<u32>> {
        let mut log = lock(&self.log);
        let end = log.first + log.entries.len() as u64;
        let known = cursor.log == log.id && cursor.position >= log.first && cursor.position <= end;
        let from = cursor.position;
        *cursor = ChangeCursor { log: log.id, position: end };
        log.read = log.read.max(end);
        known.then(|| log.entries[(from - log.first) as usize..].to_vec())
    }

    /// Every object, in no particular order. `object_ids()[i]` is the id of `objects()[i]`.
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }
    pub fn objects_mut(&mut self) -> &mut [Object] {
        &mut self.objects
    }
    pub fn object_ids(&self) -> &[ObjectId] {
        &self.ids
    }
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// One more than the largest `ObjectId::index` ever handed out: how many slots per-object state indexed by
    /// `ObjectId::index` needs.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}
