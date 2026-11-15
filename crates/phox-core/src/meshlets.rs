//! Meshlets, their culling bounds and the cluster LOD hierarchy (as in Unreal's Nanite), built with
//! meshoptimizer 1.3; plus the frustum and sphere helpers the culling shaders mirror.

use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::meshopt::{self, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_LOCK_BORDER, SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK};
use crate::resources::Extent2D;
use crate::scene::Mesh;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

/// Meshlet size limits. 64 vertices and 124 triangles suit both NVIDIA's and AMD's mesh shading hardware;
/// shaders/forward.mesh declares the same maximum outputs.
pub const MESHLET_MAX_VERTICES: u32 = 64;
pub const MESHLET_MAX_TRIANGLES: u32 = 124;

/// `Meshlet::parent_error` of a meshlet that has no coarser version: it is drawn however small it gets.
pub const LOD_NO_PARENT: f32 = 3.0e38;

/// A cluster of up to [`MESHLET_MAX_TRIANGLES`] triangles of one mesh, with the bounds that let a task
/// shader cull it as a whole, and its place in the mesh's cluster LOD hierarchy. Laid out as the shaders
/// read it (std430, 112 bytes), so arrays of these upload as they are.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Meshlet {
    /// xyz: centre of a sphere enclosing the meshlet (object space); w: its radius.
    pub sphere: Vec4,
    /// xyz: axis of a cone containing every triangle normal; w: cutoff. The meshlet faces away from a camera
    /// at c (object space) if `dot(normalize(cone_apex - c), axis) >= cutoff`. Above 1 when no cone fits.
    pub cone: Vec4,
    /// xyz; w unused.
    pub cone_apex: Vec4,
    /// Where `lod_error` is measured: the bounds of the group whose simplification made this meshlet.
    pub lod_sphere: Vec4,
    /// The same for the coarser version of this meshlet.
    pub parent_sphere: Vec4,
    /// Object-space error of this meshlet's geometry; 0 for the original.
    pub lod_error: f32,
    /// The error of its coarser version; never below `lod_error`.
    pub parent_error: f32,
    /// 0: the original triangles; each level about halves them.
    pub level: u32,
    pub padding: u32,
    /// First entry of `MeshletMesh::data` holding this meshlet's vertex indices.
    pub vertex_offset: u32,
    /// First entry of `MeshletMesh::data` holding its triangles (see there).
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}
const _: () = assert!(size_of::<Meshlet>() == 112);

impl Default for Meshlet {
    fn default() -> Self {
        Self {
            sphere: Vec4::ZERO,
            cone: Vec4::new(0.0, 0.0, 1.0, 2.0),
            cone_apex: Vec4::ZERO,
            lod_sphere: Vec4::ZERO,
            parent_sphere: Vec4::ZERO,
            lod_error: 0.0,
            parent_error: LOD_NO_PARENT,
            level: 0,
            padding: 0,
            vertex_offset: 0,
            triangle_offset: 0,
            vertex_count: 0,
            triangle_count: 0,
        }
    }
}

/// A mesh split into meshlets. `data` holds, for each meshlet, `vertex_count` indices into the mesh's
/// vertices, then `triangle_count` triangles packed one per entry as three 8-bit indices into that list
/// (bits 0-7, 8-15, 16-23), wound like the mesh's own triangles.
#[derive(Clone, Debug)]
pub struct MeshletMesh {
    pub meshlets: Vec<Meshlet>,
    pub data: Vec<u32>,
    /// LOD levels: 1 + the highest `Meshlet::level`.
    pub levels: u32,
}

impl Default for MeshletMesh {
    fn default() -> Self {
        Self { meshlets: Vec::new(), data: Vec::new(), levels: 1 }
    }
}

impl MeshletMesh {
    /// Triangles of all meshlets (of every level).
    pub fn triangle_count(&self) -> usize {
        self.meshlets.iter().map(|m| m.triangle_count as usize).sum()
    }

    /// Triangles of the meshlets of `level`.
    pub fn triangle_count_at(&self, level: u32) -> usize {
        self.meshlets.iter().filter(|m| m.level == level).map(|m| m.triangle_count as usize).sum()
    }
}

/// `mesh`'s indices reordered for the GPU's post-transform vertex cache (meshoptimizer). The same triangles
/// with the same winding, in a different order.
pub fn optimize_index_order(mesh: &Mesh) -> Vec<u32> {
    meshopt::optimize_vertex_cache(mesh.indices(), mesh.vertices().len())
}

/// A meshlet while the hierarchy is built: its triangles as indices into the mesh's vertices, and the LOD
/// fields that end up in its [`Meshlet`].
#[derive(Clone, Debug)]
struct Cluster {
    indices: Vec<u32>,
    lod_sphere: Vec4,
    lod_error: f32,
    parent_sphere: Vec4,
    parent_error: f32,
    level: u32,
}

/// Splits the triangles in `indices` into clusters of at most [`MESHLET_MAX_VERTICES`] vertices and
/// [`MESHLET_MAX_TRIANGLES`] triangles, keeping close triangles together. The triangles' vertices are first
/// copied into a compact array, so the work depends on the triangles given, not on the whole mesh.
fn clusterize(mesh: &Mesh, indices: &[u32], level: u32, out: &mut Vec<Cluster>) {
    let mut global: Vec<u32> = Vec::new(); // local vertex -> mesh vertex
    let mut to_local: HashMap<u32, u32> = HashMap::with_capacity(indices.len());
    let local: Vec<u32> = indices
        .iter()
        .map(|&i| {
            *to_local.entry(i).or_insert_with(|| {
                global.push(i);
                (global.len() - 1) as u32
            })
        })
        .collect();
    let positions: Vec<Vec3> = global.iter().map(|&g| mesh.vertices()[g as usize].position).collect();

    // A cone weight of 0.25 trades a little vertex reuse for tighter normal cones, so more meshlets can be
    // culled as back-facing.
    let (meshlets, meshlet_vertices, meshlet_triangles) =
        meshopt::build_meshlets(&local, &positions, MESHLET_MAX_VERTICES as usize, MESHLET_MAX_TRIANGLES as usize, 0.25);
    for m in &meshlets {
        let count = m.triangle_count as usize * 3;
        let triangle_offset = m.triangle_offset as usize;
        let vertex_offset = m.vertex_offset as usize;
        let indices = (0..count)
            .map(|i| global[meshlet_vertices[vertex_offset + meshlet_triangles[triangle_offset + i] as usize] as usize])
            .collect();
        out.push(Cluster { indices, lod_sphere: Vec4::ZERO, lod_error: 0.0, parent_sphere: Vec4::ZERO, parent_error: LOD_NO_PARENT, level });
    }
}

/// Appends `cluster` to `result` as a [`Meshlet`]: local vertex and triangle lists ordered for the GPU,
/// culling bounds from its geometry, and its LOD fields.
fn append_meshlet(mesh: &Mesh, cluster: &Cluster, result: &mut MeshletMesh) {
    let triangle_count = cluster.indices.len() / 3;
    let (mut local_vertices, mut local_triangles, vertex_count) = meshopt::extract_meshlet_indices(&cluster.indices);
    meshopt::optimize_meshlet(&mut local_vertices, &mut local_triangles, triangle_count, vertex_count);
    let bounds = meshopt::compute_meshlet_bounds(&local_vertices, &local_triangles, triangle_count, mesh.vertices());
    // meshoptimizer reports "no usable cone" as a cutoff of exactly 1, which rounding could still satisfy;
    // anything above 1 never is.
    let cutoff = if bounds.cone_cutoff >= 1.0 { 2.0 } else { bounds.cone_cutoff };
    let mut meshlet = Meshlet {
        sphere: Vec4::new(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius),
        cone: Vec4::new(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2], cutoff),
        cone_apex: Vec4::new(bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2], 0.0),
        lod_sphere: cluster.lod_sphere,
        lod_error: cluster.lod_error,
        parent_sphere: cluster.parent_sphere,
        parent_error: cluster.parent_error,
        level: cluster.level,
        vertex_offset: result.data.len() as u32,
        vertex_count: vertex_count as u32,
        ..Meshlet::default()
    };
    result.data.extend_from_slice(&local_vertices[..vertex_count]);
    meshlet.triangle_offset = result.data.len() as u32;
    meshlet.triangle_count = triangle_count as u32;
    for t in local_triangles[..triangle_count * 3].chunks_exact(3) {
        result.data.push(t[0] as u32 | (t[1] as u32) << 8 | (t[2] as u32) << 16);
    }
    result.meshlets.push(meshlet);
    result.levels = result.levels.max(cluster.level + 1);
}

fn cluster_sphere(mesh: &Mesh, indices: &[u32]) -> Vec4 {
    let b = meshopt::compute_cluster_bounds(indices, mesh.vertices());
    Vec4::new(b.center[0], b.center[1], b.center[2], b.radius)
}

/// Groups of clusters that share many vertices, about `group_size` each, as lists of indices into `all`.
/// Connectivity is by position (`remap`), so seams do not separate neighbours.
fn partition(mesh: &Mesh, all: &[Cluster], clusters: &[usize], remap: &[u32], group_size: usize) -> Vec<Vec<usize>> {
    if clusters.len() <= group_size {
        return vec![clusters.to_vec()];
    }
    let mut indices = Vec::new();
    let mut counts = Vec::with_capacity(clusters.len());
    for &c in clusters {
        counts.push(all[c].indices.len() as u32);
        indices.extend(all[c].indices.iter().map(|&v| remap[v as usize]));
    }
    let (part, group_count) = meshopt::partition_clusters(&indices, &counts, mesh.vertices(), group_size);
    let mut groups = vec![Vec::new(); group_count];
    for (i, &c) in clusters.iter().enumerate() {
        groups[part[i] as usize].push(c);
    }
    groups
}

/// Splits the triangles in `indices` (of `mesh`'s vertices) into meshlets with culling bounds, keeping
/// spatially close triangles together, and orders each meshlet's vertices and triangles for locality. Pass
/// the output of [`optimize_index_order`] for the best clusters.
pub fn build_meshlets(mesh: &Mesh, indices: &[u32]) -> MeshletMesh {
    let mut clusters = Vec::new();
    clusterize(mesh, indices, 0, &mut clusters);
    let mut result = MeshletMesh::default();
    result.meshlets.reserve(clusters.len());
    for cluster in &mut clusters {
        cluster.lod_sphere = cluster_sphere(mesh, &cluster.indices);
        append_meshlet(mesh, cluster, &mut result);
    }
    result
}

/// Splits `mesh` into a cluster LOD hierarchy (as in Unreal's Nanite): level 0 is [`build_meshlets`] on
/// `indices`; then, until nothing simplifies further, neighbouring meshlets are grouped (about 16 per
/// group), each group is merged and simplified to about half its triangles with the vertices it shares with
/// other groups locked, and the result is split into meshlets again. Grouping differs from level to level,
/// so the borders locked at one level are simplified at the next.
///
/// Every level uses the mesh's own vertices; only meshlet data is added (about as much again as level 0).
/// Each meshlet records the error and bounds of its own group's simplification and of the one above it;
/// [`lod_selected`] then picks, for any camera, a set of meshlets that covers the surface once without cracks.
pub fn build_meshlet_lod(mesh: &Mesh, indices: &[u32]) -> MeshletMesh {
    // Clusters per group, and how far a group must simplify to count as progress (as in meshoptimizer's
    // clusterlod example): groups that keep more than 85% of their triangles are left as they are.
    const GROUP_SIZE: usize = 16;
    const SIMPLIFY_RATIO: f32 = 0.5;
    const STUCK_RATIO: f32 = 0.85;

    let vertices = mesh.vertices();
    // Vertices with the same position, for connectivity and locking: seams must not open up.
    let remap = meshopt::generate_position_remap(vertices);

    let mut all: Vec<Cluster> = Vec::new(); // every cluster of every level, in the order they are made
    clusterize(mesh, indices, 0, &mut all);
    for cluster in &mut all {
        cluster.lod_sphere = cluster_sphere(mesh, &cluster.indices);
    }
    let mut current: Vec<usize> = (0..all.len()).collect();

    let mut owner = vec![0i32; vertices.len()];
    let mut shared_vertex = vec![0u8; vertices.len()];
    let mut locks = vec![0u8; vertices.len()];
    let mut level = 0u32;
    while current.len() > 1 {
        let groups = partition(mesh, &all, &current, &remap, GROUP_SIZE);

        // Lock every position used by more than one group: both sides of a group border then keep the same
        // edge, so whatever level each side is drawn at, they meet without a crack.
        owner.fill(-1);
        shared_vertex.fill(0);
        for (g, group) in groups.iter().enumerate() {
            for &c in group {
                for &v in &all[c].indices {
                    let o = &mut owner[remap[v as usize] as usize];
                    if *o < 0 {
                        *o = g as i32;
                    } else if *o != g as i32 {
                        shared_vertex[remap[v as usize] as usize] = 1;
                    }
                }
            }
        }
        for v in 0..vertices.len() {
            locks[v] = if shared_vertex[remap[v] as usize] != 0 { SIMPLIFY_VERTEX_LOCK } else { 0 };
        }

        let mut next = Vec::new();
        for group in &groups {
            let mut merged: Vec<u32> = Vec::new();
            let mut spheres = Vec::new();
            let mut group_error = 0.0f32;
            for &c in group {
                merged.extend_from_slice(&all[c].indices);
                spheres.push(all[c].lod_sphere);
                group_error = group_error.max(all[c].lod_error);
            }
            // The group's bounds enclose its clusters' (not its own geometry, which could be smaller):
            // spheres and errors then never shrink from one level to the next.
            let merge = meshopt::compute_sphere_bounds(&spheres);
            let group_sphere = Vec4::new(merge.center[0], merge.center[1], merge.center[2], merge.radius);

            let target = ((merged.len() / 3) as f32 * SIMPLIFY_RATIO) as usize * 3;
            // Positions only, so the error is geometric (what lod_selected measures in pixels). Attribute
            // seams are kept anyway: vertices that share a position but not attributes only collapse
            // together. Locked: the vertices shared with other groups (by position), and the group's open
            // border, which also covers borders with clusters that stopped simplifying at an earlier level.
            let (simplified, error) = meshopt::simplify(
                &merged,
                vertices,
                &locks,
                target,
                SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE | SIMPLIFY_LOCK_BORDER,
            );
            if simplified.is_empty() || simplified.len() as f32 > merged.len() as f32 * STUCK_RATIO {
                continue; // stuck: these clusters stay the coarsest version of their part of the mesh
            }
            let parent_error = group_error.max(error);
            for &c in group {
                all[c].parent_sphere = group_sphere;
                all[c].parent_error = parent_error;
            }
            let first = all.len();
            clusterize(mesh, &simplified, level + 1, &mut all);
            for (c, cluster) in all.iter_mut().enumerate().skip(first) {
                cluster.lod_sphere = group_sphere;
                cluster.lod_error = parent_error;
                next.push(c);
            }
        }
        current = next;
        level += 1;
    }

    let mut result = MeshletMesh::default();
    result.meshlets.reserve(all.len());
    for cluster in &all {
        append_meshlet(mesh, cluster, &mut result);
    }
    result
}

/// The error `error` (object space), at a sphere `sphere` (xyz centre, w radius), as seen from `camera`, in
/// pixels: `error / distance to the sphere * pixels_per_unit`, where `pixels_per_unit` is the projection's
/// focal length in pixels (`proj[1][1] * height / 2`). Rotation invariant, and never decreasing for spheres
/// that contain each other with errors that do not decrease, which is what keeps the selection consistent.
/// The task shader computes the same.
pub fn lod_projected_error(sphere: Vec4, error: f32, camera: Vec3, pixels_per_unit: f32) -> f32 {
    if error >= LOD_NO_PARENT {
        return LOD_NO_PARENT;
    }
    // The distance to the sphere, not its centre: a camera inside it sees the error at its largest.
    let distance = ((sphere.truncate() - camera).length() - sphere.w).max(1e-6);
    error / distance * pixels_per_unit
}

/// Whether `meshlet` is drawn for a camera at `camera` (object space) and an error budget of `threshold`
/// pixels: its own error projects within the budget and its coarser version's does not. The task shader's
/// LOD test.
pub fn lod_selected(meshlet: &Meshlet, camera: Vec3, pixels_per_unit: f32, threshold: f32) -> bool {
    lod_projected_error(meshlet.lod_sphere, meshlet.lod_error, camera, pixels_per_unit) <= threshold
        && lod_projected_error(meshlet.parent_sphere, meshlet.parent_error, camera, pixels_per_unit) > threshold
}

/// Whether every triangle of `meshlet` faces away from a camera at `camera` (object space): the task
/// shader's cone test. Conservative: false whenever it cannot tell.
pub fn back_facing(meshlet: &Meshlet, camera: Vec3) -> bool {
    let to_apex = meshlet.cone_apex.truncate() - camera;
    let distance = to_apex.length();
    if distance <= 0.0 {
        return false;
    }
    (to_apex / distance).dot(meshlet.cone.truncate()) >= meshlet.cone.w
}

/// The six planes of the view frustum of `view_proj` (Vulkan clip space: 0 <= z <= w), as (normal,
/// distance) with unit normals pointing inwards: a point p is inside if `dot(n, p) + d >= 0` for all six.
/// Order: left, right, top, bottom, near, far.
pub fn frustum_planes(view_proj: &Mat4) -> [Vec4; 6] {
    // Rows of the matrix: clip = (r0.p, r1.p, r2.p, r3.p) for p = (x, y, z, 1).
    let row = |i: usize| Vec4::new(view_proj.x_axis[i], view_proj.y_axis[i], view_proj.z_axis[i], view_proj.w_axis[i]);
    let mut planes = [
        row(3) + row(0), // -w <= x
        row(3) - row(0), //  x <= w
        row(3) + row(1), // -w <= y
        row(3) - row(1), //  y <= w
        row(2),          //  0 <= z
        row(3) - row(2), //  z <= w
    ];
    for plane in &mut planes {
        *plane /= plane.truncate().length();
    }
    planes
}

/// Whether a sphere lies entirely outside one of `planes` (from [`frustum_planes`]).
pub fn sphere_outside_frustum(planes: &[Vec4; 6], center: Vec3, radius: f32) -> bool {
    planes.iter().any(|p| p.truncate().dot(center) + p.w < -radius)
}

/// The slopes x/d of the two lines from the eye that touch a circle of radius r centred at (x, d) in one
/// plane through the view axis (d: distance in front of the camera, greater than r).
fn tangent_slopes(x: f32, d: f32, r: f32) -> Vec2 {
    let l = (x * x + d * d - r * r).sqrt();
    Vec2::new((x * l - d * r) / (d * l + x * r), (x * l + d * r) / (d * l - x * r))
}

fn floor_power_of_two(value: u32) -> u32 {
    let mut result = 1;
    while result * 2 <= value {
        result *= 2;
    }
    result
}

/// The screen rectangle a sphere covers under the symmetric perspective projection `proj`
/// ([`crate::math::perspective`]): (min x, min y, max x, max y) in texture coordinates of the render target,
/// (0, 0) top left and (1, 1) bottom right, not clamped. Exact for the sphere's silhouette, so it holds every
/// pixel the sphere's contents can cover. `center` is in view space (the camera looks down -Z). Returns
/// `None` when the sphere reaches the near plane: it then covers no bounded rectangle.
pub fn project_sphere(center: Vec3, radius: f32, proj: &Mat4) -> Option<Vec4> {
    let d = -center.z;
    let near_z = proj.w_axis.z / proj.z_axis.z;
    if d - radius < near_z {
        return None;
    }
    // Clip x = proj[0][0] * view x and clip y = proj[1][1] * view y (negative: Vulkan's y points down),
    // both over w = d.
    let x = tangent_slopes(center.x, d, radius) * proj.x_axis.x;
    let y = tangent_slopes(center.y, d, radius) * proj.y_axis.y;
    Some(Vec4::new(x.x.min(x.y), y.x.min(y.y), x.x.max(x.y), y.x.max(y.y)) * 0.5 + Vec4::splat(0.5))
}

/// The size of mip 0 of the depth pyramid for a depth buffer of `extent`: the largest power of two not above
/// it in each axis, so that every further mip exactly halves it.
pub fn depth_pyramid_extent(extent: Extent2D) -> Extent2D {
    Extent2D::new(floor_power_of_two(extent.width.max(1)), floor_power_of_two(extent.height.max(1)))
}

/// Mips of a pyramid with mip 0 of `extent`, down to 1x1.
pub fn depth_pyramid_levels(extent: Extent2D) -> u32 {
    let mut levels = 1;
    let mut size = extent.width.max(extent.height);
    while size > 1 {
        levels += 1;
        size /= 2;
    }
    levels
}

