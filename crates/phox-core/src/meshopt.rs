//! Hand-written FFI for the meshoptimizer v1.3 functions the meshlet builder calls (vendor/meshoptimizer,
//! compiled by build.rs), with small safe wrappers. The raw types mirror meshoptimizer.h.

use crate::math::{Vec3, Vec4};
use crate::scene::Vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RawMeshlet {
    pub vertex_offset: u32,
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub cone_axis_s8: [i8; 3],
    pub cone_cutoff_s8: i8,
}

pub const SIMPLIFY_LOCK_BORDER: u32 = 1 << 0;
pub const SIMPLIFY_SPARSE: u32 = 1 << 1;
pub const SIMPLIFY_ERROR_ABSOLUTE: u32 = 1 << 2;
pub const SIMPLIFY_VERTEX_LOCK: u8 = 1 << 0;

unsafe extern "C" {
    fn meshopt_optimizeVertexCache(destination: *mut u32, indices: *const u32, index_count: usize, vertex_count: usize);
    fn meshopt_buildMeshletsBound(index_count: usize, max_vertices: usize, max_triangles: usize) -> usize;
    fn meshopt_buildMeshlets(
        meshlets: *mut RawMeshlet,
        meshlet_vertices: *mut u32,
        meshlet_triangles: *mut u8,
        indices: *const u32,
        index_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
        max_vertices: usize,
        max_triangles: usize,
        cone_weight: f32,
    ) -> usize;
    fn meshopt_extractMeshletIndices(vertices: *mut u32, triangles: *mut u8, indices: *const u32, index_count: usize) -> usize;
    fn meshopt_optimizeMeshlet(meshlet_vertices: *mut u32, meshlet_triangles: *mut u8, triangle_count: usize, vertex_count: usize);
    fn meshopt_computeMeshletBounds(
        meshlet_vertices: *const u32,
        meshlet_triangles: *const u8,
        triangle_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
    ) -> Bounds;
    fn meshopt_computeClusterBounds(
        indices: *const u32,
        index_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
    ) -> Bounds;
    fn meshopt_generatePositionRemap(destination: *mut u32, vertex_positions: *const f32, vertex_count: usize, vertex_positions_stride: usize);
    fn meshopt_partitionClusters(
        destination: *mut u32,
        cluster_indices: *const u32,
        total_index_count: usize,
        cluster_index_counts: *const u32,
        cluster_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
        target_partition_size: usize,
    ) -> usize;
    fn meshopt_computeSphereBounds(positions: *const f32, count: usize, positions_stride: usize, radii: *const f32, radii_stride: usize) -> Bounds;
    fn meshopt_simplifyWithAttributes(
        destination: *mut u32,
        indices: *const u32,
        index_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
        vertex_attributes: *const f32,
        vertex_attributes_stride: usize,
        attribute_weights: *const f32,
        attribute_count: usize,
        vertex_lock: *const u8,
        target_index_count: usize,
        target_error: f32,
        options: u32,
        result_error: *mut f32,
    ) -> usize;
}

const STRIDE: usize = size_of::<Vertex>();

fn positions(vertices: &[Vertex]) -> *const f32 {
    vertices.as_ptr().cast()
}

/// `indices` reordered for the GPU's post-transform vertex cache.
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut out = vec![0u32; indices.len()];
    // SAFETY: both arrays hold `indices.len()` entries, all below `vertex_count` (checked by Mesh).
    unsafe { meshopt_optimizeVertexCache(out.as_mut_ptr(), indices.as_ptr(), indices.len(), vertex_count) };
    out
}

/// Meshlets (with `max_vertices` / `max_triangles` limits) over `local` indices into `positions`.
pub fn build_meshlets(
    local: &[u32],
    positions: &[Vec3],
    max_vertices: usize,
    max_triangles: usize,
    cone_weight: f32,
) -> (Vec<RawMeshlet>, Vec<u32>, Vec<u8>) {
    // SAFETY: the bound sizes the three output buffers as meshoptimizer documents.
    unsafe {
        let max_meshlets = meshopt_buildMeshletsBound(local.len(), max_vertices, max_triangles);
        let mut meshlets = vec![RawMeshlet::default(); max_meshlets];
        let mut vertices = vec![0u32; max_meshlets * max_vertices];
        let mut triangles = vec![0u8; max_meshlets * max_triangles * 3];
        let count = meshopt_buildMeshlets(
            meshlets.as_mut_ptr(),
            vertices.as_mut_ptr(),
            triangles.as_mut_ptr(),
            local.as_ptr(),
            local.len(),
            positions.as_ptr().cast(),
            positions.len(),
            size_of::<Vec3>(),
            max_vertices,
            max_triangles,
            cone_weight,
        );
        meshlets.truncate(count);
        (meshlets, vertices, triangles)
    }
}

/// Meshlet-local vertex and triangle indices of a cluster's absolute `indices`: the vertex list, the local
/// triangle indices, and how many vertices the cluster uses.
pub fn extract_meshlet_indices(indices: &[u32]) -> (Vec<u32>, Vec<u8>, usize) {
    let mut vertices = vec![0u32; indices.len()];
    let mut triangles = vec![0u8; indices.len()];
    // SAFETY: the outputs hold `indices.len()` entries each.
    let count = unsafe { meshopt_extractMeshletIndices(vertices.as_mut_ptr(), triangles.as_mut_ptr(), indices.as_ptr(), indices.len()) };
    (vertices, triangles, count)
}

pub fn optimize_meshlet(vertices: &mut [u32], triangles: &mut [u8], triangle_count: usize, vertex_count: usize) {
    // SAFETY: the slices hold the meshlet's `vertex_count` vertices and `triangle_count` * 3 indices.
    unsafe { meshopt_optimizeMeshlet(vertices.as_mut_ptr(), triangles.as_mut_ptr(), triangle_count, vertex_count) };
}

pub fn compute_meshlet_bounds(vertices: &[u32], triangles: &[u8], triangle_count: usize, mesh: &[Vertex]) -> Bounds {
    // SAFETY: meshlet indices are below `mesh.len()`; positions are the first 12 bytes of each vertex.
    unsafe { meshopt_computeMeshletBounds(vertices.as_ptr(), triangles.as_ptr(), triangle_count, positions(mesh), mesh.len(), STRIDE) }
}

pub fn compute_cluster_bounds(indices: &[u32], mesh: &[Vertex]) -> Bounds {
    // SAFETY: as above.
    unsafe { meshopt_computeClusterBounds(indices.as_ptr(), indices.len(), positions(mesh), mesh.len(), STRIDE) }
}

/// For each vertex, the first vertex with the same position.
pub fn generate_position_remap(mesh: &[Vertex]) -> Vec<u32> {
    let mut remap = vec![0u32; mesh.len()];
    // SAFETY: `remap` holds one entry per vertex.
    unsafe { meshopt_generatePositionRemap(remap.as_mut_ptr(), positions(mesh), mesh.len(), STRIDE) };
    remap
}

/// Groups clusters (`indices` concatenated, `counts` per cluster) into about `target` per group. Returns
/// each cluster's group and the group count.
pub fn partition_clusters(indices: &[u32], counts: &[u32], mesh: &[Vertex], target: usize) -> (Vec<u32>, usize) {
    let mut part = vec![0u32; counts.len()];
    // SAFETY: `part` holds one entry per cluster; the counts sum to `indices.len()`.
    let groups = unsafe {
        meshopt_partitionClusters(
            part.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            counts.as_ptr(),
            counts.len(),
            positions(mesh),
            mesh.len(),
            STRIDE,
            target,
        )
    };
    (part, groups)
}

/// A sphere around spheres (xyz centre, w radius).
pub fn compute_sphere_bounds(spheres: &[Vec4]) -> Bounds {
    let base: *const f32 = spheres.as_ptr().cast();
    // SAFETY: each element is 4 floats; the radius is the fourth.
    unsafe { meshopt_computeSphereBounds(base, spheres.len(), size_of::<Vec4>(), base.add(3), size_of::<Vec4>()) }
}

/// Simplifies `indices` to about `target_index_count` with the vertices marked in `locks` kept in place.
/// Returns the new indices and the error reached.
pub fn simplify(indices: &[u32], mesh: &[Vertex], locks: &[u8], target_index_count: usize, options: u32) -> (Vec<u32>, f32) {
    let mut out = vec![0u32; indices.len()];
    let mut error = 0.0f32;
    // SAFETY: `locks` has one byte per vertex; `out` holds as many indices as the input.
    let count = unsafe {
        meshopt_simplifyWithAttributes(
            out.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions(mesh),
            mesh.len(),
            STRIDE,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            locks.as_ptr(),
            target_index_count,
            3.0e38,
            options,
            &mut error,
        )
    };
    out.truncate(count);
    (out, error)
}
