use crate::error::{Error, Result};
use crate::math::{Vec2, Vec3};
use crate::scene::{Mesh, Vertex};
use std::path::Path;

/// Loads every triangle mesh in an OBJ file into one [`Mesh`], with missing normals generated (smooth,
/// area-weighted).
pub fn load_mesh(path: &Path) -> Result<Mesh> {
    let options = tobj::LoadOptions { triangulate: true, single_index: true, ignore_points: true, ignore_lines: true };
    let (models, _materials) =
        tobj::load_obj(path, &options).map_err(|e| Error::Format { path: path.to_path_buf(), message: format!("cannot load: {e}") })?;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for model in &models {
        let m = &model.mesh;
        let base = vertices.len() as u32;
        let count = m.positions.len() / 3;
        let has_normals = m.normals.len() == m.positions.len();
        let has_uvs = m.texcoords.len() / 2 == count;
        let first = vertices.len();
        for i in 0..count {
            let mut v = Vertex { position: Vec3::from_slice(&m.positions[i * 3..i * 3 + 3]), ..Vertex::default() };
            if has_normals {
                v.normal = Vec3::from_slice(&m.normals[i * 3..i * 3 + 3]);
            }
            if has_uvs {
                v.uv = Vec2::from_slice(&m.texcoords[i * 2..i * 2 + 2]);
            }
            vertices.push(v);
        }
        indices.extend(m.indices.iter().map(|&i| base + i));
        if !has_normals {
            // Smooth normals: the sum of the (area-weighted) face normals at each vertex.
            let mut normals = vec![Vec3::ZERO; count];
            for tri in m.indices.chunks_exact(3) {
                let p = |k: usize| vertices[first + tri[k] as usize].position;
                let n = (p(1) - p(0)).cross(p(2) - p(0));
                for &i in tri {
                    normals[i as usize] += n;
                }
            }
            for (v, n) in vertices[first..].iter_mut().zip(normals) {
                v.normal = if n.length() > 0.0 { n.normalize() } else { Vec3::Z };
            }
        }
    }
    if indices.is_empty() {
        return Err(Error::NoTriangles(path.to_path_buf()));
    }
    Mesh::new(vertices, indices)
}
