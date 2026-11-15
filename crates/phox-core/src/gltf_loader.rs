//! glTF 2.0 loading (`.gltf` and `.glb`) into a [`Scene`].

use crate::error::{Error, Result};
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::scene::{AlphaMode, Bounds, Material, Mesh, ObjectDesc, ObjectId, Scene, Transform, Vertex};
use crate::texture::{SamplerDesc, Texture, TextureEncoding, TextureFilter, TextureWrap};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// What [`load_gltf`] added to a scene.
#[derive(Clone, Debug, Default)]
pub struct GltfImport {
    /// One per mesh primitive per node that uses the mesh.
    pub objects: Vec<ObjectId>,
    /// Of every object, in world space (after the root transform).
    pub bounds: Bounds,
    /// Distinct Phox meshes (glTF primitives) created.
    pub meshes: usize,
    pub materials: usize,
    pub textures: usize,
    /// Features that were ignored, e.g. extensions Phox does not support.
    pub warnings: Vec<String>,
}

/// Extensions Phox supports; any other entry of `extensionsRequired` is fatal, any other in `extensionsUsed`
/// is a warning.
const SUPPORTED_EXTENSIONS: [&str; 2] = ["KHR_materials_emissive_strength", "KHR_mesh_quantization"];

/// Loads the default scene of a glTF 2.0 file (.gltf with external, embedded or data-URI buffers and images,
/// or binary .glb) into `scene`, each object placed by its node's world transform times `root`.
/// Supported: triangle primitives (lists, strips and fans; indexed or not) with POSITION, NORMAL (flat
/// normals when missing, as the spec says), TEXCOORD_0 and TANGENT (generated when missing), the
/// metallic-roughness material model with its five textures, alpha modes, double-sided materials, samplers,
/// KHR_materials_emissive_strength. PNG and JPEG images. Primitives that share a glTF mesh share their Phox
/// Mesh, and textures their Texture. Not supported (listed in `warnings`): skins, morph targets, animations,
/// cameras, lights, texture coordinates other than TEXCOORD_0, KHR_texture_transform and other extensions;
/// points and lines are skipped.
pub fn load_gltf(path: &Path, scene: &mut Scene, root: &Mat4) -> Result<GltfImport> {
    let format = |message: String| Error::Format { path: path.to_path_buf(), message };
    // Extensions are checked before the importer sees the file, so an unsupported required one is named.
    let probe = gltf::Gltf::open(path).map_err(|e| format(format!("cannot parse glTF: {e}")))?;
    for extension in probe.document.extensions_required() {
        if !SUPPORTED_EXTENSIONS.contains(&extension) {
            return Err(Error::UnsupportedExtension { path: path.to_path_buf(), extension: extension.to_string() });
        }
    }
    drop(probe);
    let (document, buffers, images) = gltf::import(path).map_err(|e| format(format!("cannot load glTF: {e}")))?;

    let mut loader = Loader {
        path,
        scene,
        buffers: &buffers,
        images: &images,
        result: GltfImport::default(),
        meshes: HashMap::new(),
        materials: HashMap::new(),
        textures: HashMap::new(),
    };
    for extension in document.extensions_used() {
        if !SUPPORTED_EXTENSIONS.contains(&extension) {
            loader.warn(format!("extension {extension} is ignored"));
        }
    }
    if document.skins().count() > 0 {
        loader.warn("skins are ignored".into());
    }
    if document.animations().count() > 0 {
        loader.warn("animations are ignored".into());
    }
    if document.cameras().count() > 0 {
        loader.warn("cameras are ignored".into());
    }
    if document.lights().is_some_and(|mut l| l.next().is_some()) {
        loader.warn("lights (KHR_lights_punctual) are ignored".into());
    }

    if let Some(gltf_scene) = document.default_scene().or_else(|| document.scenes().next()) {
        for node in gltf_scene.nodes() {
            loader.add_node(&node, root)?;
        }
    } else {
        // No scene: every root node is shown.
        let mut children = vec![false; document.nodes().len()];
        for node in document.nodes() {
            for child in node.children() {
                children[child.index()] = true;
            }
        }
        for node in document.nodes().filter(|n| !children[n.index()]) {
            loader.add_node(&node, root)?;
        }
    }
    loader.result.meshes = loader.meshes.len();
    loader.result.materials = loader.materials.len();
    loader.result.textures = loader.textures.len();
    let result = loader.result;
    log::debug!(
        "glTF {}: {} object(s), {} mesh(es), {} material(s), {} texture(s)",
        path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        result.objects.len(),
        result.meshes,
        result.materials,
        result.textures
    );
    Ok(result)
}

struct Loader<'a> {
    path: &'a Path,
    scene: &'a mut Scene,
    buffers: &'a [gltf::buffer::Data],
    images: &'a [gltf::image::Data],
    result: GltfImport,
    meshes: HashMap<(usize, usize), Option<Arc<Mesh>>>,
    materials: HashMap<Option<usize>, Material>,
    textures: HashMap<(usize, TextureEncoding), Option<Arc<Texture>>>,
}

fn to_filter(filter: Option<gltf::texture::MinFilter>) -> TextureFilter {
    use gltf::texture::MinFilter::*;
    match filter {
        Some(Nearest | NearestMipmapNearest | NearestMipmapLinear) => TextureFilter::Nearest,
        _ => TextureFilter::Linear,
    }
}

fn to_wrap(wrap: gltf::texture::WrappingMode) -> TextureWrap {
    use gltf::texture::WrappingMode::*;
    match wrap {
        ClampToEdge => TextureWrap::ClampToEdge,
        MirroredRepeat => TextureWrap::MirroredRepeat,
        Repeat => TextureWrap::Repeat,
    }
}

impl Loader<'_> {
    fn warn(&mut self, message: String) {
        if !self.result.warnings.contains(&message) {
            log::warn!("glTF {}: {}", self.path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), message);
            self.result.warnings.push(message);
        }
    }

    fn add_node(&mut self, node: &gltf::Node, parent: &Mat4) -> Result<()> {
        let world = *parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(gltf_mesh) = node.mesh() {
            let primitive_count = gltf_mesh.primitives().len();
            for primitive in gltf_mesh.primitives() {
                let Some(mesh) = self.mesh_for(&gltf_mesh, &primitive) else { continue };
                let mut name = node.name().or(gltf_mesh.name()).unwrap_or("node").to_string();
                if primitive_count > 1 {
                    name += &format!(" {}", primitive.index());
                }
                let local = *mesh.bounds();
                for corner in 0..8 {
                    let point = Vec3::new(
                        if corner & 1 != 0 { local.max.x } else { local.min.x },
                        if corner & 2 != 0 { local.max.y } else { local.min.y },
                        if corner & 4 != 0 { local.max.z } else { local.min.z },
                    );
                    let w = (world * point.extend(1.0)).truncate();
                    if self.result.objects.is_empty() && corner == 0 {
                        self.result.bounds = Bounds { min: w, max: w };
                    }
                    self.result.bounds.min = self.result.bounds.min.min(w);
                    self.result.bounds.max = self.result.bounds.max.max(w);
                }
                let material = self.material_for(&primitive.material());
                let id = self.scene.add_object(ObjectDesc {
                    mesh: Some(mesh),
                    material,
                    transform: Transform::from_matrix(&world),
                    name,
                })?;
                self.result.objects.push(id);
            }
        }
        for child in node.children() {
            self.add_node(&child, &world)?;
        }
        Ok(())
    }

    fn mesh_for(&mut self, gltf_mesh: &gltf::Mesh, primitive: &gltf::Primitive) -> Option<Arc<Mesh>> {
        let key = (gltf_mesh.index(), primitive.index());
        if let Some(mesh) = self.meshes.get(&key) {
            return mesh.clone();
        }
        let mesh = self.build_mesh(primitive).map(Arc::new);
        self.meshes.insert(key, mesh.clone());
        mesh
    }

    fn build_mesh(&mut self, primitive: &gltf::Primitive) -> Option<Mesh> {
        use gltf::mesh::Mode;
        let mode = primitive.mode();
        if !matches!(mode, Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan) {
            self.warn("points and lines are skipped".into());
            return None;
        }
        if primitive.morph_targets().len() > 0 {
            self.warn("morph targets are ignored".into());
        }
        let buffers = self.buffers;
        let reader = primitive.reader(|b| Some(&buffers[b.index()].0[..]));
        let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
        if positions.is_empty() {
            return None;
        }
        let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
        let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().collect());
        let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());
        let mut vertices: Vec<Vertex> = positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut v = Vertex { position: Vec3::from(*p), ..Vertex::default() };
                if let Some(n) = normals.as_ref().and_then(|n| n.get(i)) {
                    v.normal = Vec3::from(*n);
                }
                if let Some(t) = uvs.as_ref().and_then(|t| t.get(i)) {
                    v.uv = Vec2::from(*t);
                }
                if let Some(t) = tangents.as_ref().and_then(|t| t.get(i)) {
                    v.tangent = Vec4::from(*t);
                }
                v
            })
            .collect();

        // The primitive's vertex order as a triangle list.
        let sequence: Vec<u32> = match reader.read_indices() {
            Some(indices) => indices.into_u32().collect(),
            None => (0..positions.len() as u32).collect(),
        };
        let mut order: Vec<u32> = Vec::new();
        match mode {
            Mode::Triangles => {
                for t in sequence.chunks_exact(3) {
                    order.extend_from_slice(t);
                }
            }
            Mode::TriangleStrip => {
                for i in 0..sequence.len().saturating_sub(2) {
                    // Every other triangle is reversed to keep the winding.
                    if i % 2 == 0 {
                        order.extend([sequence[i], sequence[i + 1], sequence[i + 2]]);
                    } else {
                        order.extend([sequence[i + 1], sequence[i], sequence[i + 2]]);
                    }
                }
            }
            _ => {
                for i in 1..sequence.len().saturating_sub(1) {
                    order.extend([sequence[i], sequence[i + 1], sequence[0]]);
                }
            }
        }
        if order.is_empty() || order.iter().any(|&i| i as usize >= vertices.len()) {
            return None;
        }
        if normals.is_none() {
            // Flat normals, as the spec asks: every triangle gets its own vertices.
            let mut flat = Vec::with_capacity(order.len());
            for t in order.chunks_exact(3) {
                let mut corners = [vertices[t[0] as usize], vertices[t[1] as usize], vertices[t[2] as usize]];
                let cross = (corners[1].position - corners[0].position).cross(corners[2].position - corners[0].position);
                let length = cross.length();
                let normal = if length > 0.0 { cross / length } else { Vec3::Z };
                for corner in &mut corners {
                    corner.normal = normal;
                    corner.tangent = Vec4::ZERO;
                    flat.push(*corner);
                }
            }
            vertices = flat;
            order = (0..vertices.len() as u32).collect();
        }
        match Mesh::new(vertices, order) {
            Ok(mesh) => Some(mesh),
            Err(e) => {
                self.warn(format!("a primitive was skipped: {e}"));
                None
            }
        }
    }

    fn material_for(&mut self, source: &gltf::Material) -> Material {
        let key = source.index();
        if let Some(material) = self.materials.get(&key) {
            return material.clone();
        }
        let material = if key.is_none() {
            // The spec's default material.
            Material { name: "default".into(), albedo: Vec4::ONE, metallic: 1.0, roughness: 1.0, double_sided: false, ..Material::default() }
        } else {
            let pbr = source.pbr_metallic_roughness();
            let mut emissive = Vec3::from(source.emissive_factor());
            if let Some(strength) = source.emissive_strength() {
                emissive *= strength;
            }
            Material {
                name: source.name().unwrap_or("material").to_string(),
                albedo: Vec4::from(pbr.base_color_factor()),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                base_color_texture: self
                    .texture_for(pbr.base_color_texture().map(|i| (i.texture(), i.tex_coord(), i.texture_transform().is_some())), TextureEncoding::Srgb),
                metallic_roughness_texture: self.texture_for(
                    pbr.metallic_roughness_texture().map(|i| (i.texture(), i.tex_coord(), i.texture_transform().is_some())),
                    TextureEncoding::Linear,
                ),
                normal_texture: self
                    .texture_for(source.normal_texture().map(|i| (i.texture(), i.tex_coord(), false)), TextureEncoding::Normal),
                normal_scale: source.normal_texture().map_or(1.0, |t| t.scale()),
                occlusion_texture: self
                    .texture_for(source.occlusion_texture().map(|i| (i.texture(), i.tex_coord(), false)), TextureEncoding::Linear),
                occlusion_strength: source.occlusion_texture().map_or(1.0, |t| t.strength()),
                emissive,
                emissive_texture: self.texture_for(
                    source.emissive_texture().map(|i| (i.texture(), i.tex_coord(), i.texture_transform().is_some())),
                    TextureEncoding::Srgb,
                ),
                alpha_mode: match source.alpha_mode() {
                    gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                },
                alpha_cutoff: source.alpha_cutoff().unwrap_or(0.5),
                double_sided: source.double_sided(),
            }
        };
        self.materials.insert(key, material.clone());
        material
    }

    fn texture_for(&mut self, view: Option<(gltf::Texture, u32, bool)>, encoding: TextureEncoding) -> Option<Arc<Texture>> {
        let (texture, tex_coord, has_transform) = view?;
        if tex_coord != 0 {
            self.warn("textures on texture coordinates other than TEXCOORD_0 use TEXCOORD_0".into());
        }
        if has_transform {
            self.warn("KHR_texture_transform is ignored".into());
        }
        let key = (texture.index(), encoding);
        if let Some(cached) = self.textures.get(&key) {
            return cached.clone();
        }
        let sampler_info = texture.sampler();
        let min = sampler_info.min_filter();
        let sampler = SamplerDesc {
            mag_filter: match sampler_info.mag_filter() {
                Some(gltf::texture::MagFilter::Nearest) => TextureFilter::Nearest,
                _ => TextureFilter::Linear,
            },
            min_filter: to_filter(min),
            mip_filter: match min {
                Some(gltf::texture::MinFilter::NearestMipmapNearest | gltf::texture::MinFilter::LinearMipmapNearest) => TextureFilter::Nearest,
                _ => TextureFilter::Linear,
            },
            wrap_u: to_wrap(sampler_info.wrap_s()),
            wrap_v: to_wrap(sampler_info.wrap_t()),
        };
        // Unset, or a mipmap mode.
        let mipmaps = !matches!(min, Some(gltf::texture::MinFilter::Nearest | gltf::texture::MinFilter::Linear));
        let result = self.decode(&texture, encoding, sampler, mipmaps);
        self.textures.insert(key, result.clone());
        result
    }

    fn decode(&mut self, texture: &gltf::Texture, encoding: TextureEncoding, sampler: SamplerDesc, mipmaps: bool) -> Option<Arc<Texture>> {
        let source = texture.source();
        let data = self.images.get(source.index())?;
        use gltf::image::Format;
        let (w, h) = (data.width as usize, data.height as usize);
        let px = &data.pixels;
        let mut rgba = Vec::with_capacity(w * h * 4);
        match data.format {
            Format::R8 => px.iter().for_each(|&r| rgba.extend([r, r, r, 255])),
            Format::R8G8 => px.chunks_exact(2).for_each(|p| rgba.extend([p[0], p[0], p[0], p[1]])),
            Format::R8G8B8 => px.chunks_exact(3).for_each(|p| rgba.extend([p[0], p[1], p[2], 255])),
            Format::R8G8B8A8 => rgba.extend_from_slice(px),
            Format::R16 => px.chunks_exact(2).for_each(|p| rgba.extend([p[1], p[1], p[1], 255])),
            Format::R16G16 => px.chunks_exact(4).for_each(|p| rgba.extend([p[1], p[1], p[1], p[3]])),
            Format::R16G16B16 => px.chunks_exact(6).for_each(|p| rgba.extend([p[1], p[3], p[5], 255])),
            Format::R16G16B16A16 => px.chunks_exact(8).for_each(|p| rgba.extend([p[1], p[3], p[5], p[7]])),
            _ => {
                self.warn("an image has an unsupported pixel format".into());
                return None;
            }
        };
        let name = source.name().map(str::to_string).unwrap_or_else(|| match source.source() {
            gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => uri.to_string(),
            _ => "image".to_string(),
        });
        match Texture::new(name, data.width, data.height, rgba, encoding, sampler, mipmaps) {
            Ok(t) => Some(Arc::new(t)),
            Err(e) => {
                self.warn(format!("an image could not be used ({e})"));
                None
            }
        }
    }
}
