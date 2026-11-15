//! Ready-made scenes and how they move over time. Used by the sample app and the golden-image tests.

use crate::color::{srgb, srgb3};
use crate::error::{Error, Result};
use crate::gltf_loader::load_gltf;
use crate::math::{HALF_PI, Mat4, Quat, Vec3, Vec4};
use crate::scene::{AlphaMode, Material, Mesh, ObjectDesc, ObjectId, Scene, Transform};
use crate::texture::{SamplerDesc, Texture, TextureEncoding, TextureFilter};
use std::path::Path;
use std::sync::Arc;

/// Poses a scene at a time in seconds.
pub type Animate = Box<dyn Fn(&mut Scene, f32) + Send + Sync>;

/// A scene and its animation: poses the scene `seconds` into it. Deterministic: the same time gives the same
/// pose.
pub struct SampleScene {
    pub name: String,
    pub scene: Scene,
    pub animate: Animate,
}

/// The built-in scenes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleSceneId {
    /// One triangle spinning about the vertical axis.
    Triangle,
    /// Coloured cubes on a floor, turning at different rates; one partly hides another.
    Cubes,
    /// Tori on a floor, one cut by the edge of the view: exercises meshlet culling.
    Tori,
    /// The material model: metal and plastic spheres from smooth to rough, base colour, metallic-roughness
    /// and normal textures, an alpha-masked screen, a blended glass sphere, emission.
    Materials,
    /// A benchmark: 576 dense tori (37.7 million triangles), many cut by the frustum.
    Field,
    /// A benchmark: 32,768 small spinning cubes, for the cost of each object to the renderer.
    Crowd,
}

/// Every [`SampleSceneId`].
pub const SAMPLE_SCENES: [SampleSceneId; 6] = [
    SampleSceneId::Triangle,
    SampleSceneId::Cubes,
    SampleSceneId::Tori,
    SampleSceneId::Materials,
    SampleSceneId::Field,
    SampleSceneId::Crowd,
];

/// The scene's name in text, e.g. "cubes": for command lines and file names.
pub fn sample_scene_name(id: SampleSceneId) -> &'static str {
    match id {
        SampleSceneId::Triangle => "triangle",
        SampleSceneId::Cubes => "cubes",
        SampleSceneId::Tori => "tori",
        SampleSceneId::Materials => "materials",
        SampleSceneId::Field => "field",
        SampleSceneId::Crowd => "crowd",
    }
}

/// The scene that [`sample_scene_name`] calls `name`, if any. For input typed by a person.
pub fn parse_sample_scene(name: &str) -> Option<SampleSceneId> {
    SAMPLE_SCENES.into_iter().find(|&id| sample_scene_name(id) == name)
}

fn spin(radians: f32, axis: Vec3) -> Quat {
    Quat::from_axis_angle(axis.normalize(), radians)
}

/// A fairly rough dielectric in a colour picked in sRGB: flat colours with a soft highlight.
fn paint(r: f32, g: f32, b: f32, double_sided: bool) -> Material {
    Material { albedo: srgb3(r, g, b), roughness: 0.6, double_sided, ..Material::default() }
}

fn at(position: Vec3) -> Transform {
    Transform::at(position)
}

fn mesh(m: Mesh) -> Arc<Mesh> {
    Arc::new(m)
}

/// Sets the rotation of object `id` (nothing happens if it has been removed).
fn rotate(scene: &mut Scene, id: ObjectId, rotation: Quat) {
    if let Some(object) = scene.get_mut(id) {
        object.transform_mut().rotation = rotation;
    }
}

fn triangle() -> Result<SampleScene> {
    let mut scene = Scene::new();
    let tri = scene.add_object(ObjectDesc {
        mesh: Some(mesh(Mesh::triangle(1.0))),
        material: paint(1.0, 0.2, 0.4, true),
        ..Default::default()
    })?;
    scene.camera.position = Vec3::new(0.0, 0.0, 3.0);
    scene.camera.look_at(Vec3::ZERO, Vec3::Y);
    scene.light.direction = Vec3::new(-0.4, -0.6, -1.0);
    Ok(SampleScene {
        name: String::new(),
        scene,
        animate: Box::new(move |s, t| rotate(s, tri, spin(1.2 * t, Vec3::Y))),
    })
}

fn cubes() -> Result<SampleScene> {
    let mut scene = Scene::new();
    let cube = mesh(Mesh::cube(1.0));
    scene.add_object(ObjectDesc {
        mesh: Some(mesh(Mesh::plane(8.0))),
        material: paint(0.55, 0.55, 0.6, true),
        transform: at(Vec3::new(0.0, -0.5, 0.0)),
        ..Default::default()
    })?;
    let [red, green, blue, _yellow, purple] = scene.add_objects([
        ObjectDesc { mesh: Some(cube.clone()), material: paint(0.9, 0.25, 0.2, true), ..Default::default() },
        ObjectDesc {
            mesh: Some(cube.clone()),
            material: paint(0.3, 0.75, 0.35, true),
            transform: at(Vec3::new(-1.6, 0.0, -0.8)),
            ..Default::default()
        },
        ObjectDesc {
            mesh: Some(cube.clone()),
            material: paint(0.25, 0.45, 0.9, true),
            transform: at(Vec3::new(1.6, 0.0, -0.8)),
            ..Default::default()
        },
        // Stands still.
        ObjectDesc {
            mesh: Some(cube.clone()),
            material: paint(0.9, 0.8, 0.3, true),
            transform: Transform { position: Vec3::new(0.0, 0.25, -2.2), scale: Vec3::new(1.0, 1.5, 1.0), ..Default::default() },
            ..Default::default()
        },
        // In front of the red cube and overlapping it on screen, so depth testing has something to do.
        ObjectDesc {
            mesh: Some(cube),
            material: paint(0.6, 0.35, 0.85, true),
            transform: Transform { position: Vec3::new(0.55, -0.25, 1.0), scale: Vec3::splat(0.5), ..Default::default() },
            ..Default::default()
        },
    ])?;
    scene.camera.position = Vec3::new(0.0, 2.2, 5.0);
    scene.camera.look_at(Vec3::new(0.0, 0.0, -0.3), Vec3::Y);
    scene.light.direction = Vec3::new(-0.5, -1.0, -0.35);
    Ok(SampleScene {
        name: String::new(),
        scene,
        animate: Box::new(move |s, t| {
            rotate(s, red, spin(0.8 * t + 0.4, Vec3::Y));
            rotate(s, green, spin(-0.6 * t, Vec3::Y));
            rotate(s, blue, spin(0.5 * t + 0.3, Vec3::new(0.3, 1.0, 0.0)));
            rotate(s, purple, spin(1.5 * t + 0.6, Vec3::new(1.0, 1.0, 0.0)));
        }),
    })
}

/// Tori on a floor, one of them cut by the left edge of the view and one close enough to fill a good part of
/// it. Each torus is a few dozen meshlets, about half of them facing away, so the mesh shading path's
/// frustum and cone culling both have work to do.
fn tori() -> Result<SampleScene> {
    let mut scene = Scene::new();
    let torus = mesh(Mesh::torus(0.6, 0.22, 64, 32));
    scene.add_object(ObjectDesc {
        mesh: Some(mesh(Mesh::plane(10.0))),
        material: paint(0.55, 0.55, 0.6, false),
        transform: at(Vec3::new(0.0, -0.5, -1.0)),
        ..Default::default()
    })?;
    let [near, edge, back, left] = scene.add_objects([
        ObjectDesc {
            mesh: Some(torus.clone()),
            material: paint(0.95, 0.5, 0.15, false),
            transform: at(Vec3::new(0.5, 0.1, 1.6)),
            ..Default::default()
        },
        ObjectDesc {
            mesh: Some(torus.clone()),
            material: paint(0.2, 0.7, 0.7, false),
            transform: at(Vec3::new(-2.6, 0.2, 0.0)),
            ..Default::default()
        },
        ObjectDesc {
            mesh: Some(torus.clone()),
            material: paint(0.6, 0.4, 0.9, false),
            transform: Transform { position: Vec3::new(0.9, 0.4, -2.2), scale: Vec3::splat(1.4), ..Default::default() },
            ..Default::default()
        },
        ObjectDesc {
            mesh: Some(torus),
            material: paint(0.55, 0.85, 0.3, false),
            transform: at(Vec3::new(-1.0, 0.0, -0.6)),
            ..Default::default()
        },
    ])?;
    scene.camera.position = Vec3::new(0.0, 1.6, 4.0);
    scene.camera.look_at(Vec3::new(0.0, 0.0, -0.4), Vec3::Y);
    scene.light.direction = Vec3::new(-0.5, -1.0, -0.4);
    Ok(SampleScene {
        name: String::new(),
        scene,
        animate: Box::new(move |s, t| {
            rotate(s, near, spin(0.7 * t + 0.9, Vec3::new(1.0, 0.2, 0.0)));
            rotate(s, edge, spin(-0.5 * t + 0.4, Vec3::new(0.0, 0.3, 1.0)));
            rotate(s, back, spin(0.4 * t + 1.2, Vec3::new(1.0, 1.0, 0.0)));
            rotate(s, left, spin(0.9 * t, Vec3::new(0.2, 0.0, 1.0)));
        }),
    })
}

/// A benchmark rather than a picture: a 24 x 24 grid of 65k-triangle tori (37.7 million triangles), seen from
/// inside the grid so that many tori are cut by the frustum. Far too heavy for the CPU rasterizer.
fn field() -> Result<SampleScene> {
    const GRID: i32 = 24;
    const SPACING: f32 = 2.0;
    let mut scene = Scene::new();
    let torus = mesh(Mesh::torus(0.6, 0.22, 256, 128));
    for x in 0..GRID {
        for z in 0..GRID {
            let hue = ((x * 7 + z * 3) % 10) as f32 / 10.0;
            let albedo = srgb3(0.4 + 0.5 * hue, 0.35 + 0.4 * (1.0 - hue), 0.5 + 0.3 * (6.0 * hue).sin());
            let position = Vec3::new(
                (x as f32 - 0.5 * (GRID - 1) as f32) * SPACING,
                0.0,
                (z as f32 - 0.5 * (GRID - 1) as f32) * SPACING,
            );
            scene.add_object(ObjectDesc {
                mesh: Some(torus.clone()),
                material: Material { albedo, roughness: 0.6, double_sided: false, ..Material::default() },
                transform: at(position),
                ..Default::default()
            })?;
        }
    }
    scene.camera.position = Vec3::new(0.0, 2.5, 6.0);
    scene.camera.look_at(Vec3::new(0.0, 0.0, -4.0), Vec3::Y);
    Ok(SampleScene {
        name: String::new(),
        scene,
        animate: Box::new(|s, t| {
            // Nothing is removed, so the objects stay in the order they were added.
            for (i, object) in s.objects_mut().iter_mut().enumerate() {
                object.transform_mut().rotation = spin(0.5 * t + 0.37 * i as f32, Vec3::new(1.0, 0.5, 0.25));
            }
        }),
    })
}

/// A tangent-space normal map of round bumps in a `cells` x `cells` grid, from the gradient of a height field.
fn bump_normal_map(size: u32, cells: u32, strength: f32) -> Result<Arc<Texture>> {
    let height = |x: f32, y: f32| {
        let u = (x * cells as f32 / size as f32) % 1.0 - 0.5;
        let v = (y * cells as f32 / size as f32) % 1.0 - 0.5;
        let r2 = (u * u + v * v) * 4.0;
        if r2 < 1.0 { (1.0 - r2).sqrt() } else { 0.0 }
    };
    let mut rgba = Vec::with_capacity(size as usize * size as usize * 4);
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let dx = (height(fx + 1.0, fy) - height(fx - 1.0, fy)) * 0.5;
            let dy = (height(fx, fy + 1.0) - height(fx, fy - 1.0)) * 0.5;
            // +y of the map points along +v (down the image), so the y slope keeps its sign here.
            let n = Vec3::new(-dx * strength, -dy * strength, 1.0).normalize();
            for c in 0..3 {
                rgba.push(((n[c] * 0.5 + 0.5) * 255.0).round() as u8);
            }
            rgba.push(255);
        }
    }
    Ok(Arc::new(Texture::new("bumps", size, size, rgba, TextureEncoding::Normal, SamplerDesc::default(), true)?))
}

/// Opaque white with round holes (alpha 0) in a `cells` x `cells` grid, for alpha masking.
fn holes_texture(size: u32, cells: u32) -> Result<Arc<Texture>> {
    let mut rgba = vec![255u8; size as usize * size as usize * 4];
    for y in 0..size {
        for x in 0..size {
            let u = ((x as f32 + 0.5) * cells as f32 / size as f32) % 1.0 - 0.5;
            let v = ((y as f32 + 0.5) * cells as f32 / size as f32) % 1.0 - 0.5;
            rgba[(y as usize * size as usize + x as usize) * 4 + 3] = if u * u + v * v < 0.09 { 0 } else { 255 };
        }
    }
    Ok(Arc::new(Texture::new("holes", size, size, rgba, TextureEncoding::Srgb, SamplerDesc::default(), true)?))
}

/// A benchmark of object count rather than triangles: 32 x 32 x 32 small spinning cubes (32,768 objects, 393k
/// triangles), so the per-object work of a renderer shows.
fn crowd() -> Result<SampleScene> {
    const SIDE: i32 = 32;
    const SPACING: f32 = 0.6;
    let mut scene = Scene::new();
    let cube = mesh(Mesh::cube(0.3));
    let paints = [paint(0.9, 0.25, 0.2, false), paint(0.3, 0.75, 0.35, false), paint(0.25, 0.45, 0.9, false), paint(0.9, 0.8, 0.3, false)];
    for x in 0..SIDE {
        for y in 0..SIDE {
            for z in 0..SIDE {
                let position = (Vec3::new(x as f32, y as f32, z as f32) - Vec3::splat(0.5 * (SIDE - 1) as f32)) * SPACING;
                scene.add_object(ObjectDesc {
                    mesh: Some(cube.clone()),
                    material: paints[(x + y + z) as usize % paints.len()].clone(),
                    transform: at(position),
                    ..Default::default()
                })?;
            }
        }
    }
    scene.camera.position = Vec3::new(6.0, 5.0, 16.0);
    scene.camera.look_at(Vec3::ZERO, Vec3::Y);
    Ok(SampleScene {
        name: String::new(),
        scene,
        animate: Box::new(|s, t| {
            for (i, object) in s.objects_mut().iter_mut().enumerate() {
                object.transform_mut().rotation = spin(1.3 * t + 0.11 * i as f32, Vec3::new(1.0, 0.7, 0.3));
            }
        }),
    })
}

/// The material model: a row of gold spheres from smooth to rough, a row of red plastic ones, a textured and
/// normal-mapped sphere, a screen with holes (alpha mask), a glass sphere (alpha blend) and a glowing cube,
/// on a checkered floor.
fn materials() -> Result<SampleScene> {
    let mut scene = Scene::new();
    let sphere = mesh(Mesh::sphere(0.35, 32, 16));

    // Nearest magnification keeps the squares' edges sharp close up; mips still smooth them far away.
    let checker = Arc::new(Texture::checker(
        "floor checker",
        64,
        8,
        srgb3(0.75, 0.75, 0.78),
        srgb3(0.35, 0.35, 0.4),
        TextureEncoding::Srgb,
        SamplerDesc { mag_filter: TextureFilter::Nearest, ..Default::default() },
    )?);
    scene.add_object(ObjectDesc {
        mesh: Some(mesh(Mesh::plane(8.0))),
        material: Material { albedo: Vec4::ONE, roughness: 0.8, double_sided: false, base_color_texture: Some(checker), ..Material::default() },
        transform: at(Vec3::new(0.0, -0.4, -0.5)),
        ..Default::default()
    })?;

    const STEPS: i32 = 5;
    for i in 0..STEPS {
        let roughness = 0.1 + 0.85 * i as f32 / (STEPS - 1) as f32;
        let x = (i as f32 - 0.5 * (STEPS - 1) as f32) * 0.8;
        scene.add_objects([
            // gold
            ObjectDesc {
                mesh: Some(sphere.clone()),
                material: Material {
                    albedo: srgb3(1.0, 0.78, 0.34),
                    metallic: 1.0,
                    roughness,
                    double_sided: false,
                    ..Material::default()
                },
                transform: at(Vec3::new(x, 0.0, -1.3)),
                ..Default::default()
            },
            // red plastic
            ObjectDesc {
                mesh: Some(sphere.clone()),
                material: Material { albedo: srgb3(0.8, 0.1, 0.1), roughness, double_sided: false, ..Material::default() },
                transform: at(Vec3::new(x, 0.0, -2.2)),
                ..Default::default()
            },
        ])?;
    }

    let tiles = Arc::new(Texture::checker(
        "tiles",
        128,
        4,
        srgb3(0.9, 0.85, 0.75),
        srgb3(0.2, 0.45, 0.6),
        TextureEncoding::Srgb,
        SamplerDesc::default(),
    )?);
    // Metallic-roughness: g roughness, b metallic. The light squares are polished metal, the dark ones rough.
    let tile_surface = Arc::new(Texture::checker(
        "tile surface",
        128,
        4,
        Vec4::new(0.0, 0.25, 1.0, 1.0),
        Vec4::new(0.0, 0.9, 0.0, 1.0),
        TextureEncoding::Linear,
        SamplerDesc::default(),
    )?);
    let [textured, _screen, _glass, glow] = scene.add_objects([
        ObjectDesc {
            mesh: Some(mesh(Mesh::sphere(0.55, 48, 24))),
            material: Material {
                albedo: Vec4::ONE,
                metallic: 1.0,
                roughness: 1.0,
                double_sided: false,
                base_color_texture: Some(tiles),
                metallic_roughness_texture: Some(tile_surface),
                normal_texture: Some(bump_normal_map(128, 8, 6.0)?),
                ..Material::default()
            },
            transform: at(Vec3::new(-1.0, 0.15, 0.2)),
            ..Default::default()
        },
        // An upright screen (a plane turned to face the camera), double-sided, its holes cut by the mask.
        ObjectDesc {
            mesh: Some(mesh(Mesh::plane(1.0))),
            material: Material {
                albedo: srgb3(0.3, 0.6, 0.95),
                roughness: 0.5,
                alpha_mode: AlphaMode::Mask,
                base_color_texture: Some(holes_texture(64, 4)?),
                ..Material::default()
            },
            transform: Transform { position: Vec3::new(0.55, 0.15, -0.3), rotation: spin(HALF_PI, Vec3::X), ..Default::default() },
            ..Default::default()
        },
        ObjectDesc {
            mesh: Some(sphere),
            material: Material {
                albedo: srgb(0.6, 0.85, 1.0, 0.35),
                roughness: 0.05,
                alpha_mode: AlphaMode::Blend,
                double_sided: false,
                ..Material::default()
            },
            transform: at(Vec3::new(1.1, 0.05, 0.6)),
            ..Default::default()
        },
        ObjectDesc {
            mesh: Some(mesh(Mesh::cube(0.3))),
            material: Material {
                albedo: srgb3(0.2, 0.1, 0.05),
                roughness: 0.9,
                emissive: Vec3::new(4.0, 1.6, 0.4),
                double_sided: false,
                ..Material::default()
            },
            transform: at(Vec3::new(0.1, -0.2, 0.9)),
            ..Default::default()
        },
    ])?;
    scene.camera.position = Vec3::new(0.0, 1.4, 3.2);
    scene.camera.look_at(Vec3::new(0.0, -0.1, -0.6), Vec3::Y);
    scene.light.direction = Vec3::new(-0.6, -1.0, -0.5);
    Ok(SampleScene {
        name: String::new(),
        scene,
        animate: Box::new(move |s, t| {
            rotate(s, textured, spin(0.5 * t + 0.6, Vec3::Y));
            rotate(s, glow, spin(0.8 * t + 0.3, Vec3::new(0.2, 1.0, 0.1)));
        }),
    })
}

/// Builds the sample scene `id`, posed at time 0.
pub fn make_sample_scene(id: SampleSceneId) -> Result<SampleScene> {
    let mut sample = match id {
        SampleSceneId::Triangle => triangle()?,
        SampleSceneId::Cubes => cubes()?,
        SampleSceneId::Tori => tori()?,
        SampleSceneId::Materials => materials()?,
        SampleSceneId::Field => field()?,
        SampleSceneId::Crowd => crowd()?,
    };
    sample.name = sample_scene_name(id).to_string();
    (sample.animate)(&mut sample.scene, 0.0);
    Ok(sample)
}

/// `mesh` centred and scaled to fit the view, turning on a turntable.
pub fn make_mesh_viewer_scene(mesh: Arc<Mesh>, name: impl Into<String>) -> Result<SampleScene> {
    let mut scene = Scene::new();
    let bounds = *mesh.bounds();
    let size = bounds.size();
    let largest = size.x.max(size.y).max(size.z).max(1e-6);
    let scale = 2.0 / largest;
    // Pivot about the mesh's centre: the turntable spins the centred mesh.
    let id = scene.add_object(ObjectDesc { mesh: Some(mesh), material: paint(0.8, 0.75, 0.7, true), ..Default::default() })?;
    scene.camera.position = Vec3::new(0.0, 0.8, 3.6);
    scene.camera.look_at(Vec3::ZERO, Vec3::Y);
    let center = bounds.center();
    let mut sample = SampleScene {
        name: name.into(),
        scene,
        animate: Box::new(move |s, t| {
            if let Some(object) = s.get_mut(id) {
                let mut transform = object.transform_mut();
                transform.rotation = spin(0.6 * t, Vec3::Y);
                transform.scale = Vec3::splat(scale);
                transform.position = transform.rotation * (-center * scale);
            }
        }),
    };
    (sample.animate)(&mut sample.scene, 0.0);
    Ok(sample)
}

/// The default scene of a glTF 2.0 file (see [`load_gltf`]), centred and scaled to fit the view, turning on a
/// turntable.
pub fn make_gltf_viewer_scene(path: &Path) -> Result<SampleScene> {
    let mut scene = Scene::new();
    let import = load_gltf(path, &mut scene, &Mat4::IDENTITY)?;
    if import.objects.is_empty() {
        return Err(Error::NothingToDraw(path.to_path_buf()));
    }
    let size = import.bounds.size();
    let scale = 2.0 / size.x.max(size.y).max(size.z).max(1e-6);
    // Each object keeps its place in the model; the turntable turns the whole model about its centre.
    let placed: Vec<(ObjectId, Mat4)> = import
        .objects
        .iter()
        .map(|&id| Ok((id, scene.at(id)?.transform().matrix())))
        .collect::<Result<_>>()?;
    scene.camera.position = Vec3::new(0.0, 0.8, 3.6);
    scene.camera.look_at(Vec3::ZERO, Vec3::Y);
    let center = import.bounds.center();
    let mut sample = SampleScene {
        name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        scene,
        animate: Box::new(move |s, t| {
            let turntable = Mat4::from_quat(spin(0.6 * t, Vec3::Y)) * Mat4::from_scale(Vec3::splat(scale)) * Mat4::from_translation(-center);
            for (id, model) in &placed {
                if let Some(object) = s.get_mut(*id) {
                    *object.transform_mut() = Transform::from_matrix(&(turntable * *model));
                }
            }
        }),
    };
    (sample.animate)(&mut sample.scene, 0.0);
    Ok(sample)
}
