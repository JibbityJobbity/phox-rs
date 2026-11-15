//! Scene, camera and view extraction, meshlets, the software rasterizer and mesh loading. No GPU needed.
//! Ported from test/source/render_view_test.cpp.

use phox_core::Error;
use phox_core::camera::Camera;
use phox_core::gltf_loader::load_gltf;
use phox_core::math::*;
use phox_core::mesh_loader::load_mesh;
use phox_core::meshlets::*;
use phox_core::render_view::{extract_view, outside_frustum};
use phox_core::resources::Extent2D;
use phox_core::scene::*;
use phox_core::shader_interface::*;
use phox_core::software_rasterizer::rasterize;
use phox_core::texture::*;
use std::collections::BTreeMap;
use std::sync::Arc;

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4
}
fn near_v(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a - b).length() <= eps
}
fn desc(mesh: &Arc<Mesh>, albedo: Vec4, position: Vec3) -> ObjectDesc {
    ObjectDesc {
        mesh: Some(mesh.clone()),
        material: Material { albedo, ..Material::default() },
        transform: Transform::at(position),
        ..Default::default()
    }
}

#[test]
fn projection_matches_vulkan_clip_space() {
    let proj = perspective(90f32.to_radians(), 1.0, 1.0, 10.0);
    let near_point = proj * Vec4::new(0.0, 0.0, -1.0, 1.0);
    let far_point = proj * Vec4::new(0.0, 0.0, -10.0, 1.0);
    assert!(near(near_point.z / near_point.w, 0.0));
    assert!(near(far_point.z / far_point.w, 1.0));
    // +y in view space is up the screen, which is -y in Vulkan clip space.
    let up = proj * Vec4::new(0.0, 1.0, -2.0, 1.0);
    assert!(up.y / up.w < 0.0);
}

#[test]
fn camera_looks_at_its_target() {
    let mut camera = Camera { position: Vec3::new(3.0, 2.0, 5.0), ..Camera::default() };
    camera.look_at(Vec3::ZERO, Vec3::Y);
    assert!(near_v(camera.forward(), (-camera.position).normalize(), 1e-4));
    // The target ends up straight ahead: on the view-space -Z axis.
    let target = camera.view() * Vec4::new(0.0, 0.0, 0.0, 1.0);
    assert!(near(target.x, 0.0) && near(target.y, 0.0) && target.z < 0.0);
    assert!(near(-target.z, camera.position.length()));
}

#[test]
fn transform_scales_then_rotates_then_translates() {
    let t = Transform {
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_axis_angle(Vec3::Y, 90f32.to_radians()),
        scale: Vec3::splat(2.0),
    };
    // (1, 0, 0) -> scaled (2, 0, 0) -> rotated about +Y to (0, 0, -2) -> translated.
    let p = (t.matrix() * Vec4::new(1.0, 0.0, 0.0, 1.0)).truncate();
    assert!(near_v(p, Vec3::new(1.0, 2.0, 1.0), 1e-4));
}

#[test]
fn meshes_validate_and_version() {
    let cube = Mesh::cube(2.0);
    assert_eq!(cube.triangle_count(), 12);
    assert!(near_v(cube.bounds().min, Vec3::splat(-1.0), 1e-4) && near_v(cube.bounds().max, Vec3::splat(1.0), 1e-4));
    let triangle = Mesh::triangle(1.0);
    assert_eq!(triangle.triangle_count(), 1);
    assert!(near(triangle.bounds().center().x, 0.0));

    let mut copy = cube.clone();
    assert_ne!(copy.id(), cube.id()); // a copy is a different mesh to caches
    let version = copy.version();
    copy.set_geometry(triangle.vertices().to_vec(), triangle.indices().to_vec()).unwrap();
    assert!(copy.version() != version && copy.triangle_count() == 1);
    let id = copy.id();
    let moved = std::mem::take(&mut copy); // a move keeps the id; what is left behind is empty with a new one
    assert!(moved.id() == id && copy.id() != id && copy.is_empty());

    assert!(matches!(Mesh::new(vec![Vertex::default()], vec![0, 0, 5]), Err(Error::IndexOutOfRange { .. })));
    assert!(matches!(Mesh::new(vec![Vertex::default()], vec![0, 0]), Err(Error::IndexCount(2))));
}

fn named(name: &str) -> ObjectDesc {
    ObjectDesc { name: name.into(), ..Default::default() }
}

#[test]
fn object_ids_survive_removal() {
    let mut scene = Scene::new();
    let [a, b, c] = scene.add_objects([named("a"), named("b"), named("c")]).unwrap();
    assert!(a.is_valid() && a != b && !ObjectId::default().is_valid() && !scene.contains(ObjectId::default()));
    assert!(scene.object_ids()[0] == a && scene.object_ids()[1] == b && scene.object_ids()[2] == c);

    // Removing from the middle moves another object into the hole; every other id still finds its object.
    assert!(scene.remove_object(a));
    assert!(!scene.contains(a) && scene.get(a).is_none());
    assert!(!scene.remove_object(a));
    assert_eq!(scene.object_count(), 2);
    assert!(scene.at(b).unwrap().name() == "b" && scene.at(c).unwrap().name() == "c");
    for i in 0..scene.object_count() {
        assert!(std::ptr::eq(scene.get(scene.object_ids()[i]).unwrap(), &scene.objects()[i]));
    }

    // The freed slot is reused with a new generation, so the stale id stays stale.
    let d = scene.add_object(named("d")).unwrap();
    assert!(d.index == a.index && d.generation != a.generation);
    assert!(!scene.contains(a) && scene.at(d).unwrap().name() == "d");
    assert!(matches!(scene.at(a), Err(Error::StaleObject { .. })));

    // Ids stay meaningful in a copy of the scene.
    let copy = scene.clone();
    assert!(copy.at(c).unwrap().name() == "c" && !copy.contains(a));

    assert!(scene.remove_object(c) && scene.remove_object(b) && scene.remove_object(d));
    assert!(scene.object_count() == 0 && !scene.contains(b));
}

fn two_cube_scene() -> Scene {
    // A red cube 2 units away and a green one 6 units away, both straight ahead, plus one behind the camera.
    let mut scene = Scene::new();
    let cube = Arc::new(Mesh::cube(1.0));
    scene
        .add_objects([
            desc(&cube, Vec4::new(0.0, 1.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -6.0)), // far
            desc(&cube, Vec4::new(1.0, 0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -2.0)), // near
            desc(&cube, Vec4::new(0.0, 0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 5.0)),  // behind
            ObjectDesc::default(),                                                  // no mesh
        ])
        .unwrap();
    scene.camera.position = Vec3::ZERO;
    scene.light.direction = Vec3::new(0.0, 0.0, -1.0); // shining straight at the cubes' front faces
    scene.light.ambient = 0.0;
    scene
}

#[test]
fn extraction_culls_and_sorts() {
    let scene = two_cube_scene();
    let view = extract_view(&scene, Extent2D::new(200, 100), true);
    assert_eq!(view.draws.len(), 2);
    assert_eq!(view.culled, 1);
    assert_eq!(view.draws[0].material.albedo.x, 1.0); // nearest first
    assert_eq!(view.draws[1].material.albedo.y, 1.0);
    assert!(near(view.draws[0].view_depth, 2.0) && near(view.draws[1].view_depth, 6.0));
    assert!(near_v(view.light_direction, Vec3::new(0.0, 0.0, 1.0), 1e-4));
    // The aspect ratio comes from the extent.
    assert!(near(view.proj.y_axis.y / view.proj.x_axis.x, -2.0));
}

#[test]
fn frustum_test_uses_every_plane() {
    let view_proj = perspective(60f32.to_radians(), 1.0, 0.1, 10.0);
    let unit = Bounds { min: Vec3::splat(-0.5), max: Vec3::splat(0.5) };
    let at = |p: Vec3| outside_frustum(&(view_proj * Mat4::from_translation(p)), &unit);
    assert!(!at(Vec3::new(0.0, 0.0, -3.0)));
    assert!(at(Vec3::new(0.0, 0.0, 3.0))); // behind
    assert!(at(Vec3::new(0.0, 0.0, -50.0))); // beyond the far plane
    assert!(at(Vec3::new(20.0, 0.0, -3.0))); // right
    assert!(at(Vec3::new(-20.0, 0.0, -3.0))); // left
    assert!(at(Vec3::new(0.0, 20.0, -3.0))); // above
    assert!(at(Vec3::new(0.0, -20.0, -3.0))); // below
    assert!(!at(Vec3::new(0.0, 0.0, -10.2))); // straddling the far plane is still drawn
}

#[test]
fn rasterizer_depth_tests_and_lights() {
    let scene = two_cube_scene();
    let image = rasterize(&extract_view(&scene, Extent2D::new(200, 100), true));
    assert!(image.width == 200 && image.height == 100);
    // The near red cube covers the far green one in the middle; its front face is fully lit, with a white
    // (dielectric) highlight adding the same to green and blue.
    let centre = image.pixel(100, 50);
    assert!(centre[0] == 255 && centre[1] == centre[2] && centre[1] > 0 && centre[1] < 160, "{centre:?}");
    // Corners show the background, sRGB (0.1, 0.2, 0.3): stored linear, encoded again on output.
    let corner = image.pixel(0, 0);
    assert!((corner[0] as i32 - 26).abs() <= 1 && (corner[1] as i32 - 51).abs() <= 1 && (corner[2] as i32 - 77).abs() <= 1, "{corner:?}");
}

/// The geometric normal (from the winding) of triangle `t` of `indices`.
fn face_normal(mesh: &Mesh, indices: &[u32], t: usize) -> Vec3 {
    let p = |k: usize| mesh.vertices()[indices[3 * t + k] as usize].position;
    (p(1) - p(0)).cross(p(2) - p(0))
}

#[test]
fn torus_is_closed_and_wound_outwards() {
    let torus = Mesh::torus(1.0, 0.25, 24, 12);
    assert_eq!(torus.triangle_count(), 24 * 12 * 2);
    assert!(near_v(torus.bounds().min, Vec3::new(-1.25, -0.25, -1.25), 1e-3));
    assert!(near_v(torus.bounds().max, Vec3::new(1.25, 0.25, 1.25), 1e-3));
    for t in 0..torus.triangle_count() {
        // Counter-clockwise from outside: the winding agrees with the vertex normals.
        assert!(face_normal(&torus, torus.indices(), t).dot(torus.vertices()[torus.indices()[3 * t] as usize].normal) > 0.0);
    }
}

/// A triangle as its three indices rotated so the smallest comes first: equal for the same triangle with the
/// same winding, wherever it starts.
fn canonical(a: u32, b: u32, c: u32) -> [u32; 3] {
    if b < a && b < c {
        [b, c, a]
    } else if c < a && c < b {
        [c, a, b]
    } else {
        [a, b, c]
    }
}

#[test]
fn meshlets_cover_every_triangle_once() {
    let torus = Mesh::torus(1.0, 0.3, 64, 32);
    let indices = optimize_index_order(&torus);
    assert_eq!(indices.len(), torus.indices().len());
    let meshlets = build_meshlets(&torus, &indices);
    assert!(meshlets.meshlets.len() >= torus.triangle_count() / MESHLET_MAX_TRIANGLES as usize);
    assert_eq!(meshlets.triangle_count(), torus.triangle_count());

    let mut expected: Vec<[u32; 3]> = torus.indices().chunks_exact(3).map(|t| canonical(t[0], t[1], t[2])).collect();
    let mut actual = Vec::new();
    for m in &meshlets.meshlets {
        assert!(m.vertex_count > 0 && m.vertex_count <= MESHLET_MAX_VERTICES);
        assert!(m.triangle_count > 0 && m.triangle_count <= MESHLET_MAX_TRIANGLES);
        assert!((m.vertex_offset + m.vertex_count) as usize <= meshlets.data.len());
        assert!((m.triangle_offset + m.triangle_count) as usize <= meshlets.data.len());
        for t in 0..m.triangle_count {
            let packed = meshlets.data[(m.triangle_offset + t) as usize];
            assert_eq!(packed >> 24, 0);
            let mut vertex = [0u32; 3];
            for (k, slot) in vertex.iter_mut().enumerate() {
                let local = (packed >> (8 * k)) & 0xFF;
                assert!(local < m.vertex_count);
                *slot = meshlets.data[(m.vertex_offset + local) as usize];
                // The bounding sphere holds every vertex.
                assert!((torus.vertices()[*slot as usize].position - m.sphere.truncate()).length() <= m.sphere.w + 1e-4);
            }
            actual.push(canonical(vertex[0], vertex[1], vertex[2]));
        }
    }
    // The same triangles with the same winding, each exactly once.
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected);
}

#[test]
fn cone_culling_is_conservative() {
    // Dense enough that meshlets are nearly flat, as in real models: their normal cones are then narrow.
    let torus = Mesh::torus(1.0, 0.3, 128, 64);
    let meshlets = build_meshlets(&torus, &optimize_index_order(&torus));
    let cameras = [Vec3::new(0.0, 0.0, 4.0), Vec3::new(3.0, 2.0, 0.5), Vec3::new(0.0, 5.0, 0.0), Vec3::new(-1.5, -0.2, -2.5)];
    for camera in cameras {
        let mut culled = 0;
        for m in &meshlets.meshlets {
            if !back_facing(m, camera) {
                continue;
            }
            culled += 1;
            // Every triangle of a culled meshlet faces away from the camera.
            for t in 0..m.triangle_count {
                let packed = meshlets.data[(m.triangle_offset + t) as usize];
                let triangle: Vec<u32> = (0..3).map(|k| meshlets.data[(m.vertex_offset + ((packed >> (8 * k)) & 0xFF)) as usize]).collect();
                let a = torus.vertices()[triangle[0] as usize].position;
                assert!(face_normal(&torus, &triangle, 0).dot(camera - a) <= 1e-6);
            }
        }
        // From outside, a good share of a smooth closed mesh faces away (at most half could).
        assert!(culled > meshlets.meshlets.len() / 5);
    }
}

#[test]
fn frustum_planes_bound_clip_space() {
    // Looking down -Z with a 90 degree field of view: the side planes are at 45 degrees.
    let view_proj = perspective(90f32.to_radians(), 1.0, 0.1, 10.0);
    let planes = frustum_planes(&view_proj);
    for plane in &planes {
        assert!(near(plane.truncate().length(), 1.0));
    }
    assert!(!sphere_outside_frustum(&planes, Vec3::new(0.0, 0.0, -5.0), 0.0));
    assert!(!sphere_outside_frustum(&planes, Vec3::new(4.9, -4.9, -5.0), 0.0));
    // Each plane: a point outside it, by d; a sphere of radius d around it just reaches back inside.
    let side = 0.5f32.sqrt(); // distance of (+-6, -5) from the plane x = +-z
    let outside = [
        (Vec3::new(-6.0, 0.0, -5.0), side),
        (Vec3::new(6.0, 0.0, -5.0), side),
        (Vec3::new(0.0, -6.0, -5.0), side),
        (Vec3::new(0.0, 6.0, -5.0), side),
        (Vec3::new(0.0, 0.0, -0.05), 0.05),
        (Vec3::new(0.0, 0.0, -10.5), 0.5),
    ];
    for (point, distance) in outside {
        assert!(sphere_outside_frustum(&planes, point, 0.0));
        assert!(sphere_outside_frustum(&planes, point, 0.9 * distance));
        assert!(!sphere_outside_frustum(&planes, point, 1.1 * distance));
    }
}

#[test]
fn projected_spheres_bound_their_silhouette() {
    let proj = perspective(70f32.to_radians(), 1.6, 0.1, 50.0);
    // Texture coordinates of a view-space point, as the GPU maps them (y down).
    let screen = |p: Vec3| {
        let clip = proj * p.extend(1.0);
        Vec2::new(clip.x, clip.y) / clip.w * 0.5 + Vec2::splat(0.5)
    };
    let spheres = [
        (Vec3::new(0.0, 0.0, -5.0), 1.0),
        (Vec3::new(3.0, -1.5, -4.0), 0.5),
        (Vec3::new(-6.0, 4.0, -3.0), 1.5), // largely off screen
        (Vec3::new(0.5, 0.2, -1.0), 0.8),  // close: a wide rectangle
        (Vec3::new(-1.0, 2.0, -20.0), 0.1),
    ];
    for (center, radius) in spheres {
        let bounds = project_sphere(center, radius, &proj).expect("the sphere is in front of the near plane");
        assert!(bounds.x < bounds.z && bounds.y < bounds.w);
        // Every point of the sphere projects inside; points on its silhouette reach every edge.
        let mut reached = Vec4::new(2.0, 2.0, -2.0, -2.0);
        for i in 0..64 {
            for j in 0..=32 {
                let theta = TWO_PI * i as f32 / 64.0;
                let phi = PI * j as f32 / 32.0;
                let p = center + radius * Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                let uv = screen(p);
                assert!(uv.x >= bounds.x - 1e-4 && uv.x <= bounds.z + 1e-4);
                assert!(uv.y >= bounds.y - 1e-4 && uv.y <= bounds.w + 1e-4);
                reached = Vec4::new(reached.x.min(uv.x), reached.y.min(uv.y), reached.z.max(uv.x), reached.w.max(uv.y));
            }
        }
        let tolerance = 0.02 * (bounds.z - bounds.x).max(bounds.w - bounds.y);
        assert!((reached - bounds).abs().max_element() <= tolerance);
    }
    assert!(project_sphere(Vec3::new(0.0, 0.0, -0.5), 0.45, &proj).is_none()); // reaches the near plane
    assert!(project_sphere(Vec3::new(0.0, 0.0, 2.0), 0.5, &proj).is_none()); // behind the camera
}

#[test]
fn depth_pyramid_is_powers_of_two() {
    assert_eq!(depth_pyramid_extent(Extent2D::new(1920, 1080)), Extent2D::new(1024, 1024));
    assert_eq!(depth_pyramid_extent(Extent2D::new(320, 240)), Extent2D::new(256, 128));
    assert_eq!(depth_pyramid_extent(Extent2D::new(512, 1)), Extent2D::new(512, 1));
    assert_eq!(depth_pyramid_levels(Extent2D::new(1024, 1024)), 11);
    assert_eq!(depth_pyramid_levels(Extent2D::new(256, 128)), 9);
    assert_eq!(depth_pyramid_levels(Extent2D::new(1, 1)), 1);
}

#[test]
fn objects_have_their_own_ids() {
    let mut scene = Scene::new();
    let [id_a, id_b] = scene.add_objects([named("a"), named("b")]).unwrap();
    let a = scene.at(id_a).unwrap().clone(); // a copy: a different object with its own id
    assert!(a.id() != 0 && a.id() != scene.at(id_b).unwrap().id());
    assert!(a.id() != scene.at(id_a).unwrap().id() && a.name() == "a");
    // Assigning keeps the target's own id.
    let id = scene.at(id_b).unwrap().id();
    scene.at_mut(id_b).unwrap().assign_from(&a);
    assert!(scene.at(id_b).unwrap().id() == id && scene.at(id_b).unwrap().name() == "a");

    // The scene moves objects around (growing, filling a removed one's slot); they keep their ids.
    let mut grown = Scene::new();
    let [first, last] = grown.add_objects([ObjectDesc::default(), ObjectDesc::default()]).unwrap();
    let last_id = grown.at(last).unwrap().id();
    for _ in 0..100 {
        grown.add_object(ObjectDesc::default()).unwrap();
    }
    assert_eq!(grown.at(last).unwrap().id(), last_id);
    let end = *grown.object_ids().last().unwrap();
    let end_id = grown.at(end).unwrap().id();
    grown.remove_object(first); // the last object moves into its place
    assert!(grown.at(end).unwrap().id() == end_id && grown.at(last).unwrap().id() == last_id);
}

#[test]
fn single_sided_materials_cull_back_faces() {
    // The triangle faces +Z; the camera looks at its back from -Z.
    let mut scene = Scene::new();
    let id = scene.add_object(desc(&Arc::new(Mesh::triangle(1.0)), Vec4::new(1.0, 0.0, 0.0, 1.0), Vec3::ZERO)).unwrap();
    scene.camera.position = Vec3::new(0.0, 0.0, -3.0);
    scene.camera.look_at(Vec3::ZERO, Vec3::Y);
    let extent = Extent2D::new(64, 64);
    let centre = |scene: &Scene| rasterize(&extract_view(scene, extent, true)).pixel(32, 32);
    assert!(!extract_view(&scene, extent, true).draws[0].cull_back_faces);
    assert!(centre(&scene)[0] > 0); // double-sided: the back face is drawn

    scene.at_mut(id).unwrap().material_mut().double_sided = false;
    assert!(extract_view(&scene, extent, true).draws[0].cull_back_faces);
    assert!((centre(&scene)[0] as i32 - 26).abs() <= 1); // background

    // A mirroring transform turns the front face into the back face, so nothing is culled.
    scene.at_mut(id).unwrap().transform_mut().scale = Vec3::new(-1.0, 1.0, 1.0);
    assert!(!extract_view(&scene, extent, true).draws[0].cull_back_faces);
}

#[test]
fn loads_obj_files() {
    let path = std::env::temp_dir().join("phox_rs_quad_test.obj");
    std::fs::write(&path, "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3 4\n").unwrap(); // one quad, no normals
    let quad = load_mesh(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(quad.triangle_count(), 2);
    assert_eq!(quad.vertices().len(), 4);
    assert!(near_v(quad.vertices()[0].normal.abs(), Vec3::Z, 1e-4)); // generated
    assert!(near_v(quad.bounds().min, Vec3::new(-1.0, -1.0, 0.0), 1e-4));
    assert!(load_mesh(&std::env::temp_dir().join("phox_rs_missing_file.obj")).is_err());
}

#[test]
fn textures_build_mip_chains() {
    assert!(mip_count_for(1, 1) == 1 && mip_count_for(256, 256) == 9 && mip_count_for(320, 240) == 9 && mip_count_for(1, 64) == 7);
    let default = SamplerDesc::default();
    // Black and white sRGB texels average to linear 0.5, which is sRGB 188, not 128.
    let checker = Texture::checker("bw", 4, 4, Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::ONE, TextureEncoding::Srgb, default).unwrap();
    assert_eq!(checker.mip_count(), 3);
    assert!(checker.mip(1).width == 2 && checker.mip(2).width == 1 && checker.mip(2).rgba.len() == 4);
    assert!(checker.mip(2).rgba[0] == 188 && checker.mip(2).rgba[3] == 255);
    // Linear data averages as stored.
    let data = Texture::checker("data", 2, 2, Vec4::ZERO, Vec4::ONE, TextureEncoding::Linear, default).unwrap();
    assert_eq!(data.mip(1).rgba[0], 128);
    // Normal maps are renormalised: two tilted normals average to straight up, not to a shorter vector.
    let mut tilted = Vec::new();
    for n in [Vec3::new(1.0, 0.0, 1.0).normalize(), Vec3::new(-1.0, 0.0, 1.0).normalize()] {
        for c in 0..3 {
            tilted.push(((n[c] * 0.5 + 0.5) * 255.0).round() as u8);
        }
        tilted.push(255);
    }
    let normals = Texture::new("normals", 2, 1, tilted, TextureEncoding::Normal, default, true).unwrap();
    assert_eq!(normals.mip(1).rgba[2], 255);
    // Odd sizes halve down to 1 x 1.
    let odd = Texture::new("odd", 5, 3, vec![7; 5 * 3 * 4], TextureEncoding::Linear, default, true).unwrap();
    assert!(odd.mip_count() == 3 && odd.mip(1).width == 2 && odd.mip(1).height == 1 && odd.mip(2).rgba[0] == 7);
    assert!(Texture::new("bad", 2, 2, vec![0; 3], TextureEncoding::Linear, default, true).is_err());
    // A copy is a different texture to caches.
    let copy = checker.clone();
    assert!(copy.id() != checker.id() && copy.mip_count() == checker.mip_count());
}

#[test]
fn textures_sample_like_the_gpu() {
    // 2 x 1: a black and a white texel. Texel centres are at u = 0.25 and 0.75.
    let rgba = vec![0, 0, 0, 255, 255, 255, 255, 255];
    let clamp_sampler = SamplerDesc { wrap_u: TextureWrap::ClampToEdge, wrap_v: TextureWrap::ClampToEdge, ..Default::default() };
    let clamp = Texture::new("clamp", 2, 1, rgba.clone(), TextureEncoding::Linear, clamp_sampler, true).unwrap();
    let s = |t: &Texture, u: f32| t.sample(Vec2::new(u, 0.5)).x;
    assert!(near(s(&clamp, 0.25), 0.0) && near(s(&clamp, 0.75), 1.0));
    assert!(near(s(&clamp, 0.5), 0.5));
    assert!(near(s(&clamp, 0.0), 0.0)); // clamped, not wrapped round to the white texel
    let repeat = Texture::new("repeat", 2, 1, rgba.clone(), TextureEncoding::Linear, SamplerDesc::default(), true).unwrap();
    assert!(near(s(&repeat, 0.0), 0.5)); // halfway between the last and the first texel
    assert!(near(s(&repeat, 1.25), 0.0));
    // sRGB texels are decoded before filtering.
    let srgb = Texture::new("srgb", 2, 1, vec![188, 188, 188, 255, 188, 188, 188, 255], TextureEncoding::Srgb, SamplerDesc::default(), true).unwrap();
    assert!((s(&srgb, 0.5) - 0.5).abs() < 0.01);
    let nearest_sampler = SamplerDesc { mag_filter: TextureFilter::Nearest, ..Default::default() };
    let nearest = Texture::new("nearest", 2, 1, rgba, TextureEncoding::Linear, nearest_sampler, true).unwrap();
    assert!(near(s(&nearest, 0.49), 0.0) && near(s(&nearest, 0.51), 1.0));
}

#[test]
fn meshes_generate_tangents() {
    // The plane's u runs along +x and v along +z, so its tangent is +x; increasing v (+z) is opposite to
    // cross(normal, tangent) = cross(+y, +x) = -z, so the handedness is -1.
    for v in Mesh::plane(2.0).vertices() {
        assert!(near_v(v.tangent.truncate(), Vec3::X, 1e-4) && v.tangent.w == -1.0);
    }
    // Generated tangents are unit length and perpendicular to the normal, everywhere on a torus.
    for v in Mesh::torus(1.0, 0.3, 16, 8).vertices() {
        assert!((v.tangent.truncate().length() - 1.0).abs() < 1e-3 && v.tangent.truncate().dot(v.normal).abs() < 1e-3);
        assert!(v.tangent.w == 1.0 || v.tangent.w == -1.0);
    }
    // Without usable texture coordinates a vertex still gets some unit tangent perpendicular to its normal.
    let flat = Mesh::new(
        vec![
            Vertex::new(Vec3::ZERO, Vec3::Z, Vec2::ZERO),
            Vertex::new(Vec3::X, Vec3::Z, Vec2::ZERO),
            Vertex::new(Vec3::Y, Vec3::Z, Vec2::ZERO),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    for v in flat.vertices() {
        assert!((v.tangent.truncate().length() - 1.0).abs() < 1e-3 && v.tangent.z.abs() < 1e-3);
    }
    // Given tangents are kept.
    let tangent = Vec4::new(0.0, 1.0, 0.0, 1.0);
    let given = Mesh::new(
        vec![
            Vertex { position: Vec3::ZERO, normal: Vec3::Z, uv: Vec2::new(0.0, 0.0), tangent },
            Vertex { position: Vec3::X, normal: Vec3::Z, uv: Vec2::new(1.0, 0.0), tangent },
            Vertex { position: Vec3::Y, normal: Vec3::Z, uv: Vec2::new(0.0, 1.0), tangent },
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    assert!(near_v(given.vertices()[0].tangent.truncate(), Vec3::Y, 1e-4));
}

#[test]
fn sphere_is_closed_and_matches_its_tangents() {
    let sphere = Mesh::sphere(0.5, 16, 8);
    assert!(near_v(sphere.bounds().min, Vec3::splat(-0.5), 1e-3) && near_v(sphere.bounds().max, Vec3::splat(0.5), 1e-3));
    // No degenerate triangles at the poles: 2 per quad, minus one per quad in the top and bottom rows.
    assert_eq!(sphere.triangle_count(), 16 * 8 * 2 - 2 * 16);
    for t in 0..sphere.triangle_count() {
        assert!(face_normal(&sphere, sphere.indices(), t).dot(sphere.vertices()[sphere.indices()[3 * t] as usize].normal) > 0.0);
    }
    // The analytic tangents agree with what generate_tangents works out from the texture coordinates (away
    // from the poles, where every direction is a tangent).
    let mut generated = sphere.vertices().to_vec();
    for v in &mut generated {
        v.tangent = Vec4::ZERO;
    }
    generate_tangents(&mut generated, sphere.indices());
    for (g, v) in generated.iter().zip(sphere.vertices()) {
        if v.normal.y.abs() > 0.9 {
            continue;
        }
        assert!(g.tangent.truncate().dot(v.tangent.truncate()) > 0.95 && g.tangent.w == v.tangent.w);
    }
}

#[test]
fn materials_pack_for_shaders() {
    let mut material = Material {
        name: "m".into(),
        albedo: Vec4::new(0.5, 0.25, 1.0, 0.5),
        metallic: 0.75,
        roughness: 0.3,
        emissive: Vec3::new(1.0, 2.0, 3.0),
        alpha_mode: AlphaMode::Mask,
        alpha_cutoff: 0.4,
        ..Material::default()
    };
    let data = material_data(&material);
    assert!(data.base_color == material.albedo && data.emissive == Vec4::new(1.0, 2.0, 3.0, 0.0));
    assert!(data.metallic == 0.75 && data.roughness == 0.3 && data.alpha_cutoff == 0.4);
    assert!(data.flags == MATERIAL_ALPHA_MASK && data.base_color_texture == NO_TEXTURE && data.emissive_texture == NO_TEXTURE);
    material.alpha_mode = AlphaMode::Blend;
    assert_eq!(material_data(&material).flags, MATERIAL_ALPHA_BLEND);
    material.alpha_mode = AlphaMode::Opaque;
    assert_eq!(material_data(&material).flags, 0);
}

#[test]
fn blended_items_are_drawn_last_and_farthest_first() {
    let mut scene = Scene::new();
    let cube = Arc::new(Mesh::cube(1.0));
    let glass = Material { albedo: Vec4::new(0.0, 0.0, 1.0, 0.5), alpha_mode: AlphaMode::Blend, ..Material::default() };
    let glass_at = |z: f32| ObjectDesc { mesh: Some(cube.clone()), material: glass.clone(), transform: Transform::at(Vec3::new(0.0, 0.0, z)), ..Default::default() };
    scene
        .add_objects([
            glass_at(-2.0),
            glass_at(-5.0),
            // a red wall
            ObjectDesc {
                mesh: Some(cube.clone()),
                material: Material { albedo: Vec4::new(1.0, 0.0, 0.0, 1.0), ..Material::default() },
                transform: Transform { position: Vec3::new(0.0, 0.0, -8.0), scale: Vec3::new(4.0, 4.0, 0.1), ..Default::default() },
                ..Default::default()
            },
        ])
        .unwrap();
    scene.camera.position = Vec3::ZERO;
    scene.camera.look_at(Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
    let view = extract_view(&scene, Extent2D::new(64, 64), true);
    assert!(view.draws.len() == 1 && view.transparent.len() == 2 && view.item_count() == 3);
    assert!(view.transparent[0].view_depth > view.transparent[1].view_depth);
    // Blended over the red wall, the two half-transparent blue cubes leave some red showing.
    let centre = rasterize(&view).pixel(32, 32);
    assert!(centre[2] > centre[1] && centre[0] > 0 && centre[0] < 200, "{centre:?}");
}

#[test]
fn masked_texels_are_cut() {
    // A white quad whose texture is transparent on its left half, in front of a red wall.
    let mut scene = Scene::new();
    let rgba = vec![255, 255, 255, 0, 255, 255, 255, 255];
    let mask = Arc::new(
        Texture::new("mask", 2, 1, rgba, TextureEncoding::Srgb, SamplerDesc { mag_filter: TextureFilter::Nearest, ..Default::default() }, false).unwrap(),
    );
    let upright = Quat::from_axis_angle(Vec3::X, HALF_PI);
    scene
        .add_objects([
            // the cutout
            ObjectDesc {
                mesh: Some(Arc::new(Mesh::plane(2.0))),
                material: Material { albedo: Vec4::ONE, alpha_mode: AlphaMode::Mask, base_color_texture: Some(mask), ..Material::default() },
                transform: Transform { position: Vec3::new(0.0, 0.0, -2.0), rotation: upright, ..Default::default() },
                ..Default::default()
            },
            // the wall
            ObjectDesc {
                mesh: Some(Arc::new(Mesh::cube(1.0))),
                material: Material { albedo: Vec4::new(1.0, 0.0, 0.0, 1.0), ..Material::default() },
                transform: Transform { position: Vec3::new(0.0, 0.0, -6.0), scale: Vec3::new(8.0, 8.0, 0.1), ..Default::default() },
                ..Default::default()
            },
        ])
        .unwrap();
    scene.camera.position = Vec3::ZERO;
    scene.camera.look_at(Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
    let image = rasterize(&extract_view(&scene, Extent2D::new(64, 64), true));
    assert!(image.pixel(20, 32)[1] < 100); // left: cut away, the red wall shows
    assert!(image.pixel(44, 32)[1] > 200); // right: the white quad
}

/// A small glTF file with a node hierarchy, a mesh used twice, a primitive without normals or material, and a
/// textured, alpha-masked material with an emissive strength (embedded PNG: red, green / blue, and white at
/// half alpha). It also lists an extension Phox ignores.
const TEST_GLTF: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_emissive_strength", "KHR_materials_clearcoat"],
  "scene": 0,
  "scenes": [{"nodes": [0, 2, 3]}],
  "nodes": [
    {"name": "parent", "translation": [0, 0, -2], "children": [1]},
    {"name": "child", "mesh": 0, "scale": [2, 2, 2]},
    {"name": "tri", "mesh": 1, "rotation": [0, 0.70710678, 0, 0.70710678]},
    {"name": "again", "mesh": 0, "translation": [3, 0, 0]}
  ],
  "meshes": [
    {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 4, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]},
    {"primitives": [{"attributes": {"POSITION": 3}}]}
  ],
  "materials": [{
    "name": "painted",
    "pbrMetallicRoughness": {"baseColorFactor": [0.5, 1, 1, 1], "baseColorTexture": {"index": 0},
                             "metallicFactor": 0.25, "roughnessFactor": 0.75},
    "emissiveFactor": [1, 0.5, 0],
    "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 2}},
    "alphaMode": "MASK", "alphaCutoff": 0.3, "doubleSided": true
  }],
  "textures": [{"source": 0, "sampler": 0}],
  "samplers": [{"magFilter": 9728, "minFilter": 9729, "wrapS": 33071, "wrapT": 33648}],
  "images": [{"uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAE0lEQVR4nGP4z8DwHwyBNAg0AABJSQl4KKDbdwAAAABJRU5ErkJggg=="}],
  "buffers": [{"byteLength": 180, "uri": "data:application/octet-stream;base64,AACAvwAAgL8AAAAAAACAPwAAgL8AAAAAAACAPwAAgD8AAAAAAACAvwAAgD8AAAAAAAAAAAAAgD8AAIA/AACAPwAAgD8AAAAAAAAAAAAAAAAAAAEAAgAAAAIAAwAAAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/"}],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 48},
    {"buffer": 0, "byteOffset": 48, "byteLength": 32},
    {"buffer": 0, "byteOffset": 80, "byteLength": 12},
    {"buffer": 0, "byteOffset": 96, "byteLength": 36},
    {"buffer": 0, "byteOffset": 132, "byteLength": 48}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
    {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC3"}
  ]
}"#;

#[test]
fn loads_gltf_files() {
    let path = std::env::temp_dir().join("phox_rs_gltf_test.gltf");
    std::fs::write(&path, TEST_GLTF).unwrap();
    let mut scene = Scene::new();
    let import = load_gltf(&path, &mut scene, &Mat4::IDENTITY);
    std::fs::remove_file(&path).unwrap();
    let import = import.unwrap();

    assert!(import.objects.len() == 3 && scene.object_count() == 3);
    assert!(import.meshes == 2 && import.materials == 2 && import.textures == 1, "{import:?}"); // the quad mesh is shared
    assert!(import.warnings.len() == 1 && import.warnings[0].contains("KHR_materials_clearcoat"), "{:?}", import.warnings);
    // World bounds: the child quad (scaled by 2, at z = -2), the triangle, the second quad at x = 3.
    assert!(near_v(import.bounds.min, Vec3::new(-2.0, -2.0, -2.0), 1e-4) && near_v(import.bounds.max, Vec3::new(4.0, 2.0, 0.0), 1e-4));

    let child = scene.at(import.objects[0]).unwrap();
    assert_eq!(child.name(), "child");
    assert!(near_v(child.transform().position, Vec3::new(0.0, 0.0, -2.0), 1e-4) && near_v(child.transform().scale, Vec3::splat(2.0), 1e-4));
    let quad = child.mesh().unwrap();
    assert!(quad.vertices().len() == 4 && quad.triangle_count() == 2);
    assert!(near_v(quad.vertices()[0].normal, Vec3::Z, 1e-4) && near(quad.vertices()[2].uv.x, 1.0));
    assert!(near_v(quad.vertices()[0].tangent.truncate(), Vec3::X, 1e-4)); // generated: u runs along +x
    assert!(Arc::ptr_eq(scene.at(import.objects[2]).unwrap().mesh().unwrap(), quad));

    let painted = child.material();
    assert!(painted.name == "painted" && painted.albedo == Vec4::new(0.5, 1.0, 1.0, 1.0));
    assert!(painted.metallic == 0.25 && painted.roughness == 0.75 && near_v(painted.emissive, Vec3::new(2.0, 1.0, 0.0), 1e-4));
    assert!(painted.alpha_mode == AlphaMode::Mask && near(painted.alpha_cutoff, 0.3) && painted.double_sided);
    assert!(painted.normal_texture.is_none() && painted.metallic_roughness_texture.is_none());
    let texture = painted.base_color_texture.as_ref().unwrap();
    assert!(texture.width() == 2 && texture.height() == 2 && texture.encoding() == TextureEncoding::Srgb);
    assert_eq!(texture.mip_count(), 1); // the sampler's minification filter does not use mips
    assert!(texture.sampler().mag_filter == TextureFilter::Nearest && texture.sampler().wrap_u == TextureWrap::ClampToEdge);
    assert_eq!(texture.sampler().wrap_v, TextureWrap::MirroredRepeat);
    let texels = &texture.mip(0).rgba;
    assert!(texels[0] == 255 && texels[1] == 0 && texels[4 + 1] == 255 && texels[8 + 2] == 255 && texels[12 + 3] == 128);

    // No normals: flat ones, one set of vertices per triangle. No material: glTF's default.
    let tri = scene.at(import.objects[1]).unwrap();
    assert!(tri.mesh().unwrap().vertices().len() == 3 && near_v(tri.mesh().unwrap().vertices()[1].normal, Vec3::Z, 1e-4));
    assert!(near_v(tri.transform().rotation * Vec3::X, Vec3::new(0.0, 0.0, -1.0), 1e-4));
    assert!(tri.material().metallic == 1.0 && tri.material().roughness == 1.0 && !tri.material().double_sided);

    assert!(load_gltf(&std::env::temp_dir().join("phox_rs_no_such_file.gltf"), &mut scene, &Mat4::IDENTITY).is_err());
}

/// The triangles (as mesh vertex indices) of the meshlets `lod_selected` picks for `camera`.
fn selected_triangles(lod: &MeshletMesh, camera: Vec3, pixels_per_unit: f32, threshold: f32) -> Vec<[u32; 3]> {
    let mut triangles = Vec::new();
    for meshlet in &lod.meshlets {
        if !lod_selected(meshlet, camera, pixels_per_unit, threshold) {
            continue;
        }
        for t in 0..meshlet.triangle_count {
            let packed = lod.data[(meshlet.triangle_offset + t) as usize];
            triangles.push(std::array::from_fn(|k| lod.data[(meshlet.vertex_offset + ((packed >> (8 * k)) & 0xFF)) as usize]));
        }
    }
    triangles
}

/// Edges (by vertex position) used by other than exactly one triangle in each direction: zero for a closed
/// surface without cracks or overlaps.
fn bad_edges(mesh: &Mesh, triangles: &[[u32; 3]]) -> usize {
    let mut ids: BTreeMap<[u32; 3], u32> = BTreeMap::new();
    let mut id = |v: u32| {
        let p = mesh.vertices()[v as usize].position;
        let next = ids.len() as u32;
        *ids.entry([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).or_insert(next)
    };
    let mut directed: BTreeMap<(u32, u32), i32> = BTreeMap::new();
    for t in triangles {
        let (a, b, c) = (id(t[0]), id(t[1]), id(t[2]));
        if a == b || b == c || a == c {
            continue; // degenerate: no area, no edges
        }
        *directed.entry((a, b)).or_default() += 1;
        *directed.entry((b, c)).or_default() += 1;
        *directed.entry((c, a)).or_default() += 1;
    }
    directed.iter().filter(|&(&(a, b), &count)| count != 1 || directed.get(&(b, a)) != Some(&1)).count()
}

#[test]
fn cluster_lod_is_crack_free() {
    // A dense closed mesh, as in the "field" scene, with texture seams (the torus duplicates its seam vertices).
    let torus = Mesh::torus(0.6, 0.22, 128, 64);
    let original: Vec<[u32; 3]> = torus.indices().chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    assert_eq!(bad_edges(&torus, &original), 0); // closed to begin with, seams included
    let lod = build_meshlet_lod(&torus, &optimize_index_order(&torus));
    assert!(lod.levels >= 4);
    assert_eq!(lod.triangle_count_at(0), torus.triangle_count());
    assert!(lod.triangle_count_at(1) < torus.triangle_count() * 3 / 4); // each level about halves the triangles
    assert_eq!(bad_edges(&torus, &selected_triangles(&lod, Vec3::ZERO, 1000.0, 0.0)), 0);
    for meshlet in &lod.meshlets {
        // Errors grow towards coarser levels, and a parent's sphere contains its children's.
        assert!(meshlet.parent_error >= meshlet.lod_error);
        if meshlet.parent_error < LOD_NO_PARENT {
            let gap = (meshlet.parent_sphere.truncate() - meshlet.lod_sphere.truncate()).length();
            assert!(gap + meshlet.lod_sphere.w <= meshlet.parent_sphere.w * 1.0001 + 1e-5);
        }
        assert_eq!(meshlet.level == 0, meshlet.lod_error == 0.0);
    }

    // From close up to far away: always closed, and fewer triangles the farther the camera.
    let mut previous = torus.triangle_count() + 1;
    for distance in [0.5, 2.0, 8.0, 32.0, 128.0, 1024.0] {
        let camera = Vec3::new(0.3, 0.8, 0.5).normalize() * distance;
        let triangles = selected_triangles(&lod, camera, 540.0, 1.0);
        assert_eq!(bad_edges(&torus, &triangles), 0);
        assert!(!triangles.is_empty() && triangles.len() <= previous);
        previous = triangles.len();
    }
    assert!(previous < torus.triangle_count() / 50); // far away, a few dozen triangles do

    // Without a hierarchy every meshlet is level 0 and always drawn.
    let flat = build_meshlets(&torus, &optimize_index_order(&torus));
    for meshlet in &flat.meshlets {
        assert!(meshlet.level == 0 && meshlet.parent_error >= LOD_NO_PARENT);
        assert!(lod_selected(meshlet, Vec3::splat(1000.0), 540.0, 1.0));
    }
}
