//! The mesh shader path's two-phase occlusion culling against the vertex path, on a scene built to need it:
//! tori hidden behind a wall, moving, with one emerging from behind it. Culling must never change the
//! picture, whatever the visibility bits remember from earlier frames.

use phox_core::cpu_image::compare_images;
use phox_core::math::{Quat, Vec3, Vec4};
use phox_core::resources::Extent2D;
use phox_core::scene::{Material, Mesh, ObjectDesc, ObjectId, Scene, Transform};
use phox_forward::{ForwardRenderer, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use serial_test::serial;
use std::sync::Arc;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn device() -> Option<VulkanDevice> {
    let _ = env_logger::Builder::from_env(env_logger::Env::new().filter_or("PHOX_LOG", "info")).try_init();
    match VulkanDevice::new(&VulkanConfig { validation: flag("PHOX_VALIDATION"), ..VulkanConfig::default() }) {
        Ok(d) if d.info().mesh_shaders => Some(d),
        Ok(d) => {
            assert!(!flag("PHOX_REQUIRE_MESH_SHADERS"), "{} has no mesh shaders", d.info().name);
            None
        }
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            None
        }
    }
}

struct Occlusion {
    scene: Scene,
    movers: Vec<ObjectId>,
}

fn occlusion_scene() -> Occlusion {
    let mut scene = Scene::new();
    let torus = Arc::new(Mesh::torus(0.5, 0.2, 48, 24));
    let wall = scene
        .add_object(ObjectDesc {
            mesh: Some(Arc::new(Mesh::cube(1.0))),
            material: Material { albedo: Vec4::new(0.7, 0.7, 0.75, 1.0), double_sided: false, ..Material::default() },
            transform: Transform { position: Vec3::new(0.0, 0.0, 1.0), scale: Vec3::new(6.0, 4.0, 0.2), ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    let _ = wall;
    let mut movers = Vec::new();
    for i in 0..6 {
        let x = -2.5 + i as f32;
        movers.push(
            scene
                .add_object(ObjectDesc {
                    mesh: Some(torus.clone()),
                    material: Material { albedo: Vec4::new(0.9, 0.4 + 0.1 * i as f32, 0.2, 1.0), double_sided: false, ..Material::default() },
                    transform: Transform::at(Vec3::new(x, 0.0, -1.5)),
                    ..Default::default()
                })
                .unwrap(),
        );
    }
    scene.camera.position = Vec3::new(0.0, 0.5, 6.0);
    scene.camera.look_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
    Occlusion { scene, movers }
}

#[test]
#[serial]
fn the_two_phase_path_draws_what_the_vertex_path_draws() {
    let Some(vulkan) = device() else { return };
    let extent = Extent2D::new(200, 150);
    let config = |mesh_shaders| RendererConfig { extent, mesh_shaders, capture_graph: true, ..RendererConfig::default() };
    let mut vertex = ForwardRenderer::new(Arc::new(vulkan.clone()), config(false)).unwrap();
    let mut mesh = ForwardRenderer::new(Arc::new(vulkan.clone()), config(true)).unwrap();
    assert!(mesh.uses_mesh_shaders() && mesh.uses_occlusion_culling());

    let Occlusion { mut scene, movers } = occlusion_scene();
    for frame in 0..12 {
        // The tori slide up from behind the wall and spin, so the early phase's memory goes stale and
        // meshlets come and go between frames.
        for (i, &id) in movers.iter().enumerate() {
            let t = frame as f32 * 0.25 - i as f32 * 0.4;
            let object = scene.at_mut(id).unwrap();
            let mut transform = object.transform_mut();
            transform.position.y = (t - 1.0).clamp(-3.0, 2.5);
            transform.rotation = Quat::from_axis_angle(Vec3::new(1.0, 0.3, 0.0).normalize(), 0.4 * t);
        }
        vertex.render(&scene).unwrap();
        mesh.render(&scene).unwrap();
        let diff = compare_images(mesh.last_frame(), vertex.last_frame(), 2);
        assert!(!diff.size_mismatch && diff.pixels_over_tolerance == 0, "frame {frame}: {diff:?}");
    }

    // The mesh path is the two-phase one: early, pyramid, late (then tonemap and readback).
    let dot = mesh.frame_graph_dot();
    for needle in ["forward early", "depth pyramid", "forward late", "scene history", "meshlet visibility", "early work"] {
        assert!(dot.contains(needle), "missing '{needle}' in:\n{dot}");
    }
    assert!(!vertex.frame_graph_dot().contains("depth pyramid"));
    let timings: Vec<&str> = mesh.gpu_timings().iter().map(|t| t.pass.as_str()).collect();
    // Objects move every frame, so the scene update runs; then the cull passes bracket the two phases.
    assert_eq!(
        timings,
        [
            "scene update",
            "clear draw counters",
            "cull early",
            "draw args early",
            "forward early",
            "depth pyramid",
            "cull late",
            "draw args late",
            "forward late",
            "tonemap",
            "readback"
        ]
    );

    drop((vertex, mesh));
    let (errors, warnings) = vulkan.validation_counts();
    assert_eq!((errors, warnings), (0, 0));
}

#[test]
#[serial]
fn the_visibility_store_survives_a_crowd_arriving() {
    let Some(vulkan) = device() else { return };
    let extent = Extent2D::new(160, 120);
    let config = |mesh_shaders| RendererConfig { extent, mesh_shaders, ..RendererConfig::default() };
    let mut vertex = ForwardRenderer::new(Arc::new(vulkan.clone()), config(false)).unwrap();
    let mut mesh = ForwardRenderer::new(Arc::new(vulkan.clone()), config(true)).unwrap();
    let Occlusion { mut scene, .. } = occlusion_scene();
    mesh.render(&scene).unwrap();
    // Thousands of small objects at once: more visibility words than the store has, so it starts over
    // (every bit cleared) in the middle of the session.
    let pebble = Arc::new(Mesh::cube(0.02));
    for i in 0..3000 {
        let (x, y) = ((i % 60) as f32 * 0.07 - 2.0, (i / 60) as f32 * 0.07 - 1.5);
        scene
            .add_object(ObjectDesc {
                mesh: Some(pebble.clone()),
                material: Material { albedo: Vec4::new(0.9, 0.9, 0.2, 1.0), ..Material::default() },
                transform: Transform::at(Vec3::new(x, y, 3.0)),
                ..Default::default()
            })
            .unwrap();
    }
    for _ in 0..3 {
        vertex.render(&scene).unwrap();
        mesh.render(&scene).unwrap();
        let diff = compare_images(mesh.last_frame(), vertex.last_frame(), 2);
        assert_eq!(diff.pixels_over_tolerance, 0, "{diff:?}");
    }
    drop((vertex, mesh));
    assert_eq!(vulkan.validation_counts(), (0, 0));
}
