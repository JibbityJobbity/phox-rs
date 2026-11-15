//! A renderer keeps the scene on the GPU and updates only what changed. After each kind of change, a
//! renderer that saw the old scene must draw exactly what a fresh renderer draws. Ported from
//! test/source/scene_test.cpp (the in-place geometry edit is not ported: a mesh behind an `Arc` cannot be
//! edited in place, so a changed mesh is always a different one).

use phox_core::cpu_image::{CpuImage, compare_images};
use phox_core::math::{Vec3, Vec4};
use phox_core::resources::Extent2D;
use phox_core::sample_scenes::*;
use phox_core::scene::{AlphaMode, Material, Mesh, ObjectDesc, Scene, Transform};
use phox_forward::{ForwardRenderer, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use serial_test::serial;
use std::sync::Arc;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn device(draw_indirect_count: bool) -> Option<VulkanDevice> {
    let _ = env_logger::Builder::from_env(env_logger::Env::new().filter_or("PHOX_LOG", "info")).try_init();
    let config = VulkanConfig { validation: flag("PHOX_VALIDATION"), draw_indirect_count, ..VulkanConfig::default() };
    match VulkanDevice::new(&config) {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            None
        }
    }
}

fn render_frames(renderer: &mut ForwardRenderer, scene: &Scene, frames: usize) -> CpuImage {
    for _ in 0..frames {
        renderer.render(scene).unwrap();
    }
    renderer.last_frame().clone()
}

/// Checks that `seasoned`, which has drawn earlier versions of `scene`, now draws what a fresh renderer does.
fn expect_like_fresh(vulkan: &VulkanDevice, seasoned: &mut ForwardRenderer, scene: &Scene, config: &RendererConfig, what: &str) {
    let changed = render_frames(seasoned, scene, 3);
    let mut fresh = ForwardRenderer::new(Arc::new(vulkan.clone()), config.clone()).unwrap();
    let expected = render_frames(&mut fresh, scene, 3);
    let diff = compare_images(&changed, &expected, 2);
    println!(
        "{what} ({}): {} pixel(s) unlike a fresh renderer's",
        if seasoned.uses_mesh_shaders() { "mesh shaders" } else { "vertex shaders" },
        diff.pixels_over_tolerance
    );
    assert!(!diff.size_mismatch && diff.pixels_over_tolerance == 0, "{what}: {diff:?}");
}

fn changing_scenes_draw_like_fresh_ones(vulkan: &VulkanDevice, mesh_shaders: bool) {
    let config = RendererConfig { extent: Extent2D::new(160, 120), mesh_shaders, ..RendererConfig::default() };
    let mut seasoned = ForwardRenderer::new(Arc::new(vulkan.clone()), config.clone()).unwrap();
    let mut sample = make_sample_scene(SampleSceneId::Cubes).unwrap();
    (sample.animate)(&mut sample.scene, 0.3);
    render_frames(&mut seasoned, &sample.scene, 4);
    let ids = sample.scene.object_ids().to_vec();
    assert_eq!(ids.len(), 6);
    let [floor, red, green, blue, yellow, purple] = [ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]];

    // Two objects go, three arrive (reusing their slots, then one more), one changes mesh.
    let scene = &mut sample.scene;
    assert!(scene.remove_object(red) && scene.remove_object(green));
    let torus = Arc::new(Mesh::torus(0.4, 0.15, 32, 16));
    let cyan = Material { albedo: Vec4::new(0.2, 0.8, 0.9, 1.0), double_sided: false, ..Material::default() };
    let at = |x: f32| ObjectDesc { mesh: Some(torus.clone()), material: cyan.clone(), transform: Transform::at(Vec3::new(x, 0.3, 0.6)), ..Default::default() };
    let [arrival, second, third] = scene.add_objects([at(-1.5), at(0.0), at(1.5)]).unwrap();
    assert!(scene.contains(second) && scene.contains(third));
    scene.at_mut(blue).unwrap().set_mesh(Some(Arc::new(Mesh::sphere(0.6, 48, 24))));
    expect_like_fresh(vulkan, &mut seasoned, scene, &config, "changed objects");

    // In place: a material recoloured, one made translucent, one moved, one given a mesh of more meshlets.
    scene.at_mut(yellow).unwrap().material_mut().albedo = Vec4::new(0.1, 0.9, 0.2, 1.0);
    {
        let mut material = scene.at_mut(purple).unwrap().material_mut();
        material.alpha_mode = AlphaMode::Blend;
        material.albedo.w = 0.5;
    }
    scene.at_mut(arrival).unwrap().transform_mut().position.y += 0.5;
    scene.at_mut(second).unwrap().set_mesh(Some(Arc::new(Mesh::sphere(0.5, 64, 32))));
    expect_like_fresh(vulkan, &mut seasoned, scene, &config, "edited objects");

    // Back to opaque, and the floor loses its mesh.
    scene.at_mut(purple).unwrap().material_mut().alpha_mode = AlphaMode::Opaque;
    scene.at_mut(floor).unwrap().set_mesh(None);
    expect_like_fresh(vulkan, &mut seasoned, scene, &config, "edited again");

    // A crowd arrives: more slots and visibility words than the renderer has, so it regrows every buffer
    // and starts the visibility bits over in the middle of the session.
    let pebble = Arc::new(Mesh::cube(0.02));
    for i in 0..20000 {
        let (x, y) = ((i % 200) as f32 * 0.03 - 3.0, (i / 200) as f32 * 0.03 - 1.0);
        scene
            .add_object(ObjectDesc {
                mesh: Some(pebble.clone()),
                material: Material { albedo: Vec4::new(0.9, 0.9, 0.9, 1.0), ..Material::default() },
                transform: Transform::at(Vec3::new(x, y, -2.0)),
                ..Default::default()
            })
            .unwrap();
    }
    expect_like_fresh(vulkan, &mut seasoned, scene, &config, "crowd arrived");
}

#[test]
#[serial]
fn mesh_shaders_keep_up_with_a_changing_scene() {
    let Some(vulkan) = device(true) else { return };
    if !vulkan.info().mesh_shaders {
        assert!(!flag("PHOX_REQUIRE_MESH_SHADERS"), "no mesh shaders");
        return;
    }
    changing_scenes_draw_like_fresh_ones(&vulkan, true);
    assert_eq!(vulkan.validation_counts(), (0, 0));
}

#[test]
#[serial]
fn vertex_shaders_keep_up_with_a_changing_scene() {
    let Some(vulkan) = device(true) else { return };
    changing_scenes_draw_like_fresh_ones(&vulkan, false);
    assert_eq!(vulkan.validation_counts(), (0, 0));
}

#[test]
#[serial]
fn vertex_shaders_without_a_draw_count_keep_up_too() {
    // The path for devices without drawIndirectCount (MoltenVK): the command buffer is cleared first and
    // drawn whole, its unused commands drawing nothing.
    let Some(vulkan) = device(false) else { return };
    assert!(!vulkan.info().draw_indirect_count);
    changing_scenes_draw_like_fresh_ones(&vulkan, false);
    assert_eq!(vulkan.validation_counts(), (0, 0));
}

#[test]
#[serial]
fn a_still_scene_uploads_nothing_after_the_first_frame() {
    let Some(vulkan) = device(true) else { return };
    let mut renderer = ForwardRenderer::new(Arc::new(vulkan.clone()), RendererConfig { extent: Extent2D::new(64, 48), capture_graph: true, ..RendererConfig::default() }).unwrap();
    let sample = make_sample_scene(SampleSceneId::Cubes).unwrap();
    renderer.render(&sample.scene).unwrap();
    assert!(renderer.frame_graph_dot().contains("scene update"));
    renderer.render(&sample.scene).unwrap();
    assert!(!renderer.frame_graph_dot().contains("scene update"), "nothing changed, so nothing is written");
}
