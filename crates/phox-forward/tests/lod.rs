//! Cluster LOD on the GPU path: the statistics the renderer reports, and that LOD draws fewer triangles the
//! farther away a mesh is while the picture stays close to full detail.

use phox_core::cpu_image::compare_images;
use phox_core::math::{Vec3, Vec4};
use phox_core::resources::Extent2D;
use phox_core::scene::{Material, Mesh, ObjectDesc, Scene, Transform};
use phox_forward::{ForwardRenderer, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use serial_test::serial;
use std::sync::Arc;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn device() -> Option<VulkanDevice> {
    match VulkanDevice::new(&VulkanConfig { validation: flag("PHOX_VALIDATION"), ..VulkanConfig::default() }) {
        Ok(d) if d.info().mesh_shaders => Some(d),
        Ok(_) => {
            assert!(!flag("PHOX_REQUIRE_MESH_SHADERS"), "no mesh shaders");
            None
        }
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            None
        }
    }
}

fn torus_scene(distance: f32) -> Scene {
    let mut scene = Scene::new();
    scene
        .add_object(ObjectDesc {
            mesh: Some(Arc::new(Mesh::torus(0.6, 0.22, 128, 64))),
            material: Material { albedo: Vec4::new(0.8, 0.5, 0.2, 1.0), double_sided: false, ..Material::default() },
            transform: Transform::at(Vec3::ZERO),
            ..Default::default()
        })
        .unwrap();
    scene.camera.position = Vec3::new(0.0, 0.0, distance);
    scene.camera.look_at(Vec3::ZERO, Vec3::Y);
    scene
}

#[test]
#[serial]
fn lod_draws_fewer_triangles_the_farther_the_mesh() {
    let Some(vulkan) = device() else { return };
    let config = |lod| RendererConfig { extent: Extent2D::new(320, 240), mesh_lod: lod, collect_statistics: true, ..RendererConfig::default() };
    let mut lod = ForwardRenderer::new(Arc::new(vulkan.clone()), config(true)).unwrap();
    let mut full = ForwardRenderer::new(Arc::new(vulkan.clone()), config(false)).unwrap();
    assert!(lod.uses_mesh_lod() && !full.uses_mesh_lod());

    let mut previous = u64::MAX;
    for distance in [2.5, 6.0, 20.0, 80.0] {
        let scene = torus_scene(distance);
        lod.render(&scene).unwrap();
        full.render(&scene).unwrap();
        let (a, b) = (lod.frame_statistics(), full.frame_statistics());
        println!("distance {distance}: {} of {} triangles ({} without LOD)", a.lod_triangles, a.triangles, b.lod_triangles);
        assert_eq!((a.draws, a.triangles), (1, 128 * 64 * 2));
        assert_eq!(b.lod_triangles, b.triangles); // without LOD every triangle is a candidate
        assert!(a.lod_triangles <= previous, "farther away, no more triangles");
        previous = a.lod_triangles;
        // Within its one-pixel error budget the picture stays close to the full-detail one.
        let diff = compare_images(lod.last_frame(), full.last_frame(), 8);
        let allowed = (0.03 * 320.0 * 240.0) as usize;
        assert!(diff.pixels_over_tolerance <= allowed, "distance {distance}: {diff:?}");
    }
    assert!(previous < 128 * 64 * 2 / 20, "far away, a few hundred triangles do ({previous})");
    drop((lod, full));
    assert_eq!(vulkan.validation_counts(), (0, 0));
}

#[test]
#[serial]
fn a_larger_error_budget_draws_fewer_triangles() {
    let Some(vulkan) = device() else { return };
    let scene = torus_scene(6.0);
    let mut counts = Vec::new();
    for error in [0.25, 1.0, 8.0] {
        let config = RendererConfig { extent: Extent2D::new(320, 240), lod_error_pixels: error, collect_statistics: true, ..RendererConfig::default() };
        let mut renderer = ForwardRenderer::new(Arc::new(vulkan.clone()), config).unwrap();
        renderer.render(&scene).unwrap();
        counts.push(renderer.frame_statistics().lod_triangles);
    }
    println!("triangles at 0.25, 1 and 8 pixels of error: {counts:?}");
    assert!(counts[0] >= counts[1] && counts[1] > counts[2], "{counts:?}");
}
