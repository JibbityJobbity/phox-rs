//! A glTF file with a node hierarchy, a shared mesh, a textured alpha-masked material and a mesh without
//! normals, loaded and drawn on the GPU, then compared with the software rasterizer.

use phox_core::cpu_image::compare_images;
use phox_core::gltf_loader::load_gltf;
use phox_core::math::{Mat4, Vec3};
use phox_core::render_view::extract_view;
use phox_core::resources::Extent2D;
use phox_core::scene::Scene;
use phox_core::software_rasterizer::rasterize;
use phox_forward::{ForwardRenderer, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use serial_test::serial;
use std::sync::Arc;

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

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

#[test]
#[serial]
fn a_gltf_scene_draws_like_the_software_rasterizer() {
    let validation = flag("PHOX_VALIDATION");
    let device = match VulkanDevice::new(&VulkanConfig { validation, ..VulkanConfig::default() }) {
        Ok(d) => d,
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            return;
        }
    };
    let path = std::env::temp_dir().join("phox_rs_forward_gltf_test.gltf");
    std::fs::write(&path, TEST_GLTF).unwrap();
    let mut scene = Scene::new();
    let import = load_gltf(&path, &mut scene, &Mat4::IDENTITY).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(import.objects.len(), 3);
    scene.camera.position = Vec3::new(1.0, 0.5, 4.0);
    scene.camera.look_at(Vec3::new(1.0, 0.0, -1.0), Vec3::Y);

    let extent = Extent2D::new(160, 120);
    let mut renderer = ForwardRenderer::new(Arc::new(device.clone()), RendererConfig { extent, ..RendererConfig::default() }).unwrap();
    for _ in 0..3 {
        renderer.render(&scene).unwrap(); // the second and third frames reuse the cached mesh and texture
    }
    let gpu = renderer.last_frame().clone();
    let software = rasterize(&extract_view(&scene, extent, true));

    let background = scene.clear_color;
    let lit = gpu.rgba.chunks_exact(4).filter(|p| p[0] as f32 / 255.0 > background.x * 2.0 + 0.3 || p[1] > 90).count();
    assert!(lit > 200, "the scene should cover part of the frame ({lit} pixels differ from the background)");
    let diff = compare_images(&gpu, &software, 2);
    let allowed = (0.02 * extent.width as f64 * extent.height as f64) as usize;
    assert!(!diff.size_mismatch && diff.pixels_over_tolerance <= allowed, "{diff:?} (allowed {allowed})");
    drop(renderer);
    let (errors, warnings) = device.validation_counts();
    assert_eq!((errors, warnings), (0, 0));
}
