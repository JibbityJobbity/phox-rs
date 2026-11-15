//! Renders the sample scenes headless on a real (or software) Vulkan device and compares them with the
//! images in test/golden/ (made on lavapipe by the C++ renderer). Ported from test/source/golden_test.cpp.
//!
//! Environment:
//!   PHOX_UPDATE_GOLDEN=1  write the rendered images to test/golden/ instead of comparing; a
//!                         comma-separated list of case names updates only those
//!   PHOX_VALIDATION=1     run with the validation layer (incl. synchronization) and fail on any message
//!   PHOX_REQUIRE_GPU=1    fail instead of skipping when no Vulkan device is available
//!   PHOX_REQUIRE_MESH_SHADERS=1  fail instead of skipping the mesh shader paths when the device lacks them
//!   PHOX_DEVICE=<name>    render on the device whose name contains <name>, e.g. "llvmpipe"

use phox_core::cpu_image::{CpuImage, compare_images, read_ppm, write_ppm};
use phox_core::resources::Extent2D;
use phox_core::sample_scenes::*;
use phox_core::scene::{ToneMapOperator, ToneMapping};
use phox_forward::{ForwardRenderer, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use serial_test::serial;
use std::path::PathBuf;
use std::sync::Arc;

/// Rasterizers may disagree on pixels an edge passes exactly through, and colour conversion may round
/// differently, so allow a small per-channel delta everywhere and larger ones on a few pixels.
const CHANNEL_TOLERANCE: u32 = 2;
const MAX_FRACTION_OVER_TOLERANCE: f64 = 0.005;
const TEXTURED_FRACTION_OVER_TOLERANCE: f64 = 0.02;
/// Each case renders a few frames so the frames in flight and pooled images are reused (and checked by
/// validation) before the last frame is compared.
const FRAMES_PER_CASE: usize = 3;

struct Case {
    name: &'static str,
    scene: SampleSceneId,
    time: f32,
    tone_mapping: ToneMapping,
    allowed_fraction: f64,
}

fn cases() -> Vec<Case> {
    let plain = ToneMapping::default();
    let case = |name, scene, time, tone_mapping| Case { name, scene, time, tone_mapping, allowed_fraction: MAX_FRACTION_OVER_TOLERANCE };
    // Textured surfaces differ more across implementations: they pick mips (and filter between them) differently.
    let textured = |name, scene, time, tone_mapping| Case { allowed_fraction: TEXTURED_FRACTION_OVER_TOLERANCE, ..case(name, scene, time, tone_mapping) };
    vec![
        case("triangle", SampleSceneId::Triangle, 0.0, plain),
        case("triangle_turned", SampleSceneId::Triangle, 2.0, plain), // past 90 degrees: the back face, lit from behind
        case("cubes", SampleSceneId::Cubes, 0.7, plain),
        case("cubes_aces", SampleSceneId::Cubes, 0.7, ToneMapping { op: ToneMapOperator::Aces, exposure: 1.6 }),
        case("tori", SampleSceneId::Tori, 0.8, plain), // meshlets cut by the frustum and facing away
        // Metal/roughness, textures, normal map, alpha mask and blend, emission.
        textured("materials", SampleSceneId::Materials, 0.7, plain),
    ]
}

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn update_requested(name: &str) -> bool {
    match std::env::var("PHOX_UPDATE_GOLDEN") {
        Ok(v) if v.is_empty() || v == "0" => false,
        Ok(v) if v == "1" => true,
        Ok(v) => v.split(',').any(|n| n == name),
        Err(_) => false,
    }
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/golden")
}

fn device() -> Option<VulkanDevice> {
    let _ = env_logger::Builder::from_env(env_logger::Env::new().filter_or("PHOX_LOG", "info")).try_init();
    let validation = flag("PHOX_VALIDATION");
    match VulkanDevice::new(&VulkanConfig { validation, ..VulkanConfig::default() }) {
        Ok(device) => {
            assert!(!validation || device.info().validation, "PHOX_VALIDATION is set but the validation layer is not available");
            Some(device)
        }
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            eprintln!("skipping: no Vulkan device ({e})");
            None
        }
    }
}

fn device_without_indirect_count() -> Option<VulkanDevice> {
    VulkanDevice::new(&VulkanConfig { validation: flag("PHOX_VALIDATION"), draw_indirect_count: false, ..VulkanConfig::default() }).ok()
}

fn render(renderer: &mut ForwardRenderer, case: &Case) -> CpuImage {
    let mut sample = make_sample_scene(case.scene).unwrap();
    (sample.animate)(&mut sample.scene, case.time);
    sample.scene.tone_mapping = case.tone_mapping;
    for _ in 0..FRAMES_PER_CASE {
        renderer.render(&sample.scene).unwrap();
    }
    renderer.last_frame().clone()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Vertex,
    /// Vertex shaders on a device without drawIndirectCount.
    VertexNoCount,
    MeshOcclusion,
    Mesh,
    /// Mesh shaders, occlusion culling and cluster LOD, against `<case>.lod.ppm`.
    MeshLod,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Path::Vertex => "vertex shaders",
            Path::VertexNoCount => "vertex shaders, no draw count",
            Path::MeshOcclusion => "mesh shaders, occlusion culling",
            Path::Mesh => "mesh shaders",
            Path::MeshLod => "mesh shaders, occlusion culling, cluster LOD",
        }
    }

    fn config(self) -> RendererConfig {
        RendererConfig {
            extent: Extent2D::new(320, 240),
            mesh_shaders: !matches!(self, Path::Vertex | Path::VertexNoCount),
            occlusion_culling: matches!(self, Path::MeshOcclusion | Path::MeshLod),
            mesh_lod: self == Path::MeshLod,
            ..RendererConfig::default()
        }
    }
}

/// Renders every case with one renderer configuration and compares with the goldens (which every path must
/// match: culling never changes the picture). Only the vertex path rewrites them. Returns the failed cases,
/// or `None` when the path was skipped.
fn run_path(vulkan: &VulkanDevice, path: Path) -> Option<Vec<String>> {
    if matches!(path, Path::MeshOcclusion | Path::Mesh | Path::MeshLod) && !vulkan.info().mesh_shaders {
        assert!(!flag("PHOX_REQUIRE_MESH_SHADERS"), "{} has no mesh shaders", vulkan.info().name);
        eprintln!("{} has no mesh shaders; skipping the mesh shader paths", vulkan.info().name);
        return None;
    }
    let mut renderer = ForwardRenderer::new(Arc::new(vulkan.clone()), path.config()).unwrap();
    assert_eq!(renderer.uses_mesh_shaders(), matches!(path, Path::MeshOcclusion | Path::Mesh | Path::MeshLod));
    assert_eq!(renderer.uses_occlusion_culling(), matches!(path, Path::MeshOcclusion | Path::MeshLod));
    assert_eq!(renderer.uses_mesh_lod(), path == Path::MeshLod);

    let mut failures = Vec::new();
    for case in cases() {
        let actual = render(&mut renderer, &case);
        let lod = path == Path::MeshLod;
        let golden = golden_dir().join(format!("{}{}.ppm", case.name, if lod { ".lod" } else { "" }));
        if (path == Path::Vertex || lod) && update_requested(case.name) {
            write_ppm(&golden, &actual).unwrap();
            println!("updated {}", golden.display());
            continue;
        }
        let expected = read_ppm(&golden).unwrap();
        let diff = compare_images(&actual, &expected, CHANNEL_TOLERANCE);
        let allowed = (case.allowed_fraction * expected.width as f64 * expected.height as f64) as usize;
        let pass = !diff.size_mismatch && diff.pixels_over_tolerance <= allowed;
        println!(
            "{} {} ({}): max channel delta {}, {} pixel(s) over tolerance (allowed {allowed}){}",
            if pass { "PASS" } else { "FAIL" },
            case.name,
            path.name(),
            diff.max_channel_delta,
            diff.pixels_over_tolerance,
            if diff.size_mismatch { ", size mismatch" } else { "" }
        );
        if !pass {
            let suffix = match path {
                Path::Vertex => "vulkan",
                Path::VertexNoCount => "nocount",
                Path::MeshOcclusion => "occlusion",
                Path::Mesh => "mesh",
                Path::MeshLod => "lod",
            };
            let out = PathBuf::from("golden_out").join(format!("{}.{suffix}.ppm", case.name));
            write_ppm(&out, &actual).unwrap();
            println!("  actual image written to {}", std::fs::canonicalize(&out).unwrap_or(out).display());
            failures.push(format!("{} ({})", case.name, path.name()));
        }
    }
    Some(failures)
}

#[test]
#[serial]
fn every_path_matches_the_goldens() {
    let Some(vulkan) = device() else { return };
    println!("Vulkan device: {}", vulkan.info().name);
    let mut failures = Vec::new();
    for path in [Path::Vertex, Path::MeshOcclusion, Path::Mesh, Path::MeshLod] {
        failures.extend(run_path(&vulkan, path).unwrap_or_default());
    }
    // PHOX_NO_DRAW_INDIRECT_COUNT=1 tests the path without it; so does asking the device not to use it.
    if let Some(no_count) = device_without_indirect_count() {
        failures.extend(run_path(&no_count, Path::VertexNoCount).unwrap_or_default());
        assert_eq!(no_count.validation_counts(), (0, 0));
    }
    let (errors, warnings) = vulkan.validation_counts();
    assert!(errors == 0 && warnings == 0, "validation reported {errors} error(s) and {warnings} warning(s)");
    assert!(failures.is_empty(), "differs from the goldens: {failures:?}");
}
