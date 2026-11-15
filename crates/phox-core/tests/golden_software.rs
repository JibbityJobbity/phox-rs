//! The software rasterizer against the golden images rendered on lavapipe (test/golden/*.ppm), as the C++
//! golden test does after its vertex-shader run. The tolerance is the C++ one: a per-channel difference of 2
//! on all but 0.5% of the pixels (2% for the textured case).

use phox_core::cpu_image::{compare_images, read_ppm, write_ppm};
use phox_core::render_view::extract_view;
use phox_core::resources::Extent2D;
use phox_core::sample_scenes::*;
use phox_core::scene::{ToneMapOperator, ToneMapping};
use phox_core::software_rasterizer::rasterize;
use std::path::PathBuf;

const CHANNEL_TOLERANCE: u32 = 2;
const MAX_FRACTION_OVER_TOLERANCE: f64 = 0.005;
/// Textured surfaces differ more: implementations pick mips (and filter between them) differently on
/// minified and slanted surfaces, and the software rasterizer only samples the first mip.
const TEXTURED_FRACTION_OVER_TOLERANCE: f64 = 0.02;

struct Case {
    name: &'static str,
    scene: SampleSceneId,
    time: f32,
    tone_mapping: ToneMapping,
    allowed_fraction: f64,
}

fn cases() -> Vec<Case> {
    let plain = ToneMapping::default();
    let case = |name, scene, time, tone_mapping, allowed_fraction| Case { name, scene, time, tone_mapping, allowed_fraction };
    vec![
        case("triangle", SampleSceneId::Triangle, 0.0, plain, MAX_FRACTION_OVER_TOLERANCE),
        // Past 90 degrees: the back face, lit from behind.
        case("triangle_turned", SampleSceneId::Triangle, 2.0, plain, MAX_FRACTION_OVER_TOLERANCE),
        case("cubes", SampleSceneId::Cubes, 0.7, plain, MAX_FRACTION_OVER_TOLERANCE),
        case("cubes_aces", SampleSceneId::Cubes, 0.7, ToneMapping { op: ToneMapOperator::Aces, exposure: 1.6 }, MAX_FRACTION_OVER_TOLERANCE),
        // Meshlets cut by the frustum and facing away.
        case("tori", SampleSceneId::Tori, 0.8, plain, MAX_FRACTION_OVER_TOLERANCE),
        // Metal/roughness, textures, normal map, alpha mask and blend, emission.
        case("materials", SampleSceneId::Materials, 0.7, plain, TEXTURED_FRACTION_OVER_TOLERANCE),
    ]
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/golden")
}

#[test]
fn software_rasterizer_matches_the_goldens() {
    let extent = Extent2D::new(320, 240);
    let mut failures = Vec::new();
    for case in cases() {
        let mut sample = make_sample_scene(case.scene).unwrap();
        (sample.animate)(&mut sample.scene, case.time);
        sample.scene.tone_mapping = case.tone_mapping;
        let software = rasterize(&extract_view(&sample.scene, extent, true));
        let expected = read_ppm(&golden_dir().join(format!("{}.ppm", case.name))).unwrap();
        let diff = compare_images(&software, &expected, CHANNEL_TOLERANCE);
        let allowed = (case.allowed_fraction * expected.width as f64 * expected.height as f64) as usize;
        let pass = !diff.size_mismatch && diff.pixels_over_tolerance <= allowed;
        println!(
            "{} {} (software): max channel delta {}, {} pixel(s) over tolerance (allowed {allowed})",
            if pass { "PASS" } else { "FAIL" },
            case.name,
            diff.max_channel_delta,
            diff.pixels_over_tolerance
        );
        if !pass {
            let out = PathBuf::from("golden_out").join(format!("{}.software.ppm", case.name));
            write_ppm(&out, &software).unwrap();
            failures.push(case.name);
        }
    }
    assert!(failures.is_empty(), "software rasterizer differs from the goldens: {failures:?} (images in golden_out/)");
}
