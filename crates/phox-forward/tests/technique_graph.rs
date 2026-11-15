//! The forward technique's graph, checked on the null device: which features it needs, which passes it
//! records and in what order. No GPU needed.

use phox_core::graph::*;
use phox_core::resources::Extent2D;
use phox_core::sample_scenes::*;
use phox_forward::*;
use phox_null::NullDevice;
use phox_rhi::*;
use std::sync::Arc;

#[test]
fn needs_the_texture_table() {
    let device = Arc::new(NullDevice::new()); // no features
    let technique = ForwardTechnique::new(device.clone());
    assert!(!technique.is_supported(&device.capabilities()));
    device.set_features([Feature::BindlessDescriptors]);
    assert!(technique.is_supported(&device.capabilities()));
}

#[test]
fn a_device_without_the_feature_is_refused_with_the_feature_named() {
    let device = Arc::new(NullDevice::new());
    let mut renderer = ForwardRenderer::new(device, RendererConfig { extent: Extent2D::new(64, 48), ..RendererConfig::default() }).unwrap();
    let sample = make_sample_scene(SampleSceneId::Cubes).unwrap();
    match renderer.render(&sample.scene) {
        Err(ForwardError::Graph(error)) => assert_eq!(error.missing_features(), [Feature::BindlessDescriptors]),
        other => panic!("expected the missing feature, got {other:?}"),
    }
}

#[test]
fn records_forward_tonemap_readback() {
    let device = Arc::new(NullDevice::new());
    device.set_features([Feature::BindlessDescriptors]);
    let mut renderer = ForwardRenderer::new(
        device.clone(),
        RendererConfig { extent: Extent2D::new(64, 48), capture_graph: true, timestamps: false },
    )
    .unwrap();
    let sample = make_sample_scene(SampleSceneId::Cubes).unwrap();
    renderer.render(&sample.scene).unwrap();

    let labels: Vec<String> = device.last_recording().into_iter().filter_map(|c| c.strip_prefix("beginDebugLabel:").map(str::to_string)).collect();
    assert_eq!(labels, ["forward", "tonemap", "readback"]);
    let recording = device.last_recording();
    assert!(recording.contains(&"drawIndexedIndirect".to_string()));
    assert!(recording.contains(&"copyTextureToBuffer".to_string()));
    assert_eq!(recording.iter().filter(|c| c.as_str() == "draw").count(), 1); // the tonemap triangle

    let dot = renderer.frame_graph_dot();
    for needle in ["forward", "tonemap", "readback", "hdr", "depth", "display"] {
        assert!(dot.contains(needle), "missing '{needle}' in:\n{dot}");
    }
    assert_eq!(renderer.last_frame().width, 64);
}

#[test]
fn an_empty_scene_still_clears_and_tonemaps() {
    let device = Arc::new(NullDevice::new());
    device.set_features([Feature::BindlessDescriptors]);
    let mut renderer = ForwardRenderer::new(device.clone(), RendererConfig { extent: Extent2D::new(16, 16), ..RendererConfig::default() }).unwrap();
    renderer.render(&phox_core::scene::Scene::new()).unwrap();
    assert!(!device.last_recording().contains(&"drawIndexedIndirect".to_string()));
    assert!(device.last_recording().contains(&"beginDebugLabel:tonemap".to_string()));
}

#[test]
fn every_shader_is_embedded() {
    let names: Vec<&str> = phox_forward::shaders::names().collect();
    for needed in ["forward.vert", "forward.frag", "fullscreen.vert", "tonemap.frag", "forward.mesh", "forward.task", "cull.comp", "depthpyramid.comp"] {
        assert!(names.contains(&needed), "{needed} missing from {names:?}");
        let spirv = phox_forward::shaders::spirv(needed).unwrap();
        assert_eq!(spirv[0], 0x0723_0203, "{needed} is not SPIR-V"); // the magic number
    }
}
