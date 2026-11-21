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
    let technique = ForwardTechnique::new(device.clone(), ForwardOptions::default());
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
        RendererConfig { extent: Extent2D::new(64, 48), capture_graph: true, timestamps: false, mesh_shaders: false, ..RendererConfig::default() },
    )
    .unwrap();
    let sample = make_sample_scene(SampleSceneId::Cubes).unwrap();
    renderer.render(&sample.scene).unwrap();

    let labels: Vec<String> = device.last_recording().into_iter().filter_map(|c| c.strip_prefix("beginDebugLabel:").map(str::to_string)).collect();
    assert_eq!(labels, ["scene update", "clear draw counters", "cull", "forward", "tonemap", "readback"]);
    let recording = device.last_recording();
    assert!(recording.contains(&"drawIndexedIndirect".to_string()));
    assert!(recording.contains(&"copyTextureToBuffer".to_string()));
    assert_eq!(recording.iter().filter(|c| c.as_str() == "draw").count(), 1); // the tonemap triangle

    let dot = renderer.frame_graph_dot();
    for needle in ["scene update", "cull", "forward", "tonemap", "readback", "hdr", "depth", "display"] {
        assert!(dot.contains(needle), "missing '{needle}' in:\n{dot}");
    }
    assert_eq!(renderer.last_frame().width, 64);
}

#[test]
fn the_mesh_path_records_two_phases_around_a_depth_pyramid() {
    let device = Arc::new(NullDevice::new());
    device.set_features([Feature::BindlessDescriptors, Feature::MeshShader]);
    let mut renderer = ForwardRenderer::new(device.clone(), RendererConfig { extent: Extent2D::new(64, 48), timestamps: false, ..RendererConfig::default() }).unwrap();
    assert!(renderer.uses_mesh_shaders() && renderer.uses_occlusion_culling());
    let sample = make_sample_scene(SampleSceneId::Cubes).unwrap();
    renderer.render(&sample.scene).unwrap(); // the first frame: nothing was visible before, so no early work
    renderer.render(&sample.scene).unwrap();

    let recording = device.last_recording();
    let labels: Vec<&str> = recording.iter().filter_map(|c| c.strip_prefix("beginDebugLabel:")).collect();
    // Nothing changed since the first frame, so there is no scene update.
    assert_eq!(
        labels,
        ["clear draw counters", "cull early", "draw args early", "forward early", "depth pyramid", "cull late", "draw args late", "forward late", "tonemap", "readback"]
    );
    assert!(recording.contains(&"drawMeshTasksIndirect".to_string()));
    assert!(recording.contains(&"dispatch".to_string())); // the pyramid's mips
    assert!(!recording.contains(&"drawIndexedIndirect".to_string())); // nothing opaque on the vertex path
}

#[test]
fn without_occlusion_culling_one_mesh_pass_draws() {
    let device = Arc::new(NullDevice::new());
    device.set_features([Feature::BindlessDescriptors, Feature::MeshShader]);
    let config = RendererConfig { extent: Extent2D::new(64, 48), timestamps: false, occlusion_culling: false, ..RendererConfig::default() };
    let mut renderer = ForwardRenderer::new(device.clone(), config).unwrap();
    assert!(renderer.uses_mesh_shaders() && !renderer.uses_occlusion_culling());
    renderer.render(&make_sample_scene(SampleSceneId::Cubes).unwrap().scene).unwrap();
    let labels: Vec<String> = device.last_recording().into_iter().filter_map(|c| c.strip_prefix("beginDebugLabel:").map(str::to_string)).collect();
    assert_eq!(labels, ["scene update", "clear draw counters", "cull", "draw args", "forward", "tonemap", "readback"]);
}

#[test]
fn an_empty_scene_still_clears_and_tonemaps() {
    let device = Arc::new(NullDevice::new());
    device.set_features([Feature::BindlessDescriptors]);
    let mut renderer = ForwardRenderer::new(device.clone(), RendererConfig { extent: Extent2D::new(16, 16), ..RendererConfig::default() }).unwrap();
    renderer.render(&phox_core::scene::Scene::new()).unwrap();
    // The GPU decides how much to draw, so the indirect draw is recorded even for nothing.
    assert!(device.last_recording().contains(&"beginDebugLabel:cull".to_string()));
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

/// Hands the forward pass one generated draw from buffers made up front.
struct Generated {
    draws: BufferHandle,
    vertices: BufferHandle,
    indices: BufferHandle,
    commands: BufferHandle,
}

impl FrameExtension for Generated {
    fn build(&mut self, b: &mut GraphBuilder<'_>) -> GraphResult<GeneratedGeometry> {
        let mut import = |h| b.import_buffer(h, ResourceState::Undefined, ResourceState::Undefined);
        let draw = GeneratedDraw {
            draws: import(self.draws)?,
            vertices: import(self.vertices)?,
            indices: import(self.indices)?,
            commands: import(self.commands)?,
            max_commands: 1,
            count_offset: 20,
            cull_back_faces: false,
            blend: false,
        };
        Ok(GeneratedGeometry { draws: vec![draw], tasks: Vec::new() })
    }
}

/// Hands the forward pass geometry that task and mesh shaders make.
struct Tasks {
    pipeline: GeneratedTaskPipeline,
    draws: BufferHandle,
    field: BufferHandle,
}

impl FrameExtension for Tasks {
    fn build(&mut self, b: &mut GraphBuilder<'_>) -> GraphResult<GeneratedGeometry> {
        let mut import = |h| b.import_buffer(h, ResourceState::Undefined, ResourceState::Undefined);
        let tasks = GeneratedTasks { pipeline: self.pipeline.pipeline, draws: import(self.draws)?, reads: vec![import(self.field)?], groups: [3, 2, 1] };
        Ok(GeneratedGeometry { draws: Vec::new(), tasks: vec![tasks] })
    }
}

#[test]
fn generated_tasks_are_drawn_by_their_own_pipeline_after_the_opaque_objects() {
    let device = Arc::new(NullDevice::new());
    device.set_features([Feature::BindlessDescriptors, Feature::MeshShader]);
    let config = RendererConfig { extent: Extent2D::new(64, 48), timestamps: false, mesh_shaders: true, occlusion_culling: false, ..RendererConfig::default() };
    let mut renderer = ForwardRenderer::new(device.clone(), config).unwrap();
    let buffer = |name: &str| device.create_buffer(&BufferDesc { size: 256, usage: BufferUsage::STORAGE, memory: MemoryKind::GpuOnly, debug_name: name.into(), device_address: true }).unwrap();
    // Any SPIR-V will do for the null device: Phox's own task and mesh shaders.
    let (task, mesh) = (phox_forward::shaders::spirv("forward.task").unwrap(), phox_forward::shaders::spirv("forward.mesh").unwrap());
    let pipeline = GeneratedTaskPipeline::new(device.as_ref(), &task, &mesh, true, true, "tasks").unwrap();
    let mut tasks = Tasks { pipeline, draws: buffer("draws"), field: buffer("field") };
    renderer.render_with(&make_sample_scene(SampleSceneId::Cubes).unwrap().scene, &mut tasks).unwrap();

    let recording = device.last_recording();
    let start = recording.iter().position(|c| c == "beginDebugLabel:forward generated").expect("a 'forward generated' pass");
    let pass = &recording[start..];
    let end = pass.iter().position(|c| c.starts_with("endDebugLabel")).unwrap_or(pass.len());
    assert!(pass[..end].iter().any(|c| c.starts_with("drawMeshTasks")), "{:?}", &pass[..end]);
    pipeline.destroy(device.as_ref());
}

#[test]
fn generated_draws_follow_the_opaque_objects() {
    for mesh_shaders in [false, true] {
        let device = Arc::new(NullDevice::new());
        device.set_features([Feature::BindlessDescriptors, Feature::MeshShader]);
        let config = RendererConfig { extent: Extent2D::new(64, 48), timestamps: false, mesh_shaders, occlusion_culling: false, ..RendererConfig::default() };
        let mut renderer = ForwardRenderer::new(device.clone(), config).unwrap();
        let buffer = |name: &str| device.create_buffer(&BufferDesc { size: 256, usage: BufferUsage::STORAGE, memory: MemoryKind::GpuOnly, debug_name: name.into(), device_address: true }).unwrap();
        let mut generated = Generated { draws: buffer("draws"), vertices: buffer("vertices"), indices: buffer("indices"), commands: buffer("commands") };
        renderer.render_with(&make_sample_scene(SampleSceneId::Cubes).unwrap().scene, &mut generated).unwrap();

        let recording = device.last_recording();
        let labels: Vec<&str> = recording.iter().filter_map(|c| c.strip_prefix("beginDebugLabel:")).collect();
        let at = |label: &str| labels.iter().position(|l| *l == label).unwrap_or_else(|| panic!("no '{label}' in {labels:?}"));
        assert_eq!(at("forward generated"), at("forward") + 1, "{labels:?}");
        assert!(at("forward generated") < at("tonemap"));
        // The generated triangles go through the vertex pipeline, whichever path the scene's objects take.
        let generated_pass = &recording[recording.iter().position(|c| c == "beginDebugLabel:forward generated").unwrap()..];
        assert!(generated_pass.contains(&"drawIndexedIndirect".to_string()));
    }
}
