//! The null device and the RHI's handle pools. Ported from test/source/rhi_null_test.cpp.

use phox_null::NullDevice;
use phox_rhi::*;

fn upload_buffer(size: u64) -> BufferDesc {
    BufferDesc::new(size, BufferUsage::VERTEX | BufferUsage::TRANSFER_DST, MemoryKind::Upload, "test")
}

fn color_texture(w: u32, h: u32) -> TextureDesc {
    TextureDesc::new(w, h, Format::RGBA8Unorm, TextureUsage::COLOR_ATTACHMENT, "target")
}

#[test]
fn slot_reuse_bumps_the_generation() {
    let device = NullDevice::new();
    let first = device.create_buffer(&upload_buffer(64)).unwrap();
    assert!(first.is_valid());
    device.destroy_buffer(first);
    let second = device.create_buffer(&upload_buffer(64)).unwrap();
    assert_eq!(second.index, first.index);
    assert_eq!(second.generation, first.generation + 1);
    assert_ne!(first, second);
    // A second live buffer gets a different slot.
    let third = device.create_buffer(&upload_buffer(16)).unwrap();
    assert_ne!(third.index, second.index);
}

#[test]
fn invalid_destroy_is_a_no_op_and_stale_use_is_an_error() {
    let device = NullDevice::new();
    device.destroy_buffer(BufferHandle::INVALID);
    device.destroy_texture(TextureHandle::INVALID);
    let handle = device.create_buffer(&upload_buffer(8)).unwrap();
    device.destroy_buffer(handle);
    device.destroy_buffer(handle); // a second destroy is a no-op too
    assert!(matches!(device.describe_buffer(handle), Err(RhiError::StaleHandle { .. })));
    assert!(matches!(device.write_buffer(handle, 0, &[1]), Err(RhiError::StaleHandle { .. })));
}

#[test]
fn features() {
    let device = NullDevice::new();
    assert_eq!(device.capabilities().device_name, "Null");
    for feature in [Feature::MeshShader, Feature::RayTracing] {
        assert!(!device.capabilities().has(feature));
    }
    device.set_features([Feature::MeshShader]);
    assert!(device.capabilities().has(Feature::MeshShader));
    assert!(!device.capabilities().has(Feature::RayTracing));
    assert_eq!(Feature::MeshShader.name(), "MeshShader");
}

#[test]
fn describe_and_host_access() {
    let device = NullDevice::new();
    let buffer = device.create_buffer(&upload_buffer(32)).unwrap();
    let desc = device.describe_buffer(buffer).unwrap();
    assert_eq!((desc.size, desc.debug_name.as_str()), (32, "test"));
    let mut bytes = [1u8; 32];
    device.read_buffer(buffer, 0, &mut bytes).unwrap();
    assert!(bytes.iter().all(|&b| b == 0)); // zero-filled
    device.write_buffer(buffer, 4, &[9, 9]).unwrap();
    let mut two = [0u8; 2];
    device.read_buffer(buffer, 4, &mut two).unwrap();
    assert_eq!(two, [9, 9]);
    assert!(device.write_buffer(buffer, 31, &[0, 0]).is_err()); // out of bounds

    let gpu_only = device.create_buffer(&BufferDesc::new(8, BufferUsage::STORAGE, MemoryKind::GpuOnly, "g")).unwrap();
    assert!(matches!(device.write_buffer(gpu_only, 0, &[0]), Err(RhiError::Misuse(_))));

    let texture = device.create_texture(&color_texture(640, 480)).unwrap();
    let t = device.describe_texture(texture).unwrap();
    assert_eq!((t.width, t.height), (640, 480));

    assert!(device.create_buffer(&upload_buffer(0)).unwrap_err().to_string().contains("size"));
    assert!(device.create_texture(&color_texture(0, 4)).unwrap_err().to_string().contains("width"));
}

#[test]
fn recording_and_tickets() {
    let device = NullDevice::new();
    assert!(device.last_recording().is_empty());
    let mut commands = device.begin_commands().unwrap();
    commands.begin_debug_label("pass");
    commands.draw(3, 1, 0);
    commands.end_debug_label();
    let ticket = device.submit(commands).unwrap();
    device.wait(ticket).unwrap();
    assert_eq!(device.last_recording(), ["beginDebugLabel:pass", "draw", "endDebugLabel"]);
    assert_eq!(ticket.value, 1);
    assert!(device.is_complete(ticket));
    let next = device.submit(device.begin_commands().unwrap()).unwrap();
    assert_eq!(next.value, 2);
    assert!(!device.is_complete(SubmitTicket { value: 3 }));
}

#[test]
fn features_gate_commands_and_the_error_comes_at_submit() {
    let device = NullDevice::new();
    let mut commands = device.begin_commands().unwrap();
    commands.draw_mesh_tasks(1, 1, 1);
    let error = device.submit(commands).unwrap_err();
    assert!(error.to_string().contains("MeshShader"), "{error}");

    device.set_features([Feature::MeshShader, Feature::DrawIndirectCount]);
    let mut commands = device.begin_commands().unwrap();
    commands.draw_mesh_tasks(1, 1, 1);
    device.submit(commands).unwrap();
    assert_eq!(device.last_recording(), ["drawMeshTasks"]);
}

#[test]
fn pipelines() {
    let device = NullDevice::new();
    let code = vec![0x0723_0203, 0, 0, 0];
    let shader = |stage| device.create_shader(&ShaderDesc::new(stage, code.clone())).unwrap();
    let (vs, fs, cs) = (shader(ShaderStage::Vertex), shader(ShaderStage::Fragment), shader(ShaderStage::Compute));

    let mut desc = GraphicsPipelineDesc { shaders: vec![vs, fs], ..Default::default() };
    assert!(device.create_graphics_pipeline(&desc).unwrap().is_valid());
    desc.shaders = vec![vs];
    assert!(device.create_graphics_pipeline(&desc).unwrap_err().to_string().contains("Fragment"));
    desc.shaders = vec![vs, vs, fs];
    assert!(device.create_graphics_pipeline(&desc).unwrap_err().to_string().contains("duplicate"));

    let compute = |shader| ComputePipelineDesc { shader, bindings: vec![], push_constants: vec![], debug_name: String::new() };
    assert!(device.create_compute_pipeline(&compute(cs)).unwrap().is_valid());
    assert!(device.create_compute_pipeline(&compute(vs)).unwrap_err().to_string().contains("Compute"));
    assert!(device.create_shader(&ShaderDesc::new(ShaderStage::Vertex, vec![])).unwrap_err().to_string().contains("spirv"));
}

#[test]
fn the_bindless_table_needs_its_feature_and_reuses_slots() {
    let device = NullDevice::new();
    let texture = device.create_texture(&TextureDesc::new(4, 4, Format::RGBA8Unorm, TextureUsage::SAMPLED, "t")).unwrap();
    assert!(matches!(device.register_bindless_texture(texture, &SamplerDesc::default()), Err(RhiError::Unsupported(_))));
    device.set_features([Feature::BindlessDescriptors]);
    let a = device.register_bindless_texture(texture, &SamplerDesc::default()).unwrap();
    let b = device.register_bindless_texture(texture, &SamplerDesc::default()).unwrap();
    assert_ne!(a, b);
    device.unregister_bindless_texture(a);
    assert_eq!(device.register_bindless_texture(texture, &SamplerDesc::default()).unwrap(), a);
}
