//! The Metal device against the real GPU: resources, copies, fills, rendering, readback, shaders translated
//! from SPIR-V, compute, scissors and misuse reporting.
//!
//! Skips (passes) when there is no Metal 3 device, unless PHOX_REQUIRE_GPU=1.
#![cfg(target_os = "macos")]

use phox_metal::*;
use phox_rhi::*;
use serial_test::serial;
use std::process::Command;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn device() -> Option<MetalDevice> {
    match MetalDevice::new(&MetalConfig { validation: true }) {
        Ok(device) => Some(device),
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Metal device: {e}");
            eprintln!("skipping: no Metal device ({e})");
            None
        }
    }
}

/// Compiles GLSL to SPIR-V with glslangValidator (as the crates' build scripts do), or skips.
fn spirv(stage: &str, source: &str) -> Option<Vec<u32>> {
    let dir = std::env::temp_dir().join(format!("phox-metal-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let input = dir.join(format!("shader.{stage}"));
    let output = dir.join(format!("shader.{stage}.spv"));
    std::fs::write(&input, source).ok()?;
    let status = Command::new("glslangValidator").args(["-V", "-o"]).arg(&output).arg(&input).output().ok()?;
    if !status.status.success() {
        panic!("glslangValidator: {}", String::from_utf8_lossy(&status.stdout));
    }
    let bytes = std::fs::read(&output).ok()?;
    Some(bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect())
}

fn readback(device: &MetalDevice, size: u64) -> BufferHandle {
    device.create_buffer(&BufferDesc::new(size, BufferUsage::TRANSFER_DST, MemoryKind::Readback, "readback")).unwrap()
}

#[test]
#[serial]
fn buffers_copy_fill_and_read_back() {
    let Some(device) = device() else { return };
    let upload = device.create_buffer(&BufferDesc::new(64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "upload")).unwrap();
    let gpu = device.create_buffer(&BufferDesc::new(64, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST, MemoryKind::GpuOnly, "gpu")).unwrap();
    let out = readback(&device, 64);
    let data: Vec<u8> = (0..64).collect();
    device.write_buffer(upload, 0, &data).unwrap();

    let mut cmd = device.begin_commands().unwrap();
    cmd.copy_buffer(upload, 0, gpu, 0, 64);
    cmd.copy_buffer(gpu, 0, out, 0, 64);
    cmd.fill_buffer(out, 0, 8, 0xAABB_CCDD); // a pattern Metal cannot fill with one byte
    cmd.fill_buffer(out, 60, 4, 0x0707_0707); // one it can
    let ticket = device.submit(cmd).unwrap();
    device.wait(ticket).unwrap();
    assert!(device.is_complete(ticket));

    let mut bytes = [0u8; 64];
    device.read_buffer(out, 0, &mut bytes).unwrap();
    assert_eq!(&bytes[..8], &[0xDD, 0xCC, 0xBB, 0xAA, 0xDD, 0xCC, 0xBB, 0xAA]);
    assert_eq!(&bytes[8..60], &data[8..60]);
    assert_eq!(&bytes[60..], &[7, 7, 7, 7]);
    assert!(device.write_buffer(gpu, 0, &[0]).is_err(), "GPU-only memory cannot be mapped");
    assert!(device.write_buffer(upload, 60, &[0; 8]).is_err(), "out of bounds");
}

#[test]
#[serial]
fn rendering_clears_a_target_and_reads_it_back() {
    let Some(device) = device() else { return };
    let target = device
        .create_texture(&TextureDesc::new(8, 4, Format::RGBA8Unorm, TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC, "target"))
        .unwrap();
    let out = readback(&device, 8 * 4 * 4);
    let mut cmd = device.begin_commands().unwrap();
    cmd.begin_debug_label("clear");
    cmd.begin_rendering(&RenderingInfo {
        colors: vec![ColorTarget { texture: target, clear: true, clear_color: [1.0, 0.0, 0.5, 1.0] }],
        depth: None,
        width: 8,
        height: 4,
    });
    cmd.end_rendering();
    cmd.end_debug_label();
    cmd.copy_texture_to_buffer(target, out);
    device.wait(device.submit(cmd).unwrap()).unwrap();

    let mut pixels = [0u8; 8 * 4 * 4];
    device.read_buffer(out, 0, &mut pixels).unwrap();
    assert!(pixels.chunks_exact(4).all(|p| p == [255, 0, 128, 255]), "{:?}", &pixels[..8]);
}

const VERTEX: &str = r#"#version 450
layout(location = 0) in vec2 position;
layout(push_constant) uniform PC { vec4 color; } pc;
layout(location = 0) out vec4 color;
void main() { gl_Position = vec4(position, 0.0, 1.0); color = pc.color; }
"#;

const FRAGMENT: &str = r#"#version 450
layout(location = 0) in vec4 color;
layout(location = 0) out vec4 out_color;
void main() { out_color = color; }
"#;

#[test]
#[serial]
fn a_translated_pipeline_draws_the_right_way_up_inside_the_scissor() {
    let Some(device) = device() else { return };
    let (Some(vs), Some(fs)) = (spirv("vert", VERTEX), spirv("frag", FRAGMENT)) else {
        eprintln!("skipping: glslangValidator is not available");
        return;
    };
    let vs = device.create_shader(&ShaderDesc::new(ShaderStage::Vertex, vs)).unwrap();
    let fs = device.create_shader(&ShaderDesc::new(ShaderStage::Fragment, fs)).unwrap();
    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDesc {
            shaders: vec![vs, fs],
            vertex_layout: VertexLayout { stride: 8, attributes: vec![VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 }] },
            color_formats: vec![Format::RGBA8Unorm],
            push_constants: vec![PushConstantRange { stages: ShaderStages::VERTEX, offset: 0, size: 16 }],
            debug_name: "smoke".into(),
            ..GraphicsPipelineDesc::default()
        })
        .unwrap();
    // A quad over the top half of clip space in Vulkan's convention (y = -1 is the top of the image).
    let quad: [[f32; 2]; 6] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 0.0], [-1.0, -1.0], [1.0, 0.0], [-1.0, 0.0]];
    let bytes: Vec<u8> = quad.iter().flatten().flat_map(|f| f.to_le_bytes()).collect();
    let vertices = device.create_buffer(&BufferDesc::new(bytes.len() as u64, BufferUsage::VERTEX, MemoryKind::Upload, "quad")).unwrap();
    device.write_buffer(vertices, 0, &bytes).unwrap();
    let target = device
        .create_texture(&TextureDesc::new(16, 16, Format::RGBA8Unorm, TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC, "target"))
        .unwrap();
    let out = readback(&device, 16 * 16 * 4);

    let mut cmd = device.begin_commands().unwrap();
    cmd.begin_rendering(&RenderingInfo {
        colors: vec![ColorTarget { texture: target, clear: true, clear_color: [0.0, 0.0, 0.0, 1.0] }],
        depth: None,
        width: 16,
        height: 16,
    });
    cmd.bind_pipeline(pipeline);
    cmd.bind_vertex_buffer(vertices, 0);
    let green: Vec<u8> = [0.0f32, 1.0, 0.0, 1.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    cmd.push_constants(ShaderStages::VERTEX, 0, &green);
    cmd.set_scissor(0, 0, 8, 100); // the left half; the height is clamped to the target
    cmd.draw(6, 1, 0);
    cmd.end_rendering();
    cmd.copy_texture_to_buffer(target, out);
    device.wait(device.submit(cmd).unwrap()).unwrap();

    let mut pixels = vec![0u8; 16 * 16 * 4];
    device.read_buffer(out, 0, &mut pixels).unwrap();
    let at = |x: usize, y: usize| &pixels[(y * 16 + x) * 4..(y * 16 + x) * 4 + 4];
    assert_eq!(at(2, 2), [0, 255, 0, 255], "the top-left is drawn");
    assert_eq!(at(2, 13), [0, 0, 0, 255], "the bottom stays clear: the image is the right way up");
    assert_eq!(at(13, 2), [0, 0, 0, 255], "the right half is outside the scissor");
}

const COMPUTE: &str = r#"#version 450
layout(local_size_x = 8) in;
layout(set = 0, binding = 1, std430) buffer Data { uint values[]; } data;
layout(push_constant) uniform PC { uint add; } pc;
void main() { data.values[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 2u + pc.add; }
"#;

#[test]
#[serial]
fn compute_writes_a_storage_buffer() {
    let Some(device) = device() else { return };
    let Some(cs) = spirv("comp", COMPUTE) else {
        eprintln!("skipping: glslangValidator is not available");
        return;
    };
    let cs = device.create_shader(&ShaderDesc::new(ShaderStage::Compute, cs)).unwrap();
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDesc {
            shader: cs,
            bindings: vec![Binding::new(0, 1, BindingType::StorageBuffer, ShaderStages::COMPUTE)],
            push_constants: vec![PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 4 }],
            debug_name: "double".into(),
        })
        .unwrap();
    let data = device.create_buffer(&BufferDesc::new(64, BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC, MemoryKind::GpuOnly, "data")).unwrap();
    let out = readback(&device, 64);
    let mut cmd = device.begin_commands().unwrap();
    cmd.bind_pipeline(pipeline);
    cmd.bind_storage_buffer(0, 1, data);
    cmd.push_constants(ShaderStages::COMPUTE, 0, &5u32.to_le_bytes());
    cmd.dispatch(2, 1, 1); // 16 invocations of 8 per group
    cmd.copy_buffer(data, 0, out, 0, 64);
    device.wait(device.submit(cmd).unwrap()).unwrap();

    let mut bytes = [0u8; 64];
    device.read_buffer(out, 0, &mut bytes).unwrap();
    let values: Vec<u32> = bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
    assert_eq!(values, (0..16).map(|i| i * 2 + 5).collect::<Vec<_>>());
}

#[test]
#[serial]
fn mismatched_requests_are_errors_not_crashes() {
    let Some(device) = device() else { return };
    assert!(device.create_buffer(&BufferDesc::new(0, BufferUsage::STORAGE, MemoryKind::GpuOnly, "empty")).is_err());
    assert!(device.create_buffer(&BufferDesc::new(8, BufferUsage::NONE, MemoryKind::GpuOnly, "no usage")).is_err());
    assert!(device.create_texture(&TextureDesc::new(0, 4, Format::RGBA8Unorm, TextureUsage::SAMPLED, "t")).is_err());
    let handle = device.create_buffer(&BufferDesc::new(8, BufferUsage::STORAGE, MemoryKind::Upload, "b")).unwrap();
    device.destroy_buffer(handle);
    assert!(matches!(device.describe_buffer(handle), Err(RhiError::StaleHandle { .. })));
    // Unsupported recording is reported at submit, not by crashing.
    let mut cmd = device.begin_commands().unwrap();
    cmd.draw_mesh_tasks(1, 1, 1);
    assert!(device.submit(cmd).is_err());
    let mut cmd = device.begin_commands().unwrap();
    cmd.draw(3, 1, 0); // outside begin_rendering
    assert!(device.submit(cmd).is_err());
    assert!(device.acquire_backbuffer(64, 64).is_err(), "no view, nothing to present to");
    assert!(device.name().len() > 1);
}
