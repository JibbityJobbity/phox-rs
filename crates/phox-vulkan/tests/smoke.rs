//! The Vulkan device against a real (or software) driver: resources, barriers, copies, rendering, readback
//! and timestamps, under the validation layer when PHOX_VALIDATION=1.
//!
//! Skips (passes) when there is no Vulkan device, unless PHOX_REQUIRE_GPU=1.

use phox_rhi::*;
use phox_vulkan::*;
use serial_test::serial;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn device() -> Option<VulkanDevice> {
    let _ = env_logger::Builder::from_env(env_logger::Env::new().filter_or("PHOX_LOG", "info")).try_init();
    let config = VulkanConfig { validation: flag("PHOX_VALIDATION"), ..VulkanConfig::default() };
    match VulkanDevice::new(&config) {
        Ok(device) => {
            if flag("PHOX_VALIDATION") {
                assert!(device.info().validation, "PHOX_VALIDATION is set but the validation layer is not available");
            }
            Some(device)
        }
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            eprintln!("skipping: no Vulkan device ({e})");
            None
        }
    }
}

fn finish(device: &VulkanDevice) {
    let (errors, warnings) = device.validation_counts();
    assert_eq!((errors, warnings), (0, 0), "the validation layer reported messages");
}

#[test]
#[serial]
fn buffers_copy_and_read_back() {
    let Some(device) = device() else { return };
    let upload = device.create_buffer(&BufferDesc::new(64, BufferUsage::TRANSFER_SRC, MemoryKind::Upload, "upload")).unwrap();
    let gpu = device.create_buffer(&BufferDesc::new(64, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST, MemoryKind::GpuOnly, "gpu")).unwrap();
    let readback = device.create_buffer(&BufferDesc::new(64, BufferUsage::TRANSFER_DST, MemoryKind::Readback, "readback")).unwrap();
    let data: Vec<u8> = (0..64).collect();
    device.write_buffer(upload, 0, &data).unwrap();

    let mut cmd = device.begin_commands().unwrap();
    let b = |buffer, from, to| BufferBarrier { buffer, from, to };
    cmd.barrier(&[], &[b(gpu, ResourceState::Undefined, ResourceState::TransferDst), b(readback, ResourceState::Undefined, ResourceState::TransferDst)]);
    cmd.copy_buffer(upload, 0, gpu, 0, 64);
    cmd.barrier(&[], &[b(gpu, ResourceState::TransferDst, ResourceState::TransferSrc)]);
    cmd.copy_buffer(gpu, 0, readback, 0, 64);
    // A second write to the same buffer needs its own barrier (a write-after-write hazard).
    cmd.barrier(&[], &[b(readback, ResourceState::TransferDst, ResourceState::TransferDst)]);
    cmd.fill_buffer(readback, 0, 8, 0xAABB_CCDD); // overwrites the first two words
    cmd.barrier(&[], &[b(readback, ResourceState::TransferDst, ResourceState::HostRead)]);
    let ticket = device.submit(cmd).unwrap();
    device.wait(ticket).unwrap();
    assert!(device.is_complete(ticket));

    let mut out = [0u8; 64];
    device.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(&out[..8], &[0xDD, 0xCC, 0xBB, 0xAA, 0xDD, 0xCC, 0xBB, 0xAA]);
    assert_eq!(&out[8..], &data[8..]);
    assert!(device.write_buffer(gpu, 0, &[0]).is_err()); // GPU-only memory
    device.destroy_buffer(upload);
    device.destroy_buffer(gpu);
    device.destroy_buffer(readback);
    finish(&device);
}

#[test]
#[serial]
fn rendering_clears_a_target_and_reads_it_back_with_timestamps() {
    let Some(device) = device() else { return };
    let target = device
        .create_texture(&TextureDesc::new(8, 4, Format::RGBA8Unorm, TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC, "target"))
        .unwrap();
    let readback = device.create_buffer(&BufferDesc::new(8 * 4 * 4, BufferUsage::TRANSFER_DST, MemoryKind::Readback, "readback")).unwrap();

    let mut cmd = device.begin_commands().unwrap();
    let timestamps = device.capabilities().has(Feature::Timestamps);
    if timestamps {
        cmd.write_timestamp("begin");
    }
    cmd.barrier(&[TextureBarrier { texture: target, from: ResourceState::Undefined, to: ResourceState::ColorAttachment }], &[]);
    cmd.begin_rendering(&RenderingInfo {
        colors: vec![ColorTarget { texture: target, clear: true, clear_color: [1.0, 0.0, 0.5, 1.0] }],
        depth: None,
        width: 8,
        height: 4,
    });
    cmd.end_rendering();
    cmd.barrier(
        &[TextureBarrier { texture: target, from: ResourceState::ColorAttachment, to: ResourceState::TransferSrc }],
        &[BufferBarrier { buffer: readback, from: ResourceState::Undefined, to: ResourceState::TransferDst }],
    );
    cmd.copy_texture_to_buffer(target, readback);
    cmd.barrier(&[], &[BufferBarrier { buffer: readback, from: ResourceState::TransferDst, to: ResourceState::HostRead }]);
    if timestamps {
        cmd.write_timestamp("end");
    }
    let ticket = device.submit(cmd).unwrap();
    device.wait(ticket).unwrap();

    let mut pixels = [0u8; 8 * 4 * 4];
    device.read_buffer(readback, 0, &mut pixels).unwrap();
    assert!(pixels.chunks_exact(4).all(|p| p == [255, 0, 128, 255]), "{:?}", &pixels[..8]);
    if timestamps {
        let stamps = device.read_timestamps(ticket).unwrap();
        assert_eq!(stamps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["begin", "end"]);
        assert!(stamps[1].nanoseconds >= stamps[0].nanoseconds);
    }
    finish(&device);
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
    // Mesh tasks without recording support are reported at submit, not by crashing.
    if !device.capabilities().has(Feature::MeshShader) {
        let mut cmd = device.begin_commands().unwrap();
        cmd.draw_mesh_tasks(1, 1, 1);
        assert!(device.submit(cmd).is_err());
    }
    assert!(device.info().name.len() > 1);
    finish(&device);
}
