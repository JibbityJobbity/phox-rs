//! A mesh handed to the forward pass as a generated draw (`FrameExtension`) looks exactly like the same mesh
//! drawn as a scene object: same pipeline, same shading, the same object's material.
//!
//! Environment: PHOX_VALIDATION=1 (fail on any validation message), PHOX_REQUIRE_GPU=1 (fail instead of skipping
//! without a Vulkan device).

use phox_core::cpu_image::{CpuImage, compare_images, write_ppm};
use phox_core::graph::{GraphBuilder, GraphResult};
use phox_core::math::{Mat3, Vec4};
use phox_core::resources::Extent2D;
use phox_core::sample_scenes::*;
use phox_core::scene::Scene;
use phox_forward::*;
use phox_rhi::*;
use phox_vulkan::{VulkanConfig, VulkanDevice};
use serial_test::serial;
use std::sync::Arc;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn device() -> Option<VulkanDevice> {
    match VulkanDevice::new(&VulkanConfig { validation: flag("PHOX_VALIDATION"), ..VulkanConfig::default() }) {
        Ok(device) => Some(device),
        Err(e) => {
            assert!(!flag("PHOX_REQUIRE_GPU"), "no Vulkan device: {e}");
            eprintln!("skipping: no Vulkan device ({e})");
            None
        }
    }
}

/// A host-visible buffer holding `bytes`, which shaders may read through its address.
fn upload(device: &dyn Device, bytes: &[u8], usage: BufferUsage, name: &str) -> BufferHandle {
    let desc = BufferDesc { size: bytes.len() as u64, usage, memory: MemoryKind::Upload, debug_name: name.into(), device_address: usage != BufferUsage::INDIRECT && usage != BufferUsage::INDEX };
    let buffer = device.create_buffer(&desc).unwrap();
    device.write_buffer(buffer, 0, bytes).unwrap();
    buffer
}

/// A device-local buffer holding `bytes`, copied in through a staging buffer and left readable as `final_state`.
fn upload_device_local(device: &dyn Device, bytes: &[u8], usage: BufferUsage, final_state: ResourceState, name: &str) -> BufferHandle {
    let desc = BufferDesc { size: bytes.len() as u64, usage: usage | BufferUsage::TRANSFER_DST, memory: MemoryKind::GpuOnly, debug_name: name.into(), device_address: false };
    let buffer = device.create_buffer(&desc).unwrap();
    let staging = upload(device, bytes, BufferUsage::TRANSFER_SRC, "staging");
    let mut cmd = device.begin_commands().unwrap();
    cmd.barrier(&[], &[BufferBarrier { buffer, from: ResourceState::Undefined, to: ResourceState::TransferDst }]);
    cmd.copy_buffer(staging, 0, buffer, 0, bytes.len() as u64);
    cmd.barrier(&[], &[BufferBarrier { buffer, from: ResourceState::TransferDst, to: final_state }]);
    let ticket = device.submit(cmd).unwrap();
    device.wait(ticket).unwrap();
    device.destroy_buffer(staging);
    buffer
}

struct Uploaded {
    draws: BufferHandle,
    vertices: BufferHandle,
    indices: BufferHandle,
    commands: BufferHandle,
    cull_back_faces: bool,
}

impl FrameExtension for Uploaded {
    fn build(&mut self, b: &mut GraphBuilder<'_>) -> GraphResult<GeneratedGeometry> {
        let mut import = |h| b.import_buffer(h, ResourceState::Undefined, ResourceState::Undefined);
        let draw = GeneratedDraw {
            draws: import(self.draws)?,
            vertices: import(self.vertices)?,
            indices: import(self.indices)?,
            commands: import(self.commands)?,
            max_commands: 1,
            count_offset: 20,
            cull_back_faces: self.cull_back_faces,
            blend: false,
        };
        Ok(GeneratedGeometry { draws: vec![draw], tasks: Vec::new() })
    }
}

fn render(renderer: &mut ForwardRenderer, scene: &Scene, extension: Option<&mut Uploaded>) -> CpuImage {
    match extension {
        Some(extension) => {
            for _ in 0..3 {
                renderer.render_with(scene, extension).unwrap();
            }
        }
        None => {
            for _ in 0..3 {
                renderer.render(scene).unwrap();
            }
        }
    }
    renderer.last_frame().clone()
}

fn generated_like_scene_objects(vulkan: &VulkanDevice, scene_id: SampleSceneId, time: f32, mesh_shaders: bool) {
    let device: Arc<dyn Device> = Arc::new(vulkan.clone());
    let config = RendererConfig { extent: Extent2D::new(160, 120), timestamps: false, mesh_shaders, mesh_lod: false, ..RendererConfig::default() };
    let mut sample = make_sample_scene(scene_id).unwrap();
    (sample.animate)(&mut sample.scene, time);
    let expected = render(&mut ForwardRenderer::new(device.clone(), config.clone()).unwrap(), &sample.scene, None);

    // Take the first object's mesh away (it keeps its material slot) and hand its triangles over instead.
    let id = sample.scene.object_ids()[0];
    let object = sample.scene.at_mut(id).unwrap();
    let mesh = object.mesh().cloned().unwrap();
    let model = object.transform().matrix();
    let cull_back_faces = !object.material().double_sided && Mat3::from_mat4(model).determinant() > 0.0;
    object.set_mesh(None);
    let vertices = upload(device.as_ref(), bytemuck::cast_slice(mesh.vertices()), BufferUsage::STORAGE, "generated vertices");
    let draw = GpuDraw {
        model,
        camera_position: Vec4::ZERO,
        vertices: split_address(device.buffer_address(vertices).unwrap()),
        primitives: [0, 0],
        count: mesh.indices().len() as u32,
        material: id.index,
        visibility_offset: 0,
        flags: 0,
    };
    let command = DrawCommand { index_count: mesh.indices().len() as u32, instance_count: 1, first_index: 0, vertex_offset: 0, first_instance: 0 };
    let mut generated = Uploaded {
        draws: upload(device.as_ref(), bytemuck::bytes_of(&draw), BufferUsage::STORAGE, "generated draws"),
        vertices,
        indices: upload(device.as_ref(), bytemuck::cast_slice(mesh.indices()), BufferUsage::INDEX, "generated indices"),
        commands: upload_device_local(device.as_ref(), &[bytemuck::bytes_of(&command), &1u32.to_ne_bytes()].concat(), BufferUsage::INDIRECT, ResourceState::IndirectArgs, "generated commands"),
        cull_back_faces,
    };
    let mut renderer = ForwardRenderer::new(device.clone(), config).unwrap();
    let actual = render(&mut renderer, &sample.scene, Some(&mut generated));
    drop(renderer);
    for b in [generated.draws, generated.vertices, generated.indices, generated.commands] {
        device.destroy_buffer(b);
    }

    let diff = compare_images(&expected, &actual, 0);
    if diff.size_mismatch || diff.pixels_over_tolerance > 0 {
        // As the golden tests do: mismatching images go to golden_out/ to be looked at.
        let out = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden_out");
        let name = format!("generated_{scene_id:?}_{}", if mesh_shaders { "mesh" } else { "vertex" });
        write_ppm(&out.join(format!("{name}.expected.ppm")), &expected).unwrap();
        write_ppm(&out.join(format!("{name}.actual.ppm")), &actual).unwrap();
    }
    assert!(!diff.size_mismatch);
    assert_eq!(diff.pixels_over_tolerance, 0, "{scene_id:?} (mesh shaders {mesh_shaders}): {diff:?}");
    assert!(expected.rgba.chunks_exact(4).any(|p| p != &expected.rgba[..4]), "the picture should not be empty");
    let (errors, warnings) = vulkan.validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation reported errors or warnings");
}

#[test]
#[serial]
fn a_generated_mesh_looks_like_the_scene_object() {
    let Some(vulkan) = device() else { return };
    generated_like_scene_objects(&vulkan, SampleSceneId::Triangle, 0.0, false);
    // A single-sided object with back faces culled, among others drawn by the mesh path.
    generated_like_scene_objects(&vulkan, SampleSceneId::Cubes, 0.7, true);
}
