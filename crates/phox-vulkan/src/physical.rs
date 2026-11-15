//! Picking the physical device.

use crate::instance::Instance;
use ash::vk;
use phox_rhi::{Result, RhiError};

pub struct Selected {
    pub physical: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_family: u32,
    pub mesh_shaders: bool,
    pub draw_indirect_count: bool,
}

fn device_name(p: &vk::PhysicalDeviceProperties) -> String {
    p.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

pub fn has_device_extension(instance: &Instance, physical: vk::PhysicalDevice, name: &std::ffi::CStr) -> bool {
    let extensions = unsafe { instance.raw.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    extensions.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
}

/// Whether the device can draw GPU-driven (multi-draw indirect with a first instance).
pub fn has_gpu_driven_drawing(f: &vk::PhysicalDeviceFeatures) -> bool {
    f.multi_draw_indirect == vk::TRUE && f.draw_indirect_first_instance == vk::TRUE
}

/// Whether the device can index one table of every texture (descriptor indexing).
pub fn has_bindless_textures(f: &vk::PhysicalDeviceVulkan12Features) -> bool {
    f.descriptor_indexing == vk::TRUE
        && f.runtime_descriptor_array == vk::TRUE
        && f.descriptor_binding_partially_bound == vk::TRUE
        && f.descriptor_binding_sampled_image_update_after_bind == vk::TRUE
        && f.descriptor_binding_update_unused_while_pending == vk::TRUE
        && f.shader_sampled_image_array_non_uniform_indexing == vk::TRUE
}

/// The best device meeting Phox's requirements (Vulkan 1.3 with synchronization2, dynamic rendering,
/// descriptor indexing, timeline semaphores, buffer device addresses and multi-draw indirect), preferring
/// discrete over integrated over virtual over CPU. `preferred` is a case-insensitive substring of its name.
pub fn select(instance: &Instance, preferred: &str, want_mesh_shaders: bool) -> Result<Selected> {
    let devices = unsafe { instance.raw.enumerate_physical_devices() }.map_err(|e| RhiError::Backend(format!("cannot list Vulkan devices: {e}")))?;
    let preferred = preferred.to_lowercase();
    let mut best: Option<(i32, Selected)> = None;
    for physical in devices {
        let properties = unsafe { instance.raw.get_physical_device_properties(physical) };
        if !preferred.is_empty() && !device_name(&properties).to_lowercase().contains(&preferred) {
            continue;
        }
        let families = unsafe { instance.raw.get_physical_device_queue_family_properties(physical) };
        let Some(queue_family) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER))
        else {
            continue;
        };

        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut f13);
        unsafe { instance.raw.get_physical_device_features2(physical, &mut f2) };
        let core = f2.features;
        if properties.api_version < vk::API_VERSION_1_3
            || f13.synchronization2 != vk::TRUE
            || f13.dynamic_rendering != vk::TRUE
            || f12.timeline_semaphore != vk::TRUE
            || f12.buffer_device_address != vk::TRUE
            || !has_bindless_textures(&f12)
            || !has_gpu_driven_drawing(&core)
        {
            continue;
        }

        let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut mesh_query = vk::PhysicalDeviceFeatures2::default().push_next(&mut mesh);
        let mesh_shaders = want_mesh_shaders && has_device_extension(instance, physical, ash::ext::mesh_shader::NAME) && {
            unsafe { instance.raw.get_physical_device_features2(physical, &mut mesh_query) };
            mesh.task_shader == vk::TRUE && mesh.mesh_shader == vk::TRUE
        };
        let disabled = std::env::var("PHOX_NO_DRAW_INDIRECT_COUNT").is_ok_and(|v| v == "1");
        let score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 50,
            vk::PhysicalDeviceType::CPU => 10,
            _ => 1,
        };
        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((
                score,
                Selected { physical, properties, queue_family: queue_family as u32, mesh_shaders, draw_indirect_count: f12.draw_indirect_count == vk::TRUE && !disabled },
            ));
        }
    }
    best.map(|(_, s)| s).ok_or_else(|| {
        RhiError::Backend(format!(
            "{} (Phox needs Vulkan 1.3 with synchronization2, dynamic rendering, descriptor indexing and multi-draw indirect)",
            if preferred.is_empty() { "No suitable Vulkan device found".to_string() } else { format!("No suitable Vulkan device matching '{preferred}'") }
        ))
    })
}
