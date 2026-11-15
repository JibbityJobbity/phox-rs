//! The Vulkan instance, with the validation layer (and its synchronization checks) on request.

use ash::{ext, khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use phox_rhi::{Result, RhiError};
use std::ffi::{CStr, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Messages the validation layer reported so far.
#[derive(Debug, Default)]
pub struct ValidationCounters {
    pub errors: AtomicU32,
    pub warnings: AtomicU32,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: `user` is the Arc<ValidationCounters> the messenger was created with, kept alive by Instance.
    let counters = unsafe { &*(user as *const ValidationCounters) };
    // SAFETY: the loader hands a valid callback data pointer.
    let data = unsafe { &*data };
    // SAFETY: the strings are valid NUL-terminated strings for the duration of the callback.
    let (id, message) = unsafe {
        (
            data.message_id_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            data.message_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        )
    };
    // GENERAL messages come from the loader (e.g. an ICD on the system that fails to load), not from
    // validating this program, so they are neither counted nor reported above debug level.
    let from_validation = types.intersects(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE);
    if !from_validation {
        log::debug!("[vulkan] {id}: {message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        counters.errors.fetch_add(1, Ordering::Relaxed);
        log::error!("[vulkan] {id}: {message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        counters.warnings.fetch_add(1, Ordering::Relaxed);
        log::warn!("[vulkan] {id}: {message}");
    } else {
        log::debug!("[vulkan] {id}: {message}");
    }
    vk::FALSE // never abort the call that triggered the message
}

pub struct Instance {
    /// Keeps the Vulkan library loaded.
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub raw: ash::Instance,
    pub validation: bool,
    pub counters: Arc<ValidationCounters>,
    messenger: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    /// The window's surface, when presenting.
    pub surface: Option<(khr::surface::Instance, vk::SurfaceKHR)>,
    counters_raw: *const ValidationCounters,
}

// SAFETY: the raw pointer is the Arc's own data pointer, only read by the callback and released on drop.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

fn backend(message: impl std::fmt::Display) -> RhiError {
    RhiError::Backend(message.to_string())
}

impl Instance {
    /// Loads the Vulkan library and creates an instance (API 1.3). `validation` enables the Khronos layer with
    /// synchronization validation, when it is installed.
    pub fn new(validation: bool, window: Option<(RawDisplayHandle, RawWindowHandle)>) -> Result<Self> {
        // SAFETY: loading the system's Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| backend(format!("Vulkan is unavailable in this environment: {e}")))?;
        let layers = unsafe { entry.enumerate_instance_layer_properties() }.map_err(backend)?;
        let layer_present = layers.iter().any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == VALIDATION_LAYER));
        if validation && !layer_present {
            log::warn!("validation requested but {} is not installed; continuing without it", VALIDATION_LAYER.to_string_lossy());
        }
        let validation = validation && layer_present;

        let app_info = vk::ApplicationInfo::default().application_name(c"Phox").api_version(vk::API_VERSION_1_3);
        let mut layer_names: Vec<*const i8> = Vec::new();
        let mut extensions: Vec<*const i8> = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(backend)?;
        let has = |name: &CStr| available.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name));
        if has(ash::khr::portability_enumeration::NAME) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        // A window's surface needs the platform's surface extensions.
        if let Some((display, _)) = window {
            let required = ash_window::enumerate_required_extensions(display).map_err(|e| backend(format!("cannot present to this window system: {e}")))?;
            for &name in required {
                if !extensions.contains(&name) {
                    extensions.push(name);
                }
            }
        }

        // Synchronization validation goes through VK_EXT_layer_settings, which the layer provides.
        let sync_value: [u32; 1] = [1];
        let setting_bytes: Vec<u8> = sync_value.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let setting = vk::LayerSettingEXT::default()
            .layer_name(VALIDATION_LAYER)
            .setting_name(c"validate_sync")
            .ty(vk::LayerSettingTypeEXT::BOOL32)
            .values(&setting_bytes);
        let settings = [setting];
        let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
        let mut use_layer_settings = false;
        if validation {
            layer_names.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ext::debug_utils::NAME.as_ptr());
            let layer_exts = unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }.unwrap_or_default();
            if layer_exts.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == ext::layer_settings::NAME)) {
                extensions.push(ext::layer_settings::NAME.as_ptr());
                use_layer_settings = true;
            } else {
                log::warn!("the validation layer lacks VK_EXT_layer_settings; synchronization validation stays off");
            }
        }

        let counters = Arc::new(ValidationCounters::default());
        let counters_raw = Arc::into_raw(counters.clone());
        let debug_enabled = log::log_enabled!(log::Level::Debug);
        let messenger_info = || {
            vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | if debug_enabled { vk::DebugUtilsMessageSeverityFlagsEXT::INFO } else { vk::DebugUtilsMessageSeverityFlagsEXT::empty() },
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback))
                .user_data(counters_raw as *mut c_void)
        };
        let mut chained_messenger = messenger_info();

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names)
            .flags(flags);
        if validation {
            // Also validates instance creation and destruction themselves.
            create_info = create_info.push_next(&mut chained_messenger);
            if use_layer_settings {
                create_info = create_info.push_next(&mut layer_settings);
            }
        }
        let raw = unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
            // SAFETY: undoes into_raw above; nothing else holds the pointer yet.
            drop(unsafe { Arc::from_raw(counters_raw) });
            backend(format!("failed to create the Vulkan instance: {e}"))
        })?;

        let messenger = if validation {
            let loader = ext::debug_utils::Instance::new(&entry, &raw);
            let handle = unsafe { loader.create_debug_utils_messenger(&messenger_info(), None) }.map_err(backend)?;
            Some((loader, handle))
        } else {
            None
        };
        let surface = match window {
            Some((display, handle)) => {
                // SAFETY: the caller keeps the window alive as long as the device (see VulkanDevice::new_for_window).
                let surface = unsafe { ash_window::create_surface(&entry, &raw, display, handle, None) }.map_err(|e| backend(format!("failed to create the window surface: {e}")))?;
                Some((khr::surface::Instance::new(&entry, &raw), surface))
            }
            None => None,
        };
        Ok(Self { entry, raw, validation, counters, messenger, surface, counters_raw })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the instance is no longer used; the callback cannot run after the messenger is destroyed.
        unsafe {
            if let Some((loader, surface)) = self.surface.take() {
                loader.destroy_surface(surface, None);
            }
            if let Some((loader, handle)) = self.messenger.take() {
                loader.destroy_debug_utils_messenger(handle, None);
            }
            self.raw.destroy_instance(None);
            drop(Arc::from_raw(self.counters_raw));
        }
    }
}
