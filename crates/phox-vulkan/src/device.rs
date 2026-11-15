//! The Vulkan device: resources, deferred destruction, the timeline semaphore and submission.

use crate::commands::VulkanCommandList;
use crate::convert;
use crate::instance::Instance;
use crate::physical;
use ash::{ext, vk};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use phox_rhi::pool::Pool;
use phox_rhi::*;
use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};

/// The most textures the bindless table holds, if the device allows.
const MAX_BINDLESS_TEXTURES: u32 = 4096;
/// GPU timestamps one command list can write.
pub(crate) const TIMESTAMP_CAPACITY: u32 = 128;

pub(crate) fn backend(e: impl std::fmt::Display) -> RhiError {
    RhiError::Backend(e.to_string())
}

/// How to create a [`VulkanDevice`].
#[derive(Clone, Debug)]
pub struct VulkanConfig {
    /// The Khronos validation layer with synchronization validation (when installed).
    pub validation: bool,
    /// Case-insensitive substring of the GPU name. Empty means the `PHOX_DEVICE` environment variable if
    /// set, otherwise the best available device.
    pub preferred_device: String,
    /// Use task and mesh shaders where the device has them.
    pub mesh_shaders: bool,
    /// Use `drawIndirectCount` where the device has it (the environment's `PHOX_NO_DRAW_INDIRECT_COUNT=1`
    /// turns it off too). Off tests the path for devices without it, such as MoltenVK.
    pub draw_indirect_count: bool,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self { validation: false, preferred_device: String::new(), mesh_shaders: true, draw_indirect_count: true }
    }
}

/// What the device is, for `--stats`-style output and tests.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub validation: bool,
    pub mesh_shaders: bool,
    pub draw_indirect_count: bool,
    /// The largest task shader dispatch in x.
    pub max_task_workgroup_count_x: u32,
}

pub(crate) struct BufferRes {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub desc: BufferDesc,
    pub address: u64,
}

pub(crate) struct TextureRes {
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// One view per mip, for storage textures with several mips; otherwise empty.
    pub mip_views: Vec<vk::ImageView>,
    pub allocation: Option<Allocation>,
    pub desc: TextureDesc,
}

pub(crate) struct ShaderRes {
    pub module: vk::ShaderModule,
    pub stage: ShaderStage,
    pub entry: CString,
}

pub(crate) struct PipelineRes {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub bind_point: vk::PipelineBindPoint,
    /// One layout per set 0..=max, `None` for sets the pipeline does not use.
    pub set_bindings: Vec<Vec<Binding>>,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
}

/// Command pool, descriptor pools and query pool of one command list, recycled once its work completes.
pub(crate) struct CommandEntry {
    pub pool: vk::CommandPool,
    pub descriptor_pools: Vec<vk::DescriptorPool>,
    pub query_pool: Option<vk::QueryPool>,
}

pub(crate) enum Deferred {
    Buffer(vk::Buffer, Option<Allocation>),
    Image(vk::Image, Vec<vk::ImageView>, Option<Allocation>),
    Shader(vk::ShaderModule),
    Pipeline(vk::Pipeline, vk::PipelineLayout),
}

pub(crate) struct BindlessTable {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    pub capacity: u32,
    pub next: u32,
    pub free: Vec<u32>,
}

/// What a descriptor set layout contains: (binding, type, stages, count, sampler) of each binding.
pub(crate) type SetLayoutKey = Vec<(u32, u32, u32, u32, SamplerDesc)>;

pub(crate) struct State {
    pub allocator: Option<Allocator>,
    pub buffers: Pool<BufferTag, BufferRes>,
    pub textures: Pool<TextureTag, TextureRes>,
    pub shaders: Pool<ShaderTag, ShaderRes>,
    pub pipelines: Pool<PipelineTag, PipelineRes>,
    /// The last ticket handed out.
    pub submitted: u64,
    pub deferred: VecDeque<(u64, Deferred)>,
    pub free_entries: Vec<CommandEntry>,
    pub inflight: Vec<(u64, CommandEntry, Vec<String>)>,
    pub timestamps: VecDeque<(u64, Vec<Timestamp>)>,
    pub samplers: HashMap<SamplerDesc, vk::Sampler>,
    pub set_layouts: HashMap<SetLayoutKey, vk::DescriptorSetLayout>,
    pub bindless: Option<BindlessTable>,
}

pub(crate) struct Inner {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub mesh_loader: Option<ext::mesh_shader::Device>,
    pub debug_utils: Option<ext::debug_utils::Device>,
    pub caps: Capabilities,
    pub info: DeviceInfo,
    pub timeline: vk::Semaphore,
    pub timestamp_period: f32,
    pub state: Mutex<State>,
    /// Dropped last: the device must go before the instance.
    pub instance: Instance,
}

impl Inner {
    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn mesh_shaders(&self) -> bool {
        self.info.mesh_shaders
    }

    fn completed(&self) -> u64 {
        unsafe { self.device.get_semaphore_counter_value(self.timeline) }.unwrap_or(0)
    }

    /// Frees what the GPU has finished with and recycles completed command entries.
    pub(crate) fn collect(&self, st: &mut State) {
        let done = self.completed();
        while st.deferred.front().is_some_and(|(t, _)| *t <= done) {
            let (_, item) = st.deferred.pop_front().expect("checked");
            self.free_now(st, item);
        }
        let mut i = 0;
        while i < st.inflight.len() {
            if st.inflight[i].0 <= done {
                let (ticket, entry, names) = st.inflight.swap_remove(i);
                if !names.is_empty() {
                    if let Some(query_pool) = entry.query_pool {
                        let stamps = self.read_queries(query_pool, &names);
                        st.timestamps.push_back((ticket, stamps));
                        if st.timestamps.len() > 16 {
                            st.timestamps.pop_front();
                        }
                    }
                }
                st.free_entries.push(entry);
            } else {
                i += 1;
            }
        }
    }

    fn read_queries(&self, query_pool: vk::QueryPool, names: &[String]) -> Vec<Timestamp> {
        let mut ticks = vec![0u64; names.len()];
        let result = unsafe { self.device.get_query_pool_results(query_pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) };
        if result.is_err() {
            return Vec::new();
        }
        let first = ticks.first().copied().unwrap_or(0);
        names
            .iter()
            .zip(&ticks)
            .map(|(name, &t)| Timestamp { name: name.clone(), nanoseconds: t.wrapping_sub(first) as f64 * self.timestamp_period as f64 })
            .collect()
    }

    fn free_now(&self, st: &mut State, item: Deferred) {
        unsafe {
            match item {
                Deferred::Buffer(buffer, allocation) => {
                    self.device.destroy_buffer(buffer, None);
                    if let (Some(a), Some(allocator)) = (allocation, st.allocator.as_mut()) {
                        let _ = allocator.free(a);
                    }
                }
                Deferred::Image(image, views, allocation) => {
                    for view in views {
                        self.device.destroy_image_view(view, None);
                    }
                    self.device.destroy_image(image, None);
                    if let (Some(a), Some(allocator)) = (allocation, st.allocator.as_mut()) {
                        let _ = allocator.free(a);
                    }
                }
                Deferred::Shader(module) => self.device.destroy_shader_module(module, None),
                Deferred::Pipeline(pipeline, layout) => {
                    self.device.destroy_pipeline(pipeline, None);
                    self.device.destroy_pipeline_layout(layout, None);
                }
            }
        }
    }

    /// Queues `item` for destruction once every submission made so far, and the one being recorded, is done.
    pub(crate) fn defer(&self, st: &mut State, item: Deferred) {
        let ticket = st.submitted + 1;
        st.deferred.push_back((ticket, item));
    }

    pub(crate) fn sampler(&self, st: &mut State, desc: &SamplerDesc) -> Result<vk::Sampler> {
        if let Some(&s) = st.samplers.get(desc) {
            return Ok(s);
        }
        let mipmap = if desc.mip_filter == Filter::Nearest { vk::SamplerMipmapMode::NEAREST } else { vk::SamplerMipmapMode::LINEAR };
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(convert::filter(desc.mag_filter))
            .min_filter(convert::filter(desc.min_filter))
            .mipmap_mode(mipmap)
            .address_mode_u(convert::wrap(desc.wrap_u))
            .address_mode_v(convert::wrap(desc.wrap_v))
            .address_mode_w(convert::wrap(desc.wrap_v))
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe { self.device.create_sampler(&info, None) }.map_err(backend)?;
        st.samplers.insert(*desc, sampler);
        Ok(sampler)
    }

    pub(crate) fn set_name(&self, object: impl vk::Handle, name: &str) {
        if name.is_empty() || !self.instance.validation {
            return;
        }
        let (Ok(c_name), Some(loader)) = (CString::new(name), &self.debug_utils) else { return };
        let info = vk::DebugUtilsObjectNameInfoEXT::default().object_handle(object).object_name(&c_name);
        let _ = unsafe { loader.set_debug_utils_object_name(&info) };
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
            while let Some((_, item)) = st.deferred.pop_front() {
                self.free_now(&mut st, item);
            }
            let st = &mut *st;
            for b in st.buffers.drain() {
                self.device.destroy_buffer(b.buffer, None);
                if let (Some(a), Some(allocator)) = (b.allocation, st.allocator.as_mut()) {
                    let _ = allocator.free(a);
                }
            }
            for t in st.textures.drain() {
                for view in std::iter::once(t.view).chain(t.mip_views) {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image(t.image, None);
                if let (Some(a), Some(allocator)) = (t.allocation, st.allocator.as_mut()) {
                    let _ = allocator.free(a);
                }
            }
            for s in st.shaders.drain() {
                self.device.destroy_shader_module(s.module, None);
            }
            for p in st.pipelines.drain() {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.layout, None);
            }
            let entries: Vec<CommandEntry> = st.free_entries.drain(..).chain(st.inflight.drain(..).map(|(_, e, _)| e)).collect();
            for entry in entries {
                for pool in entry.descriptor_pools {
                    self.device.destroy_descriptor_pool(pool, None);
                }
                if let Some(q) = entry.query_pool {
                    self.device.destroy_query_pool(q, None);
                }
                self.device.destroy_command_pool(entry.pool, None);
            }
            for (_, s) in st.samplers.drain() {
                self.device.destroy_sampler(s, None);
            }
            for (_, l) in st.set_layouts.drain() {
                self.device.destroy_descriptor_set_layout(l, None);
            }
            if let Some(table) = st.bindless.take() {
                self.device.destroy_descriptor_pool(table.pool, None);
                self.device.destroy_descriptor_set_layout(table.layout, None);
            }
            self.device.destroy_semaphore(self.timeline, None);
            st.allocator = None; // frees its memory blocks while the device still exists
            self.device.destroy_device(None);
        }
    }
}

/// A Vulkan 1.3 device. See [`VulkanDevice::new`].
#[derive(Clone)]
pub struct VulkanDevice {
    pub(crate) inner: Arc<Inner>,
}

impl VulkanDevice {
    /// Creates the instance, picks a device (see [`physical::select`]) and creates the logical device with
    /// the features Phox needs. Fails with a message when no device qualifies.
    pub fn new(config: &VulkanConfig) -> Result<Self> {
        let instance = Instance::new(config.validation)?;
        let preferred = if config.preferred_device.is_empty() { std::env::var("PHOX_DEVICE").unwrap_or_default() } else { config.preferred_device.clone() };
        let selected = physical::select(&instance, &preferred, config.mesh_shaders, config.draw_indirect_count)?;

        let priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(selected.queue_family).queue_priorities(&priority)];
        let mut extensions: Vec<*const i8> = Vec::new();
        // A portability implementation (MoltenVK) must have its subset enabled whenever it exposes it.
        if physical::has_device_extension(&instance, selected.physical, ash::khr::portability_subset::NAME) {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        if selected.mesh_shaders {
            extensions.push(ext::mesh_shader::NAME.as_ptr());
        }

        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .draw_indirect_count(selected.draw_indirect_count);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).dynamic_rendering(true).maintenance4(true);
        // Only what Phox uses: enabling e.g. multiviewMeshShader would require multiview too.
        let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().task_shader(true).mesh_shader(true);
        let mut f2 = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().multi_draw_indirect(true).draw_indirect_first_instance(true))
            .push_next(&mut f12)
            .push_next(&mut f13);
        if selected.mesh_shaders {
            f2 = f2.push_next(&mut mesh);
        }
        let create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info).enabled_extension_names(&extensions).push_next(&mut f2);
        let device = unsafe { instance.raw.create_device(selected.physical, &create_info, None) }.map_err(|e| backend(format!("failed to create the logical device: {e}")))?;
        let queue = unsafe { device.get_device_queue(selected.queue_family, 0) };

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw.clone(),
            device: device.clone(),
            physical_device: selected.physical,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| backend(format!("failed to create the memory allocator: {e}")))?;
        let _ = &mut allocator;

        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let timeline = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type), None) }.map_err(backend)?;

        // Limits and the bindless table's size.
        let mut props12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut mesh_props = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props12);
        if selected.mesh_shaders {
            props2 = props2.push_next(&mut mesh_props);
        }
        unsafe { instance.raw.get_physical_device_properties2(selected.physical, &mut props2) };
        let limits = selected.properties.limits;
        let bindless_capacity = MAX_BINDLESS_TEXTURES
            .min(props12.max_descriptor_set_update_after_bind_sampled_images)
            .min(props12.max_per_stage_descriptor_update_after_bind_sampled_images)
            .min(props12.max_descriptor_set_update_after_bind_samplers)
            .min(props12.max_per_stage_descriptor_update_after_bind_samplers);

        let name = selected.properties.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut features = FeatureSet::default();
        features.insert(Feature::BindlessDescriptors);
        if selected.mesh_shaders {
            features.insert(Feature::MeshShader);
        }
        if selected.draw_indirect_count {
            features.insert(Feature::DrawIndirectCount);
        }
        let timestamps = limits.timestamp_compute_and_graphics == vk::TRUE && limits.timestamp_period > 0.0;
        if timestamps {
            features.insert(Feature::Timestamps);
        }
        let caps = Capabilities {
            device_name: name.clone(),
            api_major: vk::api_version_major(selected.properties.api_version),
            api_minor: vk::api_version_minor(selected.properties.api_version),
            features,
            max_texture_size_2d: limits.max_image_dimension2_d,
            max_push_constant_bytes: limits.max_push_constants_size,
            max_draw_indirect_count: limits.max_draw_indirect_count,
            max_task_workgroup_count_x: if selected.mesh_shaders { mesh_props.max_task_work_group_count[0] } else { 0 },
            timestamp_period_ns: limits.timestamp_period,
            bindless_texture_capacity: bindless_capacity,
        };
        let info = DeviceInfo {
            name,
            validation: instance.validation,
            mesh_shaders: selected.mesh_shaders,
            draw_indirect_count: selected.draw_indirect_count,
            max_task_workgroup_count_x: if selected.mesh_shaders { mesh_props.max_task_work_group_count[0] } else { 0 },
        };
        log::debug!("Vulkan device {} (mesh shaders: {}, indirect count: {})", info.name, info.mesh_shaders, info.draw_indirect_count);

        let mesh_loader = selected.mesh_shaders.then(|| ext::mesh_shader::Device::new(&instance.raw, &device));
        let debug_utils = instance.validation.then(|| ext::debug_utils::Device::new(&instance.raw, &device));
        let state = State {
            allocator: Some(allocator),
            buffers: Pool::new("buffer"),
            textures: Pool::new("texture"),
            shaders: Pool::new("shader"),
            pipelines: Pool::new("pipeline"),
            submitted: 0,
            deferred: VecDeque::new(),
            free_entries: Vec::new(),
            inflight: Vec::new(),
            timestamps: VecDeque::new(),
            samplers: HashMap::new(),
            set_layouts: HashMap::new(),
            bindless: None,
        };
        let inner = Arc::new(Inner {
            device,
            queue,
            queue_family: selected.queue_family,
            mesh_loader,
            debug_utils,
            caps,
            info,
            timeline,
            timestamp_period: limits.timestamp_period,
            state: Mutex::new(state),
            instance,
        });
        inner.create_bindless_table()?;
        Ok(Self { inner })
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }

    /// Validation messages reported so far: (errors, warnings).
    pub fn validation_counts(&self) -> (u32, u32) {
        let c = &self.inner.instance.counters;
        (c.errors.load(Ordering::Relaxed), c.warnings.load(Ordering::Relaxed))
    }

    /// The device address of a buffer made with `device_address: true`.
    pub fn buffer_address(&self, buffer: BufferHandle) -> Result<u64> {
        let st = self.inner.lock();
        let b = st.buffers.get(buffer)?;
        if b.address == 0 {
            return Err(RhiError::Misuse(format!("buffer '{}' was not created with device_address", b.desc.debug_name)));
        }
        Ok(b.address)
    }
}

impl Inner {
    fn create_bindless_table(&self) -> Result<()> {
        let capacity = self.caps.bindless_texture_capacity;
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE | if self.info.mesh_shaders { vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT } else { vk::ShaderStageFlags::empty() };
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(capacity)
            .stage_flags(stages)];
        let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];
        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&binding)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut flags_info);
        let layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }.map_err(backend)?;
        let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(capacity)];
        let pool_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes).flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        let pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }.map_err(backend)?;
        let layouts = [layout];
        let set = unsafe { self.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.map_err(backend)?[0];
        self.lock().bindless = Some(BindlessTable { layout, pool, set, capacity, next: 0, free: Vec::new() });
        Ok(())
    }

    fn alloc_location(memory: MemoryKind) -> MemoryLocation {
        match memory {
            MemoryKind::GpuOnly => MemoryLocation::GpuOnly,
            MemoryKind::Upload => MemoryLocation::CpuToGpu,
            MemoryKind::Readback => MemoryLocation::GpuToCpu,
        }
    }
}

impl phox_rhi::Device for VulkanDevice {
    fn capabilities(&self) -> Capabilities {
        self.inner.caps.clone()
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferHandle> {
        let inner = &self.inner;
        if desc.size == 0 {
            return Err(RhiError::InvalidDesc("create_buffer: size must be greater than zero".into()));
        }
        let usage = convert::buffer_usage(desc.usage, desc.device_address);
        if usage.is_empty() {
            return Err(RhiError::InvalidDesc(format!("buffer '{}' has no GPU usage", desc.debug_name)));
        }
        let info = vk::BufferCreateInfo::default().size(desc.size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { inner.device.create_buffer(&info, None) }.map_err(|e| backend(format!("failed to create buffer '{}': {e}", desc.debug_name)))?;
        let requirements = unsafe { inner.device.get_buffer_memory_requirements(buffer) };
        let mut st = inner.lock();
        let allocation = st
            .allocator
            .as_mut()
            .expect("allocator exists while the device does")
            .allocate(&AllocationCreateDesc {
                name: &desc.debug_name,
                requirements,
                location: Inner::alloc_location(desc.memory),
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| {
                unsafe { inner.device.destroy_buffer(buffer, None) };
                backend(format!("failed to allocate memory for buffer '{}': {e}", desc.debug_name))
            })?;
        unsafe { inner.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.map_err(backend)?;
        let address = if desc.device_address {
            unsafe { inner.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) }
        } else {
            0
        };
        inner.set_name(buffer, &desc.debug_name);
        Ok(st.buffers.create(BufferRes { buffer, allocation: Some(allocation), desc: desc.clone(), address }))
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureHandle> {
        let inner = &self.inner;
        if desc.width == 0 || desc.height == 0 || desc.mips == 0 {
            return Err(RhiError::InvalidDesc("create_texture: width, height and mips must be greater than zero".into()));
        }
        if desc.format == Format::Undefined {
            return Err(RhiError::InvalidDesc("create_texture: format must not be Undefined".into()));
        }
        let usage = convert::image_usage(desc.usage);
        if usage.is_empty() {
            return Err(RhiError::InvalidDesc(format!("texture '{}' has no GPU usage", desc.debug_name)));
        }
        let format = convert::format(desc.format);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: desc.width, height: desc.height, depth: 1 })
            .mip_levels(desc.mips)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { inner.device.create_image(&info, None) }.map_err(|e| backend(format!("failed to create image '{}': {e}", desc.debug_name)))?;
        let requirements = unsafe { inner.device.get_image_memory_requirements(image) };
        let mut st = inner.lock();
        let allocation = st
            .allocator
            .as_mut()
            .expect("allocator exists while the device does")
            .allocate(&AllocationCreateDesc {
                name: &desc.debug_name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| {
                unsafe { inner.device.destroy_image(image, None) };
                backend(format!("failed to allocate memory for image '{}': {e}", desc.debug_name))
            })?;
        unsafe { inner.device.bind_image_memory(image, allocation.memory(), allocation.offset()) }.map_err(backend)?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: convert::aspect(desc.format), base_mip_level: 0, level_count: desc.mips, base_array_layer: 0, layer_count: 1 });
        let view = unsafe { inner.device.create_image_view(&view_info, None) }.map_err(backend)?;
        let mut mip_views = Vec::new();
        if desc.usage.contains(TextureUsage::STORAGE) && desc.mips > 1 {
            for mip in 0..desc.mips {
                let mip_info = view_info.subresource_range(vk::ImageSubresourceRange { aspect_mask: convert::aspect(desc.format), base_mip_level: mip, level_count: 1, base_array_layer: 0, layer_count: 1 });
                mip_views.push(unsafe { inner.device.create_image_view(&mip_info, None) }.map_err(backend)?);
            }
        }
        inner.set_name(image, &desc.debug_name);
        Ok(st.textures.create(TextureRes { image, view, mip_views, allocation: Some(allocation), desc: desc.clone() }))
    }

    fn create_shader(&self, desc: &ShaderDesc) -> Result<ShaderHandle> {
        if desc.spirv.is_empty() {
            return Err(RhiError::InvalidDesc("create_shader: spirv must not be empty".into()));
        }
        let info = vk::ShaderModuleCreateInfo::default().code(&desc.spirv);
        let module = unsafe { self.inner.device.create_shader_module(&info, None) }.map_err(|e| backend(format!("failed to create a shader module: {e}")))?;
        let entry = CString::new(desc.entry.clone()).map_err(|_| RhiError::InvalidDesc("shader entry point contains a NUL".into()))?;
        Ok(self.inner.lock().shaders.create(ShaderRes { module, stage: desc.stage, entry }))
    }

    fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle> {
        self.inner.create_graphics_pipeline(desc)
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle> {
        self.inner.create_compute_pipeline(desc)
    }

    fn destroy_buffer(&self, handle: BufferHandle) {
        let inner = &self.inner;
        let mut st = inner.lock();
        if let Some(b) = st.buffers.destroy(handle) {
            inner.defer(&mut st, Deferred::Buffer(b.buffer, b.allocation));
        }
    }

    fn destroy_texture(&self, handle: TextureHandle) {
        let inner = &self.inner;
        let mut st = inner.lock();
        if let Some(t) = st.textures.destroy(handle) {
            inner.defer(&mut st, Deferred::Image(t.image, std::iter::once(t.view).chain(t.mip_views).collect(), t.allocation));
        }
    }

    fn destroy_shader(&self, handle: ShaderHandle) {
        let inner = &self.inner;
        let mut st = inner.lock();
        if let Some(s) = st.shaders.destroy(handle) {
            inner.defer(&mut st, Deferred::Shader(s.module));
        }
    }

    fn destroy_pipeline(&self, handle: PipelineHandle) {
        let inner = &self.inner;
        let mut st = inner.lock();
        if let Some(p) = st.pipelines.destroy(handle) {
            inner.defer(&mut st, Deferred::Pipeline(p.pipeline, p.layout));
        }
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let mut st = self.inner.lock();
        let b = st.buffers.get_mut(buffer)?;
        if b.desc.memory == MemoryKind::GpuOnly {
            return Err(RhiError::Misuse("write_buffer: buffer memory is GpuOnly".into()));
        }
        let size = b.desc.size;
        let end = offset.checked_add(data.len() as u64).filter(|&e| e <= size).ok_or_else(|| RhiError::Misuse("write_buffer: write is out of bounds".into()))?;
        let mapped = b.allocation.as_mut().and_then(|a| a.mapped_slice_mut()).ok_or_else(|| RhiError::Misuse("write_buffer: buffer is not mapped".into()))?;
        mapped[offset as usize..end as usize].copy_from_slice(data);
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, out: &mut [u8]) -> Result<()> {
        let st = self.inner.lock();
        let b = st.buffers.get(buffer)?;
        if b.desc.memory == MemoryKind::GpuOnly {
            return Err(RhiError::Misuse("read_buffer: buffer memory is GpuOnly".into()));
        }
        let end = offset.checked_add(out.len() as u64).filter(|&e| e <= b.desc.size).ok_or_else(|| RhiError::Misuse("read_buffer: read is out of bounds".into()))?;
        let mapped = b.allocation.as_ref().and_then(|a| a.mapped_slice()).ok_or_else(|| RhiError::Misuse("read_buffer: buffer is not mapped".into()))?;
        out.copy_from_slice(&mapped[offset as usize..end as usize]);
        Ok(())
    }

    fn describe_texture(&self, handle: TextureHandle) -> Result<TextureDesc> {
        self.inner.lock().textures.get(handle).map(|t| t.desc.clone())
    }

    fn describe_buffer(&self, handle: BufferHandle) -> Result<BufferDesc> {
        self.inner.lock().buffers.get(handle).map(|b| b.desc.clone())
    }

    fn buffer_address(&self, buffer: BufferHandle) -> Result<u64> {
        VulkanDevice::buffer_address(self, buffer)
    }

    fn register_bindless_texture(&self, texture: TextureHandle, sampler: &SamplerDesc) -> Result<BindlessIndex> {
        let inner = &self.inner;
        let mut st = inner.lock();
        let sampler = inner.sampler(&mut st, sampler)?;
        let (view, depth) = {
            let t = st.textures.get(texture)?;
            (t.view, t.desc.format.is_depth())
        };
        let table = st.bindless.as_mut().ok_or_else(|| RhiError::Unsupported("BindlessDescriptors".into()))?;
        let index = if let Some(i) = table.free.pop() {
            i
        } else if table.next < table.capacity {
            table.next += 1;
            table.next - 1
        } else {
            return Err(RhiError::Misuse("the bindless texture table is full".into()));
        };
        let layout = if depth { vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL } else { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL };
        let image_info = [vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(layout)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(table.set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);
        unsafe { inner.device.update_descriptor_sets(&[write], &[]) };
        Ok(BindlessIndex(index))
    }

    fn unregister_bindless_texture(&self, index: BindlessIndex) {
        let mut st = self.inner.lock();
        if let Some(table) = st.bindless.as_mut() {
            if index.0 < table.next && !table.free.contains(&index.0) {
                table.free.push(index.0);
            }
        }
    }

    fn begin_commands(&self) -> Result<Box<dyn CommandList>> {
        Ok(Box::new(VulkanCommandList::begin(self.inner.clone())?))
    }

    fn submit(&self, commands: Box<dyn CommandList>) -> Result<SubmitTicket> {
        let list = commands
            .into_any()
            .downcast::<VulkanCommandList>()
            .map_err(|_| RhiError::Misuse("submit: the command list was not made by this device".into()))?;
        list.submit()
    }

    fn is_complete(&self, ticket: SubmitTicket) -> bool {
        ticket.value <= self.inner.completed()
    }

    fn wait(&self, ticket: SubmitTicket) -> Result<()> {
        let semaphores = [self.inner.timeline];
        let values = [ticket.value];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        unsafe { self.inner.device.wait_semaphores(&info, u64::MAX) }.map_err(backend)
    }

    fn wait_idle(&self) -> Result<()> {
        unsafe { self.inner.device.device_wait_idle() }.map_err(backend)
    }

    fn read_timestamps(&self, ticket: SubmitTicket) -> Result<Vec<Timestamp>> {
        let inner = &self.inner;
        let mut st = inner.lock();
        inner.collect(&mut st);
        Ok(st.timestamps.iter().find(|(t, _)| *t == ticket.value).map(|(_, s)| s.clone()).unwrap_or_default())
    }
}
