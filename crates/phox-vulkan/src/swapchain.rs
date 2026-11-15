//! The window's swapchain: images, acquire and present semaphores, and recreation when the window changes.

use crate::convert;
use crate::device::{Inner, TextureRes, backend};
use ash::vk;
use phox_rhi::*;

pub(crate) struct SwapchainState {
    swapchain: vk::SwapchainKHR,
    format: Format,
    extent: vk::Extent2D,
    images: Vec<TextureHandle>,
    /// Signalled when an image is acquired; used round-robin, one per image.
    acquire: Vec<vk::Semaphore>,
    next_acquire: usize,
    /// The semaphore the next submission waits on (set by acquire).
    pending: Option<vk::Semaphore>,
    /// Signalled when rendering to an image is finished; presents wait on it. One per image.
    finished: Vec<vk::Semaphore>,
    out_of_date: bool,
}

/// The surface format Phox draws to, by preference.
fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> Option<(vk::SurfaceFormatKHR, Format)> {
    [
        (vk::Format::B8G8R8A8_SRGB, Format::BGRA8Srgb),
        (vk::Format::B8G8R8A8_UNORM, Format::BGRA8Unorm),
        (vk::Format::R8G8B8A8_SRGB, Format::RGBA8Srgb),
        (vk::Format::R8G8B8A8_UNORM, Format::RGBA8Unorm),
    ]
    .into_iter()
    .find_map(|(vk_format, format)| formats.iter().find(|f| f.format == vk_format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR).map(|f| (*f, format)))
}

impl Inner {
    /// Frees the swapchain and its images' views and semaphores. The device must be idle.
    pub(crate) fn destroy_swapchain(&self, st: &mut crate::device::State) {
        let Some(old) = st.swapchain.take() else { return };
        let Some(loader) = &self.swapchain_loader else { return };
        unsafe {
            for handle in old.images {
                if let Some(t) = st.textures.destroy(handle) {
                    self.device.destroy_image_view(t.view, None);
                }
            }
            for s in old.acquire.into_iter().chain(old.finished) {
                self.device.destroy_semaphore(s, None);
            }
            loader.destroy_swapchain(old.swapchain, None);
        }
    }

    fn create_swapchain(&self, st: &mut crate::device::State, width: u32, height: u32) -> Result<()> {
        let (surface_loader, surface) = self.instance.surface.as_ref().ok_or_else(|| RhiError::Unsupported("presenting to a window".into()))?;
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| RhiError::Unsupported("presenting to a window".into()))?;
        let physical = self.physical;
        let caps = unsafe { surface_loader.get_physical_device_surface_capabilities(physical, *surface) }.map_err(backend)?;
        let formats = unsafe { surface_loader.get_physical_device_surface_formats(physical, *surface) }.map_err(backend)?;
        let (surface_format, format) = choose_format(&formats).ok_or_else(|| RhiError::Unsupported("a BGRA8/RGBA8 surface format".into()))?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let mut count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            count = count.min(caps.max_image_count);
        }
        let old = st.swapchain.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.swapchain);
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let swapchain = unsafe { loader.create_swapchain(&info, None) }.map_err(|e| backend(format!("failed to create the swapchain: {e}")))?;
        // The old swapchain is retired now that its replacement exists; its images are freed with it.
        self.destroy_swapchain(st);

        let images = unsafe { loader.get_swapchain_images(swapchain) }.map_err(backend)?;
        let mut handles = Vec::with_capacity(images.len());
        for (i, &image) in images.iter().enumerate() {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: convert::aspect(format), base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            let view = unsafe { self.device.create_image_view(&view_info, None) }.map_err(backend)?;
            handles.push(st.textures.create(TextureRes {
                image,
                view,
                mip_views: Vec::new(),
                allocation: None,
                external: true,
                desc: TextureDesc { width: extent.width, height: extent.height, mips: 1, format, usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_DST, debug_name: format!("swapchain image {i}") },
            }));
        }
        let semaphore = || unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.map_err(backend);
        let acquire = (0..images.len()).map(|_| semaphore()).collect::<Result<Vec<_>>>()?;
        let finished = (0..images.len()).map(|_| semaphore()).collect::<Result<Vec<_>>>()?;
        st.swapchain = Some(SwapchainState { swapchain, format, extent, images: handles, acquire, next_acquire: 0, pending: None, finished, out_of_date: false });
        Ok(())
    }

    pub(crate) fn acquire_backbuffer(&self, width: u32, height: u32) -> Result<Option<Backbuffer>> {
        if width == 0 || height == 0 {
            return Ok(None); // minimized
        }
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| RhiError::Unsupported("presenting to a window".into()))?;
        let (surface_loader, surface) = self.instance.surface.as_ref().ok_or_else(|| RhiError::Unsupported("presenting to a window".into()))?;
        let mut st = self.lock();
        self.collect(&mut st);
        for _attempt in 0..2 {
            // A swapchain made for another size, or flagged out of date, is replaced (the window changed).
            let caps = unsafe { surface_loader.get_physical_device_surface_capabilities(self.physical, *surface) }.map_err(backend)?;
            let wanted = if caps.current_extent.width != u32::MAX { caps.current_extent } else { vk::Extent2D { width, height } };
            if wanted.width == 0 || wanted.height == 0 {
                return Ok(None);
            }
            let stale = st.swapchain.as_ref().is_none_or(|s| s.out_of_date || s.extent != wanted);
            if stale {
                // Nothing may still use the old images.
                unsafe { self.device.device_wait_idle() }.map_err(backend)?;
                self.create_swapchain(&mut st, width, height)?;
            }
            let sc = st.swapchain.as_mut().expect("created above");
            let semaphore = sc.acquire[sc.next_acquire];
            sc.next_acquire = (sc.next_acquire + 1) % sc.acquire.len();
            match unsafe { loader.acquire_next_image(sc.swapchain, u64::MAX, semaphore, vk::Fence::null()) } {
                Ok((index, suboptimal)) => {
                    sc.out_of_date |= suboptimal;
                    sc.pending = Some(semaphore);
                    return Ok(Some(Backbuffer { texture: sc.images[index as usize], format: sc.format, width: sc.extent.width, height: sc.extent.height, index }));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => sc.out_of_date = true,
                Err(e) => return Err(backend(format!("failed to acquire a swapchain image: {e}"))),
            }
        }
        Err(RhiError::Backend("the swapchain stayed out of date".into()))
    }

    /// The semaphores a presenting submission waits on and signals.
    pub(crate) fn present_semaphores(&self, index: u32) -> Result<(vk::Semaphore, vk::Semaphore)> {
        let mut st = self.lock();
        let sc = st.swapchain.as_mut().ok_or_else(|| RhiError::Misuse("no swapchain image was acquired".into()))?;
        let acquired = sc.pending.take().ok_or_else(|| RhiError::Misuse("no swapchain image was acquired".into()))?;
        Ok((acquired, sc.finished[index as usize]))
    }

    pub(crate) fn present(&self, index: u32, wait: vk::Semaphore) -> Result<()> {
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| RhiError::Unsupported("presenting to a window".into()))?;
        let mut st = self.lock();
        let Some(sc) = st.swapchain.as_mut() else { return Ok(()) };
        let swapchains = [sc.swapchain];
        let indices = [index];
        let waits = [wait];
        let info = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices);
        match unsafe { loader.queue_present(self.queue, &info) } {
            Ok(suboptimal) => {
                sc.out_of_date |= suboptimal;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                sc.out_of_date = true;
                Ok(())
            }
            Err(e) => Err(backend(format!("present failed: {e}"))),
        }
    }
}
