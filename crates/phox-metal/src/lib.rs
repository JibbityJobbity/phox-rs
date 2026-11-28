//! The Metal 3 backend of the Phox RHI, for Apple Silicon Macs (macOS 14 or later). See `docs/metal.md`.
//!
//! Stage 1 of the plan: buffers, textures, SPIR-V shaders translated to MSL with SPIRV-Cross, graphics and
//! compute pipelines, rendering, copies, readback, a timeline (`MTLSharedEvent`) and a `CAMetalLayer`
//! swapchain in an `NSView`. Bindless textures, indirect draws, mesh shaders and timestamps are not reported
//! in [`phox_rhi::Capabilities`] yet, so work that needs them fails with the render graph's feature error.
//!
//! On other platforms the crate builds, and [`MetalDevice::new`] fails with `Unsupported`.

#[cfg(target_os = "macos")]
mod commands;
#[cfg(target_os = "macos")]
mod convert;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod shader;

#[cfg(target_os = "macos")]
pub use device::MetalDevice;

/// How to create a [`MetalDevice`].
#[derive(Clone, Debug, Default)]
pub struct MetalConfig {
    /// Record extra error information in every command buffer and fail [`phox_rhi::Device::wait`] with it.
    /// For Metal's API validation as well, run with `MTL_DEBUG_LAYER=1` in the environment.
    pub validation: bool,
}

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use phox_rhi::{Result, RhiError};

    /// Metal exists only on Apple platforms.
    pub struct MetalDevice(std::convert::Infallible);

    impl MetalDevice {
        pub fn new(_config: &super::MetalConfig) -> Result<Self> {
            Err(RhiError::Unsupported("Metal, which only exists on macOS".into()))
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use unsupported::MetalDevice;
