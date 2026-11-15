//! The Vulkan 1.3 backend of the RHI: [`VulkanDevice`] implements [`phox_rhi::Device`] with dynamic
//! rendering, synchronization2, a timeline semaphore for submit tickets, buffer device addresses, a bindless
//! texture table, indirect draws (with an optional count buffer), task and mesh shaders where available
//! and GPU timestamps.

mod commands;
mod convert;
mod device;
mod instance;
mod physical;
mod pipeline;

pub use device::{DeviceInfo, VulkanConfig, VulkanDevice};
pub use instance::ValidationCounters;
