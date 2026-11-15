//! The render hardware interface: typed handles, resource and pipeline descriptions, capabilities and the
//! [`Device`] and [`CommandList`] traits every backend implements (Vulkan, the null backend, later Metal).

mod bitflags;
pub mod capabilities;
pub mod commands;
pub mod descs;
pub mod device;
pub mod error;
pub mod formats;
pub mod handles;
pub mod pool;

pub use capabilities::{Capabilities, Feature, FeatureSet};
pub use commands::*;
pub use descs::*;
pub use device::{Backbuffer, Device, SubmitTicket, Timestamp};
pub use error::{Result, RhiError};
pub use formats::*;
pub use handles::*;
