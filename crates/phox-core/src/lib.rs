//! Phox core: scene, math, view extraction, the software rasterizer and the loaders. Nothing in this
//! crate touches a GPU.

pub mod camera;
pub mod color;
pub mod cpu_image;
pub mod error;
pub mod gltf_loader;
pub mod log_setup;
pub mod math;
pub mod mesh_loader;
pub mod meshlets;
mod meshopt;
pub mod render_view;
pub mod resources;
pub mod sample_scenes;
pub mod scene;
pub mod shader_interface;
pub mod shading;
pub mod software_rasterizer;
pub mod texture;

pub use error::{Error, Result};
