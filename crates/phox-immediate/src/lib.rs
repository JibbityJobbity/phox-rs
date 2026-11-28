//! Immediate-mode 2D drawing on Phox: build a [`DrawList`] every frame (anti-aliased rectangles, circles, arcs,
//! lines and stroke-font text) and draw it with an [`ImmediateRenderer`], which records one render-graph pass
//! per frame into a window's swapchain image or an offscreen image.
//!
//! ```no_run
//! # fn demo(device: std::sync::Arc<dyn phox_rhi::Device>) -> phox_immediate::Result<()> {
//! use phox_immediate::{Color, DrawList, ImmediateRenderer, Rect, vec2};
//! let mut renderer = ImmediateRenderer::new(device)?;
//! let mut list = DrawList::new();
//! list.begin(400.0, 300.0, 2.0); // logical size, physical pixels per logical pixel
//! list.rounded_rect(Rect::new(20.0, 20.0, 360.0, 260.0), 12.0, Color::hex(0xd8d8d0));
//! list.text(vec2(40.0, 40.0), 14.0, "HELLO", Color::hex(0x202020));
//! renderer.render_window(&list, 800, 600, Color::hex(0x101010))?;
//! # Ok(()) }
//! ```

pub mod draw;
pub mod error;
pub mod font;
pub mod renderer;

pub use draw::{text_width, vec2, Batch, Color, DrawList, Rect, Vec2, Vertex};
pub use error::{ImmediateError, Result};
pub use renderer::ImmediateRenderer;
