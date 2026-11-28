//! A declarative UI framework on Phox: Elm-style apps (state, `update`, `view`) built from Flutter-style
//! widgets with builder calls, laid out with box constraints and drawn with `phox-immediate`.
//!
//! Nothing here touches a GPU or a clock. A host turns platform input into [`Event`]s, passes time in, and
//! draws the [`phox_immediate::DrawList`] each [`Runtime::frame`] fills, only when [`Runtime::needs_frame`]
//! says something changed. The [`testing`] module drives the same runtime with synthetic input.
//!
//! See `docs/ui.md` for the design.

pub mod element;
pub mod event;
pub mod geometry;
pub mod runtime;
pub mod testing;
pub mod theme;
mod tree;
pub mod widgets;

pub use element::{AnyElement, Element, EventCx, Key, LayoutCx, PaintCx, Role, Widget};
pub use event::{Event, KeyCode, Modifiers, Phase, PointerId, PointerKind};
pub use geometry::{vec2, Alignment, Constraints, Insets, Rect, Size, Vec2};
pub use phox_immediate::Color;
pub use runtime::{App, Runtime, UpdateCx};
pub use theme::Theme;
pub use tree::WidgetId;

/// Everything an app usually needs.
pub mod prelude {
    pub use crate::element::{Element, Widget};
    pub use crate::geometry::{vec2, Alignment, Insets, Rect, Size, Vec2};
    pub use crate::runtime::{App, UpdateCx};
    pub use crate::theme::Theme;
    pub use crate::widgets::*;
    pub use phox_immediate::Color;
}
