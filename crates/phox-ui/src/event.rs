//! Input, as the host delivers it: pointers (mouse, touch, pen), scrolling, keys and text.

use crate::geometry::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct PointerId(pub u64);

impl PointerId {
    /// The mouse (there is one).
    pub const MOUSE: PointerId = PointerId(0);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PointerKind {
    #[default]
    Mouse,
    Touch,
    Pen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// A button or finger went down.
    Down,
    Move,
    /// The button or finger came up.
    Up,
    /// The pointer went away without an `Up` (left the window, a touch was cancelled).
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    Enter,
    Space,
    Tab,
    Backspace,
    Delete,
    Escape,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    /// Editing commands. The host turns the platform's shortcuts into these (Cmd+A/C/X on macOS, Ctrl
    /// elsewhere), so widgets never need to know which modifier a platform uses.
    SelectAll,
    Copy,
    Cut,
    /// Any other key; text arrives separately as [`Event::Text`].
    Other,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    /// Command on macOS, the Windows key elsewhere.
    pub meta: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Pointer { id: PointerId, kind: PointerKind, phase: Phase, pos: Vec2 },
    /// A wheel or trackpad scroll at `pos`; positive `delta.y` scrolls the content up (shows what is below).
    Scroll { pos: Vec2, delta: Vec2 },
    /// A key went down (`pressed`) or up.
    Key { key: KeyCode, pressed: bool, modifiers: Modifiers },
    /// Text typed, after the platform's keyboard layout and input method.
    Text(String),
    /// Text pasted: the host reads the system clipboard on the platform's paste shortcut and sends it here, to
    /// the focused widget. (Copying goes the other way: see [`crate::EventCx::set_clipboard`].)
    Paste(String),
    /// The window gained or lost keyboard focus.
    WindowFocus(bool),
}

impl Event {
    pub fn mouse(phase: Phase, pos: Vec2) -> Self {
        Event::Pointer { id: PointerId::MOUSE, kind: PointerKind::Mouse, phase, pos }
    }

    pub fn key(key: KeyCode) -> Self {
        Event::Key { key, pressed: true, modifiers: Modifiers::default() }
    }
}
