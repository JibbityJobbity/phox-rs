//! Shows a phox-ui [`App`] in a native window: as a child view inside a window someone else owns (an audio
//! plugin's editor inside a DAW: [`open_parented`]) or in a window of its own ([`run`]). The view is a
//! baseview window drawn with Phox's Metal backend, so this is macOS only for now; on other platforms the
//! crate has only the traits.
//!
//! The host turns the platform's input into phox-ui events, keeps the runtime's clock (for timers), hands
//! copy and paste between widgets and the system clipboard, and draws only when something changed.
//!
//! [`App`]: phox_ui::App

use phox_ui::App;

/// An app the host can show. Everything here has a default, so a plain [`App`] only needs an empty impl.
pub trait HostedApp: App + 'static {
    /// Called once per display frame for news from outside the view (a parameter the host moved, meters, a
    /// worker thread), as a message for `update`; `None` when there is none.
    fn poll(&self) -> Option<Self::Message> {
        None
    }

    /// Called once per display frame: whether something outside the app's messages changed what it shows,
    /// so the view must be rebuilt.
    fn changed(&self) -> bool {
        false
    }

    /// A window-wide shortcut, whatever has focus. Returns whether the app used it (if not, the key goes on
    /// to the window's owner, so a DAW's own undo keeps working).
    fn command(&mut self, _command: Command) -> bool {
        false
    }
}

/// Shortcuts the host turns into app commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// Cmd+Z (Ctrl+Z elsewhere).
    Undo,
    /// Cmd+Shift+Z or Cmd+Y.
    Redo,
}

#[cfg(target_os = "macos")]
mod window;

#[cfg(target_os = "macos")]
pub use window::{open_parented, run, WindowHandle};
