//! The baseview window: a Phox Metal device presenting to it, a phox-ui runtime, and the input mapping.

use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use baseview::{Event, EventStatus, MouseButton, MouseEvent, ScrollDelta, Window, WindowEvent, WindowHandler, WindowOpenOptions, WindowScalePolicy};
use keyboard_types::{Key, KeyState, KeyboardEvent};
use phox_immediate::{DrawList, ImmediateRenderer};
use phox_metal::{MetalConfig, MetalDevice};
use phox_ui::{vec2, Event as UiEvent, KeyCode, Modifiers as UiModifiers, Phase, Runtime, Size, Vec2};
use raw_window_handle_05::HasRawWindowHandle;

use crate::{Command, HostedApp};

/// An open child view. Dropping it closes the view.
pub struct WindowHandle {
    window: baseview::WindowHandle,
}

// SAFETY: the handle is only used to close the window, which baseview does on the main thread that plugin
// hosts call from; nih-plug's own baseview editors do the same.
unsafe impl Send for WindowHandle {}

impl Drop for WindowHandle {
    fn drop(&mut self) {
        self.window.close();
    }
}

fn options(title: &str, (width, height): (f32, f32)) -> WindowOpenOptions {
    WindowOpenOptions { title: title.into(), size: baseview::Size::new(width as f64, height as f64), scale: WindowScalePolicy::SystemScaleFactor, gl_config: None }
}

/// Opens `size` logical pixels of view inside `parent` (for a plugin, the window handle the host gave it),
/// showing the app `build` makes on the window's thread.
pub fn open_parented<A, P, F>(parent: &P, title: &str, size: (f32, f32), build: F) -> WindowHandle
where
    A: HostedApp,
    P: HasRawWindowHandle,
    F: FnOnce() -> A + Send + 'static,
{
    let window = Window::open_parented(parent, options(title, size), move |window| HostWindow::new(window, build(), size));
    WindowHandle { window }
}

/// Opens a window of its own and runs until it closes.
pub fn run<A, F>(title: &str, size: (f32, f32), build: F)
where
    A: HostedApp,
    F: FnOnce() -> A + Send + 'static,
{
    Window::open_blocking(options(title, size), move |window| HostWindow::new(window, build(), size));
}

/// The view's state. Field order is drop order: the renderer before the device, the device (and its layer)
/// before baseview tears the view down.
struct HostWindow<A: HostedApp> {
    renderer: Option<ImmediateRenderer>,
    device: Option<MetalDevice>,
    runtime: Runtime<A>,
    list: DrawList,
    physical: (u32, u32),
    /// A full redraw is owed (the window was resized or just opened).
    dirty: bool,
    /// Where the mouse last was (button events carry no position).
    last_pointer: Vec2,
    opened: Instant,
}

impl<A: HostedApp> HostWindow<A> {
    fn new(window: &mut Window, app: A, (width, height): (f32, f32)) -> Self {
        let mut this = Self {
            renderer: None,
            device: None,
            runtime: Runtime::new(app, Size::new(width, height), 1.0),
            list: DrawList::new(),
            physical: (width as u32, height as u32),
            dirty: true,
            last_pointer: vec2(-1.0, -1.0),
            opened: Instant::now(),
        };
        match open_device(window) {
            Ok((device, renderer)) => {
                log::info!("phox-ui window drawing on {}", device.name());
                this.device = Some(device);
                this.renderer = Some(renderer);
            }
            Err(e) => log::error!("phox-ui window cannot draw: {e}"),
        }
        this
    }

    /// The platform's editing shortcuts (Cmd on macOS) for the focused widget, and the app's commands.
    /// Paste reads the system clipboard here; phox-ui widgets never touch it.
    fn shortcut(&mut self, k: &KeyboardEvent) -> Option<EventStatus> {
        use keyboard_types::Modifiers as M;
        let Key::Character(c) = &k.key else { return None };
        if !(k.modifiers.contains(M::META) || k.modifiers.contains(M::CONTROL)) {
            return None;
        }
        let c = c.to_lowercase();
        let command = match (c.as_str(), k.modifiers.contains(M::SHIFT)) {
            ("z", false) => Some(Command::Undo),
            ("z", true) | ("y", _) => Some(Command::Redo),
            _ => None,
        };
        if let Some(command) = command {
            return Some(if self.runtime.app_mut().command(command) { EventStatus::Captured } else { EventStatus::Ignored });
        }
        if self.runtime.focused().is_none() {
            // Not ours: the window owner's shortcuts (copying clips in a DAW, say) keep working.
            return Some(EventStatus::Ignored);
        }
        let key = match c.as_str() {
            "v" => {
                if let Some(text) = clipboard::get() {
                    self.runtime.event(UiEvent::Paste(text));
                }
                return Some(EventStatus::Captured);
            }
            "c" => KeyCode::Copy,
            "x" => KeyCode::Cut,
            "a" => KeyCode::SelectAll,
            _ => return Some(EventStatus::Ignored),
        };
        self.runtime.event(UiEvent::Key { key, pressed: true, modifiers: modifiers(&k.modifiers) });
        self.flush_clipboard();
        Some(EventStatus::Captured)
    }

    /// Hands anything a widget copied to the system clipboard.
    fn flush_clipboard(&mut self) {
        if let Some(text) = self.runtime.take_clipboard() {
            clipboard::set(text);
        }
    }
}

fn open_device(window: &mut Window) -> Result<(MetalDevice, ImmediateRenderer), String> {
    use raw_window_handle_05::RawWindowHandle as Raw05;
    let handle = match window.raw_window_handle() {
        Raw05::AppKit(h) => {
            let view = NonNull::new(h.ns_view).ok_or("the view has no NSView")?;
            raw_window_handle::RawWindowHandle::AppKit(raw_window_handle::AppKitWindowHandle::new(view))
        }
        other => return Err(format!("unsupported window system: {other:?}")),
    };
    // SAFETY: the device is dropped (HostWindow's field order) before baseview destroys the view, and baseview
    // calls this on the main thread.
    let device = unsafe { MetalDevice::new_for_view(&MetalConfig::default(), handle) }.map_err(|e| e.to_string())?;
    let renderer = ImmediateRenderer::new(Arc::new(device.clone())).map_err(|e| e.to_string())?;
    Ok((device, renderer))
}

fn modifiers(m: &keyboard_types::Modifiers) -> UiModifiers {
    use keyboard_types::Modifiers as M;
    UiModifiers { shift: m.contains(M::SHIFT), ctrl: m.contains(M::CONTROL), alt: m.contains(M::ALT), meta: m.contains(M::META) }
}

impl<A: HostedApp> WindowHandler for HostWindow<A> {
    fn on_frame(&mut self, _window: &mut Window) {
        self.runtime.advance_to(self.opened.elapsed());
        if let Some(message) = self.runtime.app().poll() {
            self.runtime.send(message);
            self.flush_clipboard();
        }
        // Something outside the messages moved (a host's automation, a preset): rebuild the view.
        if self.runtime.app().changed() {
            self.runtime.app_mut();
        }
        if !self.dirty && !self.runtime.needs_frame() {
            return;
        }
        let Some(renderer) = self.renderer.as_mut() else { return };
        self.runtime.frame(&mut self.list);
        let clear = self.runtime.theme().background;
        match renderer.render_window(&self.list, self.physical.0, self.physical.1, clear) {
            Ok(_) => self.dirty = false,
            Err(e) => {
                log::error!("phox-ui window stopped drawing: {e}");
                self.renderer = None;
            }
        }
    }

    fn on_event(&mut self, _window: &mut Window, event: Event) -> EventStatus {
        self.runtime.advance_to(self.opened.elapsed());
        let ui = match event {
            Event::Window(WindowEvent::Resized(info)) => {
                let size = info.physical_size();
                self.physical = (size.width, size.height);
                let logical = info.logical_size();
                self.runtime.set_size(Size::new(logical.width as f32, logical.height as f32), info.scale() as f32);
                self.dirty = true;
                return EventStatus::Captured;
            }
            Event::Mouse(MouseEvent::CursorMoved { position, .. }) => UiEvent::mouse(Phase::Move, vec2(position.x as f32, position.y as f32)),
            Event::Mouse(MouseEvent::ButtonPressed { button: MouseButton::Left, .. }) => UiEvent::mouse(Phase::Down, self.last_pointer),
            Event::Mouse(MouseEvent::ButtonReleased { button: MouseButton::Left, .. }) => UiEvent::mouse(Phase::Up, self.last_pointer),
            Event::Mouse(MouseEvent::CursorLeft) => UiEvent::mouse(Phase::Cancel, vec2(-1.0, -1.0)),
            Event::Mouse(MouseEvent::WheelScrolled { delta, .. }) => {
                let (x, y) = match delta {
                    ScrollDelta::Lines { x, y } => (x * 24.0, y * 24.0),
                    ScrollDelta::Pixels { x, y } => (x, y),
                };
                UiEvent::Scroll { pos: self.last_pointer, delta: vec2(-x, -y) }
            }
            Event::Keyboard(k) if k.state == KeyState::Down => {
                if let Some(status) = self.shortcut(&k) {
                    return status;
                }
                let key = match &k.key {
                    Key::Enter => KeyCode::Enter,
                    Key::Tab => KeyCode::Tab,
                    Key::Backspace => KeyCode::Backspace,
                    Key::Delete => KeyCode::Delete,
                    Key::Escape => KeyCode::Escape,
                    Key::ArrowLeft => KeyCode::Left,
                    Key::ArrowRight => KeyCode::Right,
                    Key::ArrowUp => KeyCode::Up,
                    Key::ArrowDown => KeyCode::Down,
                    Key::Home => KeyCode::Home,
                    Key::End => KeyCode::End,
                    Key::Character(c) if c == " " => KeyCode::Space,
                    Key::Character(c) => {
                        // Typed text only matters to a focused text field; let the window's owner have keys
                        // otherwise (a DAW's keyboard shortcuts, its computer keyboard MIDI).
                        if self.runtime.focused().is_none() {
                            return EventStatus::Ignored;
                        }
                        self.runtime.event(UiEvent::Text(c.clone()));
                        return EventStatus::Captured;
                    }
                    _ => KeyCode::Other,
                };
                if self.runtime.focused().is_none() {
                    return EventStatus::Ignored;
                }
                UiEvent::Key { key, pressed: true, modifiers: modifiers(&k.modifiers) }
            }
            _ => return EventStatus::Ignored,
        };
        if let UiEvent::Pointer { pos, .. } = &ui
            && pos.x >= 0.0
        {
            self.last_pointer = *pos;
        }
        self.runtime.event(ui);
        self.flush_clipboard();
        EventStatus::Captured
    }
}

mod clipboard {
    pub fn get() -> Option<String> {
        arboard::Clipboard::new().ok()?.get_text().ok()
    }

    pub fn set(text: String) {
        if let Ok(mut c) = arboard::Clipboard::new() {
            let _ = c.set_text(text);
        }
    }
}
