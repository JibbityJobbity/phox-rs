//! The Elm loop: input becomes events, widgets turn events into messages, `update` changes the app, `view`
//! describes it again, and frames happen only when something changed.
//!
//! The runtime never reads a clock or touches a GPU: the host passes time in with [`Runtime::advance_to`] and
//! takes frames out as [`DrawList`]s, so the same app runs in a window, inside a plugin host, or in a test.

use std::time::Duration;

use phox_immediate::DrawList;

use crate::element::{AnyElement, Element, Shared};
use crate::event::{Event, KeyCode, Phase, PointerKind};
use crate::geometry::{Size, Vec2};
use crate::theme::Theme;
use crate::tree::{self, Ids, Node, WidgetId};

/// An application: state, a way to change it, and a way to show it.
pub trait App: 'static {
    type Message: 'static;

    fn update(&mut self, message: Self::Message, cx: &mut UpdateCx<Self::Message>);
    fn view(&self) -> Element<Self::Message>;
    fn theme(&self) -> Theme {
        Theme::default()
    }
}

/// What `update` can ask of the runtime besides changing the app.
pub struct UpdateCx<M> {
    now: Duration,
    timers: Vec<(Duration, M)>,
    clipboard: Option<String>,
}

impl<M> UpdateCx<M> {
    /// The runtime's clock (see [`Runtime::advance_to`]).
    pub fn now(&self) -> Duration {
        self.now
    }

    /// Puts `text` on the clipboard, as a widget's copy does (the host collects it with
    /// [`Runtime::take_clipboard`]). For "copy" buttons.
    pub fn set_clipboard(&mut self, text: impl Into<String>) {
        self.clipboard = Some(text.into());
    }

    /// Sends `message` to `update` once the clock reaches `delay` from now.
    pub fn after(&mut self, delay: Duration, message: M) {
        self.timers.push((self.now + delay, message));
    }
}

pub struct Runtime<A: App> {
    app: A,
    root: AnyElement,
    tree: Node,
    ids: Ids,
    theme: Theme,
    size: Size,
    scale: f32,
    now: Duration,
    /// Due time, order of creation, message.
    timers: Vec<(Duration, u64, A::Message)>,
    timer_seq: u64,
    pointer: Option<Vec2>,
    hot: Option<WidgetId>,
    captured: Option<(crate::event::PointerId, WidgetId)>,
    focused: Option<WidgetId>,
    needs_view: bool,
    needs_layout: bool,
    needs_paint: bool,
    /// Text a widget copied, waiting for the host.
    clipboard: Option<String>,
}

impl<A: App> Runtime<A> {
    /// Builds the first view and lays it out at `size` logical pixels, shown at `scale` physical pixels per
    /// logical one.
    pub fn new(app: A, size: Size, scale: f32) -> Self {
        let root = app.view().into_any();
        let theme = app.theme();
        let mut ids = Ids::new();
        let tree = tree::build(&root, &mut ids);
        let mut runtime = Self {
            app,
            root,
            tree,
            ids,
            theme,
            size,
            scale,
            now: Duration::ZERO,
            timers: Vec::new(),
            timer_seq: 0,
            pointer: None,
            hot: None,
            captured: None,
            focused: None,
            needs_view: false,
            needs_layout: true,
            needs_paint: true,
            clipboard: None,
        };
        runtime.refresh();
        runtime
    }

    pub fn app(&self) -> &A {
        &self.app
    }

    /// Changes the app directly (outside `update`); the next frame shows the change.
    pub fn app_mut(&mut self) -> &mut A {
        self.needs_view = true;
        self.needs_paint = true;
        &mut self.app
    }

    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn now(&self) -> Duration {
        self.now
    }

    /// Something changed since the last [`Runtime::frame`]: the host should draw one.
    pub fn needs_frame(&self) -> bool {
        self.needs_paint || self.needs_view || self.needs_layout
    }

    /// When the earliest timer is due, for hosts that sleep until then.
    pub fn next_timer(&self) -> Option<Duration> {
        self.timers.iter().map(|t| t.0).min()
    }

    pub fn set_size(&mut self, size: Size, scale: f32) {
        if size != self.size || scale != self.scale {
            self.size = size;
            self.scale = scale;
            self.needs_layout = true;
            self.needs_paint = true;
        }
    }

    /// Moves the clock to `now` and delivers every timer due by then, in order. While a timer is delivered the
    /// clock reads its due time, so timers set from `update` stay on schedule however far the clock jumps.
    pub fn advance_to(&mut self, now: Duration) {
        let target = self.now.max(now);
        loop {
            let due = self.timers.iter().enumerate().filter(|(_, t)| t.0 <= target).min_by_key(|(_, t)| (t.0, t.1)).map(|(i, t)| (i, t.0));
            let Some((i, at)) = due else { break };
            let (_, _, message) = self.timers.remove(i);
            self.now = self.now.max(at);
            self.update(message);
        }
        self.now = target;
        self.refresh();
    }

    /// Delivers `message` to `update` from outside the view, as Elm's subscriptions do: data arriving from
    /// another thread, a plugin parameter moved by host automation. The view is brought up to date after.
    pub fn send(&mut self, message: A::Message) {
        self.update(message);
        self.refresh();
    }

    fn update(&mut self, message: A::Message) {
        let mut cx = UpdateCx { now: self.now, timers: Vec::new(), clipboard: None };
        self.app.update(message, &mut cx);
        if cx.clipboard.is_some() {
            self.clipboard = cx.clipboard;
        }
        for (due, message) in cx.timers {
            self.timer_seq += 1;
            self.timers.push((due, self.timer_seq, message));
        }
        self.needs_view = true;
        self.needs_paint = true;
    }

    /// Rebuilds the view and the layout if they are out of date.
    fn refresh(&mut self) {
        if self.needs_view {
            self.root = self.app.view().into_any();
            self.theme = self.app.theme();
            tree::reconcile_root(&mut self.tree, &self.root, &mut self.ids);
            self.needs_view = false;
            self.needs_layout = true;
        }
        if self.needs_layout {
            tree::layout(&self.root, &mut self.tree, self.size, &self.theme);
            self.needs_layout = false;
            self.needs_paint = true;
            // Widgets may have gone, or moved under the pointer.
            if self.captured.is_some_and(|(_, id)| !tree::contains(&self.tree, id)) {
                self.captured = None;
            }
            if self.focused.is_some_and(|id| !tree::contains(&self.tree, id)) {
                self.focused = None;
            }
            self.hot = self.hit(true);
        }
    }

    fn hit(&self, interactive_only: bool) -> Option<WidgetId> {
        let pos = self.pointer?;
        let path = tree::hit_test(&self.root, &self.tree, pos, interactive_only)?;
        Some(tree::node_at(&self.tree, &path).id)
    }

    /// Delivers one input event, runs `update` for every message it causes, and brings the view up to date.
    pub fn event(&mut self, event: Event) {
        self.refresh();
        let target = match &event {
            Event::Pointer { id, kind, phase, pos } => {
                let lifted = matches!(phase, Phase::Up | Phase::Cancel) && *kind != PointerKind::Mouse;
                self.pointer = Some(*pos);
                let hot = self.hit(true);
                if hot != self.hot {
                    self.hot = hot;
                    self.needs_paint = true;
                }
                if *phase == Phase::Down && self.focused.is_some() && self.focused != hot {
                    self.focused = None;
                    self.needs_paint = true;
                }
                let target = self.captured.filter(|(p, _)| p == id).map(|(_, w)| w).or(hot);
                if lifted {
                    // A finger that lifts stops hovering, after this event.
                    self.pointer = None;
                }
                target
            }
            Event::Scroll { pos, .. } => {
                self.pointer = Some(*pos);
                self.hit(false)
            }
            Event::Key { .. } | Event::Text(_) | Event::Paste(_) => self.focused,
            Event::WindowFocus(focused) => {
                if !focused {
                    self.captured = None;
                    self.needs_paint = true;
                }
                None
            }
        };

        let mut shared = Shared {
            messages: Vec::new(),
            hot: self.hot,
            captured: self.captured,
            focused: self.focused,
            theme: self.theme.clone(),
            handled: false,
            repaint: false,
            relayout: false,
            clipboard: None,
        };
        if let Some(path) = target.and_then(|t| tree::path_to(&self.tree, t)) {
            tree::dispatch(&self.root, &mut self.tree, &path, &event, &mut shared);
        }
        if let Event::Pointer { id, phase: Phase::Up | Phase::Cancel, .. } = &event {
            // A lifted pointer releases its capture even if the widget forgot to.
            if shared.captured.is_some_and(|(p, _)| p == *id) {
                shared.captured = None;
                shared.repaint = true;
            }
        }
        if !shared.handled
            && let Event::Key { key: KeyCode::Tab, pressed: true, modifiers } = &event {
                self.move_focus(!modifiers.shift);
                shared.focused = self.focused;
                shared.repaint = true;
            }
        if let Event::Pointer { .. } = &event
            && self.pointer.is_none() && self.hot.is_some() {
                self.hot = None;
                shared.repaint = true;
            }

        if shared.clipboard.is_some() {
            self.clipboard = shared.clipboard.take();
        }
        self.captured = shared.captured;
        self.focused = shared.focused;
        self.needs_paint |= shared.repaint;
        self.needs_layout |= shared.relayout;
        for message in shared.messages {
            match message.downcast::<A::Message>() {
                Ok(m) => self.update(*m),
                Err(_) => panic!("a widget emitted a message that is not the app's Message type; wrap the component in Element::map"),
            }
        }
        self.refresh();
    }

    fn move_focus(&mut self, forward: bool) {
        let mut order = Vec::new();
        tree::focusables(&self.root, &self.tree, &mut order);
        if order.is_empty() {
            self.focused = None;
            return;
        }
        let current = self.focused.and_then(|f| order.iter().position(|&id| id == f));
        let next = match (current, forward) {
            (None, true) => 0,
            (None, false) => order.len() - 1,
            (Some(i), true) => (i + 1) % order.len(),
            (Some(i), false) => (i + order.len() - 1) % order.len(),
        };
        self.focused = Some(order[next]);
    }

    /// Draws the current view into `list` (which it begins at the runtime's size and scale).
    pub fn frame(&mut self, list: &mut DrawList) {
        self.refresh();
        list.begin(self.size.w, self.size.h, self.scale);
        let state = tree::PaintState { theme: &self.theme, hot: self.hot, active: self.captured.map(|(_, id)| id), focused: self.focused };
        tree::paint(&self.root, &self.tree, list, &state);
        self.needs_paint = false;
    }

    /// The laid-out tree as text (see the `testing` module).
    pub fn layout_snapshot(&mut self) -> String {
        self.refresh();
        tree::snapshot(&self.root, &self.tree)
    }

    pub(crate) fn root(&mut self) -> (&AnyElement, &Node) {
        self.refresh();
        (&self.root, &self.tree)
    }

    /// Text a widget put on the clipboard since the last call. Hosts check after every event and copy it to
    /// the system clipboard.
    pub fn take_clipboard(&mut self) -> Option<String> {
        self.clipboard.take()
    }

    pub fn focused(&self) -> Option<WidgetId> {
        self.focused
    }
}
