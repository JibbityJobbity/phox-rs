//! Widget tests without a GPU, in the style of Flutter's `WidgetTester`: build an app, find widgets by text,
//! key or role, act on them with real input events, and check the app, the layout and the draw list.
//!
//! ```
//! use phox_ui::prelude::*;
//! use phox_ui::testing::{find, Tester};
//!
//! #[derive(Default)]
//! struct Counter { count: i32 }
//! #[derive(Clone)]
//! enum Msg { Inc }
//!
//! impl App for Counter {
//!     type Message = Msg;
//!     fn update(&mut self, Msg::Inc: Msg, _: &mut UpdateCx<Msg>) { self.count += 1; }
//!     fn view(&self) -> Element<Msg> {
//!         Column::new().child(Text::new(format!("COUNT {}", self.count))).child(Button::new("+").on_press(Msg::Inc)).into()
//!     }
//! }
//!
//! let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
//! t.tap(find::text("+"));
//! assert_eq!(t.app().count, 1);
//! assert!(t.exists(find::text("COUNT 1")));
//! ```

use std::time::Duration;

use phox_immediate::{Color, DrawList, ImmediateRenderer};

use crate::element::Role;
use crate::event::{Event, KeyCode, Modifiers, Phase, PointerId, PointerKind};
use crate::geometry::{vec2, Rect, Size, Vec2};
use crate::runtime::{App, Runtime};
use crate::tree::{self, WidgetId};

/// What to look for in the tree.
#[derive(Clone, Debug)]
pub enum Finder {
    /// A widget whose label (a text's content, a button's caption) is exactly this.
    Text(String),
    /// A widget given this key.
    Key(String),
    Role(Role),
    /// Widgets matching both.
    And(Box<Finder>, Box<Finder>),
    /// The `n`th match (from 0, in tree order) of the inner finder.
    Nth(Box<Finder>, usize),
}

impl Finder {
    pub fn nth(self, n: usize) -> Finder {
        Finder::Nth(Box::new(self), n)
    }
    pub fn and(self, other: Finder) -> Finder {
        Finder::And(Box::new(self), Box::new(other))
    }
}

pub mod find {
    use super::Finder;
    use crate::element::Role;

    pub fn text(text: impl Into<String>) -> Finder {
        Finder::Text(text.into())
    }
    pub fn key(key: impl Into<String>) -> Finder {
        Finder::Key(key.into())
    }
    pub fn role(role: Role) -> Finder {
        Finder::Role(role)
    }
}

/// A widget a finder matched.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub id: WidgetId,
    pub role: Role,
    pub label: Option<String>,
    pub key: Option<String>,
    /// In window coordinates.
    pub rect: Rect,
}

fn matches(f: &Finder, found: &Found) -> bool {
    match f {
        Finder::Text(t) => found.label.as_deref() == Some(t.as_str()),
        Finder::Key(k) => found.key.as_deref() == Some(k.as_str()),
        Finder::Role(r) => found.role == *r,
        Finder::And(a, b) => matches(a, found) && matches(b, found),
        Finder::Nth(..) => unreachable!("Nth is resolved before matching"),
    }
}

fn select(f: &Finder, all: &[Found]) -> Vec<Found> {
    match f {
        Finder::Nth(inner, n) => select(inner, all).into_iter().nth(*n).into_iter().collect(),
        Finder::And(a, b) => {
            let left = select(a, all);
            let right = select(b, all);
            left.into_iter().filter(|x| right.iter().any(|y| y.id == x.id)).collect()
        }
        _ => all.iter().filter(|x| matches(f, x)).cloned().collect(),
    }
}

/// Runs an app against a fake clock and synthetic input. Every action delivers real events through the
/// runtime, so hit testing, capture and focus behave as they do in a window.
pub struct Tester<A: App> {
    runtime: Runtime<A>,
    list: DrawList,
    now: Duration,
    /// The fake system clipboard.
    clipboard: Option<String>,
}

impl<A: App> Tester<A> {
    /// The app at `size` logical pixels (scale 1).
    pub fn new(app: A, size: Size) -> Self {
        Self { runtime: Runtime::new(app, size, 1.0), list: DrawList::new(), now: Duration::ZERO, clipboard: None }
    }

    pub fn runtime(&mut self) -> &mut Runtime<A> {
        &mut self.runtime
    }

    pub fn app(&self) -> &A {
        self.runtime.app()
    }

    pub fn app_mut(&mut self) -> &mut A {
        self.runtime.app_mut()
    }

    /// Every widget the finder matches, in tree order.
    pub fn find(&mut self, finder: Finder) -> Vec<Found> {
        let (root, node) = self.runtime.root();
        let mut all = Vec::new();
        tree::visit(root, node, &mut |element, node| {
            all.push(Found {
                id: node.id,
                role: element.widget().role(),
                label: element.widget().label().map(str::to_string),
                key: element.key().map(|k| k.0.clone()),
                rect: node.rect,
            });
        });
        select(&finder, &all)
    }

    /// The one widget the finder matches; panics, listing what it found, if there is not exactly one.
    pub fn get(&mut self, finder: Finder) -> Found {
        let found = self.find(finder.clone());
        match found.len() {
            1 => found.into_iter().next().expect("one"),
            0 => panic!("no widget matches {finder:?}. The tree:\n{}", self.layout_snapshot()),
            n => panic!("{n} widgets match {finder:?}; narrow it with .nth(i) or a key: {found:#?}"),
        }
    }

    pub fn exists(&mut self, finder: Finder) -> bool {
        !self.find(finder).is_empty()
    }

    pub fn rect(&mut self, finder: Finder) -> Rect {
        self.get(finder).rect
    }

    pub fn event(&mut self, event: Event) {
        self.runtime.event(event);
    }

    /// Moves the mouse over the widget's centre.
    pub fn hover(&mut self, finder: Finder) {
        let pos = self.rect(finder).center();
        self.event(Event::mouse(Phase::Move, pos));
    }

    /// A mouse click at the widget's centre.
    pub fn tap(&mut self, finder: Finder) {
        let pos = self.rect(finder).center();
        self.tap_at(pos);
    }

    pub fn tap_at(&mut self, pos: Vec2) {
        self.event(Event::mouse(Phase::Move, pos));
        self.event(Event::mouse(Phase::Down, pos));
        self.event(Event::mouse(Phase::Up, pos));
    }

    /// A finger tapping the widget's centre.
    pub fn touch(&mut self, finder: Finder) {
        let pos = self.rect(finder).center();
        let touch = |phase| Event::Pointer { id: PointerId(1), kind: PointerKind::Touch, phase, pos };
        self.event(touch(Phase::Down));
        self.event(touch(Phase::Up));
    }

    /// A mouse drag from the widget's centre by `delta`, in four moves.
    pub fn drag(&mut self, finder: Finder, delta: Vec2) {
        let start = self.rect(finder).center();
        self.drag_from(start, delta);
    }

    pub fn drag_from(&mut self, start: Vec2, delta: Vec2) {
        self.event(Event::mouse(Phase::Move, start));
        self.event(Event::mouse(Phase::Down, start));
        for i in 1..=4 {
            let t = i as f32 / 4.0;
            self.event(Event::mouse(Phase::Move, vec2(start.x + delta.x * t, start.y + delta.y * t)));
        }
        self.event(Event::mouse(Phase::Up, vec2(start.x + delta.x, start.y + delta.y)));
    }

    /// Scrolls with the wheel over the widget's centre (positive y shows what is below).
    pub fn scroll(&mut self, finder: Finder, delta: Vec2) {
        let pos = self.rect(finder).center();
        self.event(Event::Scroll { pos, delta });
    }

    pub fn press_key(&mut self, key: KeyCode) {
        self.press_key_with(key, Modifiers::default());
    }

    pub fn press_key_with(&mut self, key: KeyCode, modifiers: Modifiers) {
        self.event(Event::Key { key, pressed: true, modifiers });
        self.event(Event::Key { key, pressed: false, modifiers });
    }

    /// Types text into whatever has focus.
    pub fn type_text(&mut self, text: &str) {
        self.event(Event::Text(text.to_string()));
    }

    /// Pastes `text` into whatever has focus, as the host does on Cmd+V (Ctrl+V).
    pub fn paste(&mut self, text: &str) {
        self.event(Event::Paste(text.to_string()));
    }

    /// What was last copied to the clipboard (Cmd+C or Cmd+X, sent as [`KeyCode::Copy`] or [`KeyCode::Cut`]).
    pub fn clipboard(&mut self) -> Option<String> {
        if let Some(c) = self.runtime.take_clipboard() {
            self.clipboard = Some(c);
        }
        self.clipboard.clone()
    }

    /// Moves the fake clock forward, firing due timers.
    pub fn advance(&mut self, by: Duration) {
        self.now += by;
        self.runtime.advance_to(self.now);
    }

    pub fn resize(&mut self, size: Size) {
        let scale = self.runtime.scale();
        self.runtime.set_size(size, scale);
    }

    /// Draws a frame and returns it.
    pub fn pump(&mut self) -> &DrawList {
        self.runtime.frame(&mut self.list);
        &self.list
    }

    pub fn needs_frame(&self) -> bool {
        self.runtime.needs_frame()
    }

    /// The laid-out tree as indented text, for comparing against a checked-in snapshot.
    pub fn layout_snapshot(&mut self) -> String {
        self.runtime.layout_snapshot()
    }

    /// Draws a frame on a real device and returns its RGBA pixels (row by row from the top), at the
    /// runtime's size times its scale, cleared to the theme's background.
    pub fn render(&mut self, renderer: &mut ImmediateRenderer) -> phox_immediate::Result<(u32, u32, Vec<u8>)> {
        self.runtime.frame(&mut self.list);
        let (size, scale) = (self.runtime.size(), self.runtime.scale());
        let (w, h) = ((size.w * scale).round() as u32, (size.h * scale).round() as u32);
        let clear: Color = self.runtime.theme().background;
        let pixels = renderer.render_image(&self.list, w, h, clear)?;
        Ok((w, h, pixels))
    }
}
