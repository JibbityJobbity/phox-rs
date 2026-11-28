//! Widgets and elements.
//!
//! A [`Widget`] is an immutable description that `view` builds fresh every time: configuration and children,
//! never state that must survive a rebuild. That state (hover, a drag in progress, a scroll offset) lives in
//! the framework's tree, created by [`Widget::new_state`] and matched to the new widgets by position and
//! [`Key`] after every view, like Flutter's elements.
//!
//! Widgets are not generic over the app's message type: they emit messages type-erased, and the typed
//! [`Element<M>`] wrapper keeps the builders type-safe. That is what lets [`Element::map`] turn a component's
//! messages into its parent's.

use std::any::{Any, TypeId};
use std::marker::PhantomData;

use phox_immediate::DrawList;

use crate::event::{Event, PointerId};
use crate::geometry::{Constraints, Rect, Size, Vec2};
use crate::theme::Theme;
use crate::tree::{Node, WidgetId};

/// What a widget is, for tests (finders, layout snapshots) and, later, accessibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Not shown in layout snapshots; its children are listed in its place.
    None,
    Group,
    Container,
    Row,
    Column,
    Stack,
    Text,
    Button,
    Toggle,
    Slider,
    Knob,
    TextField,
    Scroll,
    Custom(&'static str),
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::None => "None",
            Role::Group => "Group",
            Role::Container => "Container",
            Role::Row => "Row",
            Role::Column => "Column",
            Role::Stack => "Stack",
            Role::Text => "Text",
            Role::Button => "Button",
            Role::Toggle => "Toggle",
            Role::Slider => "Slider",
            Role::Knob => "Knob",
            Role::TextField => "TextField",
            Role::Scroll => "Scroll",
            Role::Custom(name) => name,
        }
    }
}

/// A stable identity for a widget among its siblings, so its state follows it when the list around it
/// changes. Unkeyed widgets are matched by position.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Key(pub String);

impl<S: Into<String>> From<S> for Key {
    fn from(s: S) -> Self {
        Key(s.into())
    }
}

pub trait Widget: 'static {
    fn role(&self) -> Role {
        Role::Group
    }
    /// The text a test finds this widget by (a button's label, a text's content).
    fn label(&self) -> Option<&str> {
        None
    }
    fn children(&self) -> &[AnyElement] {
        &[]
    }
    /// The share of a `Row` or `Column`'s free space this widget takes; 0 for its own size.
    fn flex(&self) -> u32 {
        0
    }
    /// Receives pointer events: the topmost interactive widget under a pointer is the one it is over.
    fn interactive(&self) -> bool {
        false
    }
    /// Can take keyboard focus (and is visited by Tab).
    fn focusable(&self) -> bool {
        false
    }
    /// Clips its children to its own rectangle.
    fn clips(&self) -> bool {
        false
    }
    /// The internal state this widget keeps between views.
    fn new_state(&self) -> Box<dyn Any> {
        Box::new(())
    }
    /// Lays out the children (through `cx`) and returns this widget's size, within `constraints`.
    fn layout(&self, cx: &mut LayoutCx, constraints: Constraints) -> Size;
    /// Handles an event delivered to this widget: one it is the target of, or one bubbling up from a child
    /// that did not handle it.
    fn event(&self, _cx: &mut EventCx, _event: &Event) {}
    /// Draws this widget, under its children.
    fn paint(&self, _cx: &mut PaintCx) {}
    /// Converts a message from a descendant on its way up (see [`Element::map`]).
    fn map_message(&self, message: Box<dyn Any>) -> Box<dyn Any> {
        message
    }
}

/// A widget with its type and key, as the tree stores children.
pub struct AnyElement {
    pub(crate) widget: Box<dyn Widget>,
    pub(crate) tag: TypeId,
    pub(crate) key: Option<Key>,
}

impl AnyElement {
    pub fn widget(&self) -> &dyn Widget {
        self.widget.as_ref()
    }
    pub fn key(&self) -> Option<&Key> {
        self.key.as_ref()
    }
}

/// A widget that emits messages of type `M`. Every builder converts into one with `.into()`.
pub struct Element<M> {
    inner: AnyElement,
    _message: PhantomData<fn() -> M>,
}

impl<M: 'static> Element<M> {
    pub fn new<W: Widget>(widget: W) -> Self {
        Self { inner: AnyElement { widget: Box::new(widget), tag: TypeId::of::<W>(), key: None }, _message: PhantomData }
    }

    /// Gives the widget a [`Key`].
    pub fn key(mut self, key: impl Into<Key>) -> Self {
        self.inner.key = Some(key.into());
        self
    }

    /// Turns this element's messages into `N`s, so a component written against its own message type can
    /// sit inside an app with another.
    pub fn map<N: 'static>(self, f: impl Fn(M) -> N + 'static) -> Element<N> {
        let key = self.inner.key.clone();
        let convert = move |message: Box<dyn Any>| -> Box<dyn Any> {
            match message.downcast::<M>() {
                Ok(m) => Box::new(f(*m)),
                Err(other) => other,
            }
        };
        let map = Map { child: [self.inner], convert: Box::new(convert) };
        let mut out = Element::new(map);
        out.inner.key = key;
        out
    }

    pub fn into_any(self) -> AnyElement {
        self.inner
    }
}

/// Converts a type-erased message (see [`Element::map`]).
type Convert = Box<dyn Fn(Box<dyn Any>) -> Box<dyn Any>>;

/// See [`Element::map`].
struct Map {
    child: [AnyElement; 1],
    convert: Convert,
}

impl Widget for Map {
    fn role(&self) -> Role {
        Role::None
    }
    fn children(&self) -> &[AnyElement] {
        &self.child
    }
    fn flex(&self) -> u32 {
        self.child[0].widget.flex()
    }
    fn layout(&self, cx: &mut LayoutCx, constraints: Constraints) -> Size {
        let size = cx.layout_child(0, constraints);
        cx.place_child(0, Vec2::default());
        size
    }
    fn map_message(&self, message: Box<dyn Any>) -> Box<dyn Any> {
        (self.convert)(message)
    }
}

/// Lets a widget lay out its children. Children are laid out by index, in [`Widget::children`] order.
pub struct LayoutCx<'a> {
    pub(crate) node: &'a mut Node,
    pub(crate) children: &'a [AnyElement],
    pub(crate) theme: &'a Theme,
}

impl LayoutCx<'_> {
    pub fn theme(&self) -> &Theme {
        self.theme
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// The flex factor of child `i` (see [`Widget::flex`]).
    pub fn child_flex(&self, i: usize) -> u32 {
        self.children[i].widget.flex()
    }

    /// Lays out child `i` within `constraints` and returns its size. Call [`LayoutCx::place_child`] after.
    pub fn layout_child(&mut self, i: usize, constraints: Constraints) -> Size {
        let element = &self.children[i];
        let node = &mut self.node.children[i];
        let mut cx = LayoutCx { node, children: element.widget.children(), theme: self.theme };
        let size = constraints.constrain(element.widget.layout(&mut cx, constraints));
        cx.node.size = size;
        size
    }

    /// Puts child `i`'s top-left corner at `offset` from this widget's.
    pub fn place_child(&mut self, i: usize, offset: Vec2) {
        self.node.children[i].offset = offset;
    }

    /// This widget's internal state.
    pub fn state<T: 'static>(&self) -> &T {
        self.node.state.downcast_ref().expect("the widget's state has the type its new_state made")
    }
}

/// What the event dispatcher shares with every widget it calls.
pub(crate) struct Shared {
    pub messages: Vec<Box<dyn Any>>,
    pub hot: Option<WidgetId>,
    pub captured: Option<(PointerId, WidgetId)>,
    pub focused: Option<WidgetId>,
    pub theme: Theme,
    pub handled: bool,
    pub repaint: bool,
    pub relayout: bool,
    pub clipboard: Option<String>,
}

/// Lets a widget react to an event: emit messages, capture the pointer, take focus, change its state.
pub struct EventCx<'a> {
    pub(crate) id: WidgetId,
    pub(crate) rect: Rect,
    pub(crate) state: &'a mut dyn Any,
    pub(crate) shared: &'a mut Shared,
}

impl EventCx<'_> {
    /// This widget's rectangle, in window coordinates.
    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn theme(&self) -> &Theme {
        &self.shared.theme
    }

    /// This widget's internal state. Changing it does not repaint by itself: call
    /// [`EventCx::request_repaint`] (or `request_layout`).
    pub fn state<T: 'static>(&mut self) -> &mut T {
        self.state.downcast_mut().expect("the widget's state has the type its new_state made")
    }

    /// Sends `message` to the app's `update` (through any [`Element::map`] on the way up).
    pub fn emit<M: 'static>(&mut self, message: M) {
        self.shared.messages.push(Box::new(message));
    }

    /// The pointer is over this widget (and it is the topmost interactive one there).
    pub fn is_hot(&self) -> bool {
        self.shared.hot == Some(self.id)
    }

    /// This widget holds a pointer capture.
    pub fn has_capture(&self) -> bool {
        self.shared.captured.is_some_and(|(_, id)| id == self.id)
    }

    /// Sends `pointer`'s events to this widget until it is released or lifted, wherever the pointer goes.
    pub fn capture(&mut self, pointer: PointerId) {
        self.shared.captured = Some((pointer, self.id));
        self.shared.repaint = true;
    }

    pub fn release(&mut self) {
        if self.has_capture() {
            self.shared.captured = None;
            self.shared.repaint = true;
        }
    }

    pub fn is_focused(&self) -> bool {
        self.shared.focused == Some(self.id)
    }

    pub fn request_focus(&mut self) {
        if self.shared.focused != Some(self.id) {
            self.shared.focused = Some(self.id);
            self.shared.repaint = true;
        }
    }

    pub fn blur(&mut self) {
        if self.is_focused() {
            self.shared.focused = None;
            self.shared.repaint = true;
        }
    }

    /// Puts `text` on the clipboard (the host collects it with [`crate::Runtime::take_clipboard`] and hands
    /// it to the system).
    pub fn set_clipboard(&mut self, text: impl Into<String>) {
        self.shared.clipboard = Some(text.into());
    }

    /// Stops the event from bubbling to this widget's ancestors.
    pub fn set_handled(&mut self) {
        self.shared.handled = true;
    }

    pub fn request_repaint(&mut self) {
        self.shared.repaint = true;
    }

    /// The state changed in a way that changes sizes or positions (a scroll offset).
    pub fn request_layout(&mut self) {
        self.shared.relayout = true;
    }
}

/// Lets a widget draw itself.
pub struct PaintCx<'a> {
    pub(crate) list: &'a mut DrawList,
    pub(crate) rect: Rect,
    pub(crate) state: &'a dyn Any,
    pub(crate) theme: &'a Theme,
    pub(crate) hot: bool,
    pub(crate) active: bool,
    pub(crate) focused: bool,
}

impl PaintCx<'_> {
    pub fn list(&mut self) -> &mut DrawList {
        self.list
    }
    /// This widget's rectangle, in window coordinates.
    pub fn rect(&self) -> Rect {
        self.rect
    }
    pub fn theme(&self) -> &Theme {
        self.theme
    }
    pub fn state<T: 'static>(&self) -> &T {
        self.state.downcast_ref().expect("the widget's state has the type its new_state made")
    }
    /// The pointer is over this widget.
    pub fn is_hot(&self) -> bool {
        self.hot
    }
    /// This widget holds a pointer capture (it is being pressed or dragged).
    pub fn is_active(&self) -> bool {
        self.active
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
}
