//! The persistent widget tree: one [`Node`] per widget, holding its state, identity and layout, matched to each
//! new view's widgets; and the passes over it (layout, placement, hit testing, event dispatch, painting).

use std::any::{Any, TypeId};
use std::fmt::Write as _;

use phox_immediate::DrawList;

use crate::element::{AnyElement, EventCx, Key, LayoutCx, PaintCx, Role, Shared};
use crate::event::Event;
use crate::geometry::{vec2, Constraints, Rect, Size, Vec2};
use crate::theme::Theme;

/// A widget's identity for as long as its node lives (across views).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WidgetId(pub(crate) u64);

pub(crate) struct Node {
    pub id: WidgetId,
    pub tag: TypeId,
    pub key: Option<Key>,
    pub state: Box<dyn Any>,
    pub children: Vec<Node>,
    /// From the parent's top-left corner.
    pub offset: Vec2,
    pub size: Size,
    /// In window coordinates, after `place`.
    pub rect: Rect,
    /// What the ancestors clip this node to, in window coordinates.
    pub clip: Option<Rect>,
}

pub(crate) struct Ids(u64);

impl Ids {
    pub fn new() -> Self {
        Ids(1)
    }
    fn next(&mut self) -> WidgetId {
        self.0 += 1;
        WidgetId(self.0)
    }
}

pub(crate) fn build(element: &AnyElement, ids: &mut Ids) -> Node {
    Node {
        id: ids.next(),
        tag: element.tag,
        key: element.key.clone(),
        state: element.widget.new_state(),
        children: element.widget.children().iter().map(|c| build(c, ids)).collect(),
        offset: Vec2::default(),
        size: Size::ZERO,
        rect: Rect::default(),
        clip: None,
    }
}

fn same_widget(node: &Node, element: &AnyElement) -> bool {
    node.tag == element.tag && node.key == element.key
}

/// Matches `root` to a new view: nodes whose type and key match keep their state; the rest are rebuilt.
pub(crate) fn reconcile_root(root: &mut Node, element: &AnyElement, ids: &mut Ids) {
    if same_widget(root, element) {
        reconcile(root, element, ids);
    } else {
        *root = build(element, ids);
    }
}

fn reconcile(node: &mut Node, element: &AnyElement, ids: &mut Ids) {
    let mut old: Vec<Option<Node>> = std::mem::take(&mut node.children).into_iter().map(Some).collect();
    for (i, child) in element.widget.children().iter().enumerate() {
        let found = match &child.key {
            Some(_) => old.iter().position(|n| n.as_ref().is_some_and(|n| same_widget(n, child))),
            None => old.get(i).and_then(|n| n.as_ref()).is_some_and(|n| same_widget(n, child)).then_some(i),
        };
        let next = match found.and_then(|j| old[j].take()) {
            Some(mut n) => {
                reconcile(&mut n, child, ids);
                n
            }
            None => build(child, ids),
        };
        node.children.push(next);
    }
}

/// Lays out the whole tree at `size`, then computes every node's window rectangle and clip.
pub(crate) fn layout(element: &AnyElement, node: &mut Node, size: Size, theme: &Theme) {
    let constraints = Constraints::tight(size);
    let mut cx = LayoutCx { node, children: element.widget.children(), theme };
    let laid = constraints.constrain(element.widget.layout(&mut cx, constraints));
    node.size = laid;
    node.offset = Vec2::default();
    place(element, node, Vec2::default(), None);
}

fn place(element: &AnyElement, node: &mut Node, origin: Vec2, clip: Option<Rect>) {
    node.rect = Rect::new(origin.x + node.offset.x, origin.y + node.offset.y, node.size.w, node.size.h);
    node.clip = clip;
    let child_clip = if element.widget.clips() { Some(clip.map_or(node.rect, |c| c.intersect(&node.rect))) } else { clip };
    let here = vec2(node.rect.x, node.rect.y);
    for (child, child_node) in element.widget.children().iter().zip(&mut node.children) {
        place(child, child_node, here, child_clip);
    }
}

/// The path (child indices from the root) to the deepest node under `pos`, the topmost where siblings
/// overlap; with `interactive_only`, only interactive widgets count.
pub(crate) fn hit_test(element: &AnyElement, node: &Node, pos: Vec2, interactive_only: bool) -> Option<Vec<usize>> {
    if node.clip.is_some_and(|c| !c.contains(pos)) {
        return None;
    }
    let children = element.widget.children();
    for i in (0..children.len()).rev() {
        if let Some(mut path) = hit_test(&children[i], &node.children[i], pos, interactive_only) {
            path.insert(0, i);
            return Some(path);
        }
    }
    (node.rect.contains(pos) && (!interactive_only || element.widget.interactive())).then(Vec::new)
}

pub(crate) fn path_to(node: &Node, id: WidgetId) -> Option<Vec<usize>> {
    if node.id == id {
        return Some(Vec::new());
    }
    node.children.iter().enumerate().find_map(|(i, c)| {
        path_to(c, id).map(|mut p| {
            p.insert(0, i);
            p
        })
    })
}

pub(crate) fn node_at<'a>(node: &'a Node, path: &[usize]) -> &'a Node {
    path.iter().fold(node, |n, &i| &n.children[i])
}

/// Delivers `event` to the node at the end of `path`, then lets it bubble up through the ancestors until a
/// widget handles it. Messages are mapped by every ancestor they pass.
pub(crate) fn dispatch(element: &AnyElement, node: &mut Node, path: &[usize], event: &Event, shared: &mut Shared) {
    let start = shared.messages.len();
    if let Some((&i, rest)) = path.split_first() {
        dispatch(&element.widget.children()[i], &mut node.children[i], rest, event, shared);
        for message in &mut shared.messages[start..] {
            let taken = std::mem::replace(message, Box::new(()));
            *message = element.widget.map_message(taken);
        }
    }
    if !shared.handled {
        let mut cx = EventCx { id: node.id, rect: node.rect, state: node.state.as_mut(), shared };
        element.widget.event(&mut cx, event);
    }
}

pub(crate) struct PaintState<'a> {
    pub theme: &'a Theme,
    pub hot: Option<crate::tree::WidgetId>,
    pub active: Option<crate::tree::WidgetId>,
    pub focused: Option<crate::tree::WidgetId>,
}

pub(crate) fn paint(element: &AnyElement, node: &Node, list: &mut DrawList, s: &PaintState) {
    let mut cx = PaintCx {
        list,
        rect: node.rect,
        state: node.state.as_ref(),
        theme: s.theme,
        hot: s.hot == Some(node.id),
        active: s.active == Some(node.id),
        focused: s.focused == Some(node.id),
    };
    element.widget.paint(&mut cx);
    let clips = element.widget.clips();
    if clips {
        list.push_clip(node.rect);
    }
    for (child, child_node) in element.widget.children().iter().zip(&node.children) {
        paint(child, child_node, list, s);
    }
    if clips {
        list.pop_clip();
    }
}

/// Every focusable widget, in tree order (the Tab order).
pub(crate) fn focusables(element: &AnyElement, node: &Node, out: &mut Vec<WidgetId>) {
    if element.widget.focusable() {
        out.push(node.id);
    }
    for (child, child_node) in element.widget.children().iter().zip(&node.children) {
        focusables(child, child_node, out);
    }
}

pub(crate) fn contains(node: &Node, id: WidgetId) -> bool {
    node.id == id || node.children.iter().any(|c| contains(c, id))
}

/// Calls `f` for every widget in tree order, with its node.
pub(crate) fn visit<'a>(element: &'a AnyElement, node: &'a Node, f: &mut dyn FnMut(&'a AnyElement, &'a Node)) {
    f(element, node);
    for (child, child_node) in element.widget.children().iter().zip(&node.children) {
        visit(child, child_node, f);
    }
}

/// The laid-out tree as indented text: role, label, key and rectangle of every widget with a role.
pub(crate) fn snapshot(element: &AnyElement, node: &Node) -> String {
    let mut out = String::new();
    snapshot_into(element, node, 0, &mut out);
    out
}

fn snapshot_into(element: &AnyElement, node: &Node, depth: usize, out: &mut String) {
    let role = element.widget.role();
    let child_depth = if role == Role::None {
        depth
    } else {
        let r = node.rect;
        let _ = write!(out, "{:indent$}{}", "", role.name(), indent = depth * 2);
        if let Some(label) = element.widget.label() {
            let _ = write!(out, " {label:?}");
        }
        if let Some(key) = &element.key {
            let _ = write!(out, " #{}", key.0);
        }
        let _ = writeln!(out, " @ {},{} {}x{}", fmt(r.x), fmt(r.y), fmt(r.w), fmt(r.h));
        depth + 1
    };
    for (child, child_node) in element.widget.children().iter().zip(&node.children) {
        snapshot_into(child, child_node, child_depth, out);
    }
}

/// A coordinate with at most two decimals and no trailing zeros, so snapshots are stable and readable.
fn fmt(v: f32) -> String {
    let s = format!("{:.2}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}
