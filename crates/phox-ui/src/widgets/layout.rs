//! Single-child and stacking layout widgets: Container (and Padding), SizedBox, Align, Expanded, Stack.

use std::marker::PhantomData;

use phox_immediate::Color;

use crate::element::{AnyElement, Element, LayoutCx, PaintCx, Role, Widget};
use crate::geometry::{vec2, Alignment, Constraints, Insets, Size, Vec2};

macro_rules! into_element {
    ($($ty:ident),*) => {$(
        impl<M: 'static> From<$ty<M>> for Element<M> {
            fn from(w: $ty<M>) -> Self {
                Element::new(w)
            }
        }
    )*};
}
pub(crate) use into_element;

fn one<M: 'static>(child: impl Into<Element<M>>) -> Vec<AnyElement> {
    vec![child.into().into_any()]
}

/// A box with optional padding, background, border, rounded corners, fixed size and child alignment.
pub struct Container<M> {
    child: Vec<AnyElement>,
    padding: Insets,
    background: Option<Color>,
    border: Option<(f32, Color)>,
    radius: f32,
    width: Option<f32>,
    height: Option<f32>,
    align: Option<Alignment>,
    _m: PhantomData<fn() -> M>,
}

impl<M: 'static> Default for Container<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: 'static> Container<M> {
    pub fn new() -> Self {
        Self { child: Vec::new(), padding: Insets::default(), background: None, border: None, radius: 0.0, width: None, height: None, align: None, _m: PhantomData }
    }
    pub fn child(mut self, child: impl Into<Element<M>>) -> Self {
        self.child = one(child);
        self
    }
    pub fn padding(mut self, padding: impl Into<Insets>) -> Self {
        self.padding = padding.into();
        self
    }
    pub fn background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }
    pub fn border(mut self, width: f32, color: Color) -> Self {
        self.border = Some((width, color));
        self
    }
    pub fn radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }
    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height);
        self
    }
    /// Places the child inside the box (which then fills the space it is given, like Flutter's).
    pub fn align(mut self, align: Alignment) -> Self {
        self.align = Some(align);
        self
    }
}

/// Fixes the axes that are set (within `c`), leaving the others as they are.
fn fix(c: Constraints, width: Option<f32>, height: Option<f32>) -> Constraints {
    let mut out = c;
    if let Some(w) = width {
        let w = w.clamp(c.min.w, c.max.w.max(c.min.w));
        out.min.w = w;
        out.max.w = w;
    }
    if let Some(h) = height {
        let h = h.clamp(c.min.h, c.max.h.max(c.min.h));
        out.min.h = h;
        out.max.h = h;
    }
    out
}

impl<M: 'static> Widget for Container<M> {
    fn role(&self) -> Role {
        Role::Container
    }
    fn children(&self) -> &[AnyElement] {
        &self.child
    }
    fn layout(&self, cx: &mut LayoutCx, constraints: Constraints) -> Size {
        let c = fix(constraints, self.width, self.height);
        let p = self.padding;
        let (pw, ph) = (p.left + p.right, p.top + p.bottom);
        if self.child.is_empty() {
            return c.constrain(Size::new(self.width.unwrap_or(pw), self.height.unwrap_or(ph)));
        }
        let inner = if self.align.is_some() { c.deflate(p).loosen() } else { c.deflate(p) };
        let child = cx.layout_child(0, inner);
        let size = match self.align {
            Some(_) => {
                let fill = |bounded: bool, max: f32, content: f32| if bounded { max } else { content };
                Size::new(fill(c.has_bounded_width(), c.max.w, child.w + pw), fill(c.has_bounded_height(), c.max.h, child.h + ph))
            }
            None => Size::new(child.w + pw, child.h + ph),
        };
        let size = c.constrain(size);
        let inner_size = Size::new(size.w - pw, size.h - ph);
        let offset = self.align.map_or(Vec2::default(), |a| a.offset(inner_size, child));
        cx.place_child(0, vec2(p.left + offset.x, p.top + offset.y));
        size
    }
    fn paint(&self, cx: &mut PaintCx) {
        let r = cx.rect();
        if let Some(bg) = self.background {
            cx.list().rounded_rect(r, self.radius, bg);
        }
        if let Some((width, color)) = self.border {
            cx.list().rounded_rect_outline(r.inset(width * 0.5), self.radius, width, color);
        }
    }
}

/// Space around a child.
pub struct Padding;

impl Padding {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<M: 'static>(padding: impl Into<Insets>, child: impl Into<Element<M>>) -> Container<M> {
        Container::new().padding(padding).child(child)
    }
}

/// A fixed width, height or both; with no child, an empty space (a gap).
pub struct SizedBox<M> {
    child: Vec<AnyElement>,
    width: Option<f32>,
    height: Option<f32>,
    _m: PhantomData<fn() -> M>,
}

impl<M: 'static> SizedBox<M> {
    pub fn new(width: f32, height: f32) -> Self {
        Self { child: Vec::new(), width: Some(width), height: Some(height), _m: PhantomData }
    }
    pub fn width(width: f32) -> Self {
        Self { child: Vec::new(), width: Some(width), height: None, _m: PhantomData }
    }
    pub fn height(height: f32) -> Self {
        Self { child: Vec::new(), width: None, height: Some(height), _m: PhantomData }
    }
    pub fn child(mut self, child: impl Into<Element<M>>) -> Self {
        self.child = one(child);
        self
    }
}

impl<M: 'static> Widget for SizedBox<M> {
    fn role(&self) -> Role {
        Role::Custom("SizedBox")
    }
    fn children(&self) -> &[AnyElement] {
        &self.child
    }
    fn layout(&self, cx: &mut LayoutCx, constraints: Constraints) -> Size {
        let c = fix(constraints, self.width, self.height);
        if self.child.is_empty() {
            return c.constrain(Size::new(self.width.unwrap_or(0.0), self.height.unwrap_or(0.0)));
        }
        let size = cx.layout_child(0, c);
        cx.place_child(0, Vec2::default());
        size
    }
}

/// Places a child within the space it is given.
pub struct Align<M> {
    child: Vec<AnyElement>,
    align: Alignment,
    _m: PhantomData<fn() -> M>,
}

impl<M: 'static> Align<M> {
    pub fn new(align: Alignment, child: impl Into<Element<M>>) -> Self {
        Self { child: one(child), align, _m: PhantomData }
    }
    pub fn center(child: impl Into<Element<M>>) -> Self {
        Self::new(Alignment::CENTER, child)
    }
}

impl<M: 'static> Widget for Align<M> {
    fn role(&self) -> Role {
        Role::Custom("Align")
    }
    fn children(&self) -> &[AnyElement] {
        &self.child
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let child = cx.layout_child(0, c.loosen());
        let size = c.constrain(Size::new(
            if c.has_bounded_width() { c.max.w } else { child.w },
            if c.has_bounded_height() { c.max.h } else { child.h },
        ));
        cx.place_child(0, self.align.offset(size, child));
        size
    }
}

/// Takes a share of a `Row` or `Column`'s free space (see [`Widget::flex`]).
pub struct Expanded<M> {
    child: Vec<AnyElement>,
    flex: u32,
    _m: PhantomData<fn() -> M>,
}

impl<M: 'static> Expanded<M> {
    pub fn new(child: impl Into<Element<M>>) -> Self {
        Self { child: one(child), flex: 1, _m: PhantomData }
    }
    pub fn flex(mut self, flex: u32) -> Self {
        self.flex = flex.max(1);
        self
    }
}

impl<M: 'static> Widget for Expanded<M> {
    fn role(&self) -> Role {
        Role::None
    }
    fn children(&self) -> &[AnyElement] {
        &self.child
    }
    fn flex(&self) -> u32 {
        self.flex
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let size = cx.layout_child(0, c);
        cx.place_child(0, Vec2::default());
        size
    }
}

/// Children on top of each other, later ones above, each placed by the stack's alignment.
pub struct Stack<M> {
    children: Vec<AnyElement>,
    align: Alignment,
    _m: PhantomData<fn() -> M>,
}

impl<M: 'static> Default for Stack<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: 'static> Stack<M> {
    pub fn new() -> Self {
        Self { children: Vec::new(), align: Alignment::TOP_LEFT, _m: PhantomData }
    }
    pub fn child(mut self, child: impl Into<Element<M>>) -> Self {
        self.children.push(child.into().into_any());
        self
    }
    pub fn align(mut self, align: Alignment) -> Self {
        self.align = align;
        self
    }
}

impl<M: 'static> Widget for Stack<M> {
    fn role(&self) -> Role {
        Role::Stack
    }
    fn children(&self) -> &[AnyElement] {
        &self.children
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let sizes: Vec<Size> = (0..self.children.len()).map(|i| cx.layout_child(i, c.loosen())).collect();
        let content = sizes.iter().fold(Size::ZERO, |m, s| Size::new(m.w.max(s.w), m.h.max(s.h)));
        let size = c.constrain(content);
        for (i, s) in sizes.iter().enumerate() {
            cx.place_child(i, self.align.offset(size, *s));
        }
        size
    }
}

into_element!(Container, SizedBox, Align, Expanded, Stack);
