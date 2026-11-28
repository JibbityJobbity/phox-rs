//! A vertically scrolling area. Its child gets unbounded height; the wheel or trackpad moves it, clipped to
//! the area, with a scrollbar when it overflows.

use std::cell::Cell;
use std::marker::PhantomData;

use crate::element::{AnyElement, Element, EventCx, LayoutCx, PaintCx, Role, Widget};
use crate::event::Event;
use crate::geometry::{vec2, Constraints, Rect, Size};

pub struct Scroll<M> {
    child: Vec<AnyElement>,
    _m: PhantomData<fn() -> M>,
}

/// Kept in cells because layout (which only reads state) is where the offset is clamped to the content.
#[derive(Default)]
struct ScrollState {
    offset: Cell<f32>,
    content: Cell<f32>,
    viewport: Cell<f32>,
}

impl ScrollState {
    fn max_offset(&self) -> f32 {
        (self.content.get() - self.viewport.get()).max(0.0)
    }
}

impl<M: 'static> Scroll<M> {
    pub fn new(child: impl Into<Element<M>>) -> Self {
        Self { child: vec![child.into().into_any()], _m: PhantomData }
    }
}

impl<M: 'static> Widget for Scroll<M> {
    fn role(&self) -> Role {
        Role::Scroll
    }
    fn children(&self) -> &[AnyElement] {
        &self.child
    }
    fn clips(&self) -> bool {
        true
    }
    fn new_state(&self) -> Box<dyn std::any::Any> {
        Box::new(ScrollState::default())
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let width = Constraints { min: Size::new(c.min.w, 0.0), max: Size::new(c.max.w, f32::INFINITY) };
        let child = cx.layout_child(0, width);
        let size = c.constrain(Size::new(child.w, if c.has_bounded_height() { c.max.h } else { child.h }));
        let state = cx.state::<ScrollState>();
        state.content.set(child.h);
        state.viewport.set(size.h);
        state.offset.set(state.offset.get().clamp(0.0, state.max_offset()));
        let offset = state.offset.get();
        cx.place_child(0, vec2(0.0, -offset));
        size
    }
    fn event(&self, cx: &mut EventCx, event: &Event) {
        if let Event::Scroll { delta, .. } = event {
            let state = cx.state::<ScrollState>();
            let before = state.offset.get();
            let after = (before + delta.y).clamp(0.0, state.max_offset());
            if after != before {
                state.offset.set(after);
                cx.request_layout();
                cx.set_handled();
            }
        }
    }
    fn paint(&self, cx: &mut PaintCx) {
        let state = cx.state::<ScrollState>();
        let (content, viewport, offset) = (state.content.get(), state.viewport.get(), state.offset.get());
        if content <= viewport || content <= 0.0 {
            return;
        }
        let r = cx.rect();
        let h = (viewport / content * r.h).max(16.0);
        let y = r.y + offset / (content - viewport) * (r.h - h);
        let color = cx.theme().text_muted.with_alpha(0.5);
        cx.list().rounded_rect(Rect::new(r.x + r.w - 5.0, y, 3.0, h), 1.5, color);
    }
}

impl<M: 'static> From<Scroll<M>> for Element<M> {
    fn from(w: Scroll<M>) -> Self {
        Element::new(w)
    }
}
