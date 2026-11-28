//! A single-line text field. Its text is the app's (controlled, as in Elm): editing emits `on_change` with the
//! new text, and the field shows whatever `view` passes next. The field itself keeps only the caret, the
//! selection and how far it is scrolled.
//!
//! Editing: click to place the caret, drag to select; arrows, Home and End move (with Shift they select);
//! [`KeyCode::SelectAll`], [`KeyCode::Copy`] and [`KeyCode::Cut`] work on the selection; [`Event::Paste`]
//! and typing replace it. The host turns the platform's shortcuts into those keys and events.

use std::cell::Cell;

use phox_immediate::text_width;

use crate::element::{Element, EventCx, LayoutCx, PaintCx, Role, Widget};
use crate::event::{Event, KeyCode, Phase};
use crate::geometry::{vec2, Constraints, Rect, Size};

pub struct TextField<M> {
    value: String,
    placeholder: String,
    width: Option<f32>,
    on_change: Option<Box<dyn Fn(String) -> M>>,
    on_submit: Option<Box<dyn Fn() -> M>>,
}

/// Positions are counts of characters from the start, clamped to the current text.
struct Edit {
    caret: usize,
    /// The other end of the selection, when there is one.
    anchor: Option<usize>,
    /// A mouse drag is selecting.
    dragging: bool,
    /// How far the text is scrolled left so the caret stays visible (set by paint, read by clicks).
    scroll: Cell<f32>,
}

impl Edit {
    /// The selected range, if it isn't empty.
    fn selection(&self, len: usize) -> Option<(usize, usize)> {
        let (c, a) = (self.caret.min(len), self.anchor?.min(len));
        (c != a).then(|| (c.min(a), c.max(a)))
    }
}

const PADDING: f32 = 8.0;

/// Horizontal distance from one character to the next at `size` (the stroke font is monospaced).
fn advance(size: f32) -> f32 {
    size * 5.6 / 6.0
}

impl<M: 'static> TextField<M> {
    pub fn new(value: impl Into<String>) -> Self {
        Self { value: value.into(), placeholder: String::new(), width: None, on_change: None, on_submit: None }
    }
    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = text.into();
        self
    }
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }
    pub fn on_change(mut self, f: impl Fn(String) -> M + 'static) -> Self {
        self.on_change = Some(Box::new(f));
        self
    }
    /// Sent when Enter is pressed in the field.
    pub fn on_submit(mut self, message: M) -> Self
    where
        M: Clone,
    {
        self.on_submit = Some(Box::new(move || message.clone()));
        self
    }

    fn len(&self) -> usize {
        self.value.chars().count()
    }

    /// The byte index of character `chars`.
    fn byte(&self, chars: usize) -> usize {
        self.value.char_indices().nth(chars).map_or(self.value.len(), |(i, _)| i)
    }

    fn slice(&self, from: usize, to: usize) -> &str {
        &self.value[self.byte(from)..self.byte(to)]
    }

    /// The character boundary nearest `x` (window coordinates).
    fn index_at(&self, rect: Rect, scroll: f32, size: f32, x: f32) -> usize {
        let rel = x - (rect.x + PADDING) + scroll;
        ((rel / advance(size)).round().max(0.0) as usize).min(self.len())
    }

    /// The value with `range` replaced by `text`, and the caret after it.
    fn replace(&self, range: (usize, usize), text: &str) -> (String, usize) {
        let mut value = self.value.clone();
        value.replace_range(self.byte(range.0)..self.byte(range.1), text);
        (value, range.0 + text.chars().count())
    }
}

/// Text fit for one line: line breaks at the ends dropped (a copied key often ends in one), inner ones and tabs
/// turned into spaces, other control characters removed.
fn single_line(text: &str) -> String {
    text.trim_matches(['\r', '\n']).chars().filter_map(|c| match c {
        '\n' | '\r' | '\t' => Some(' '),
        c if c.is_control() => None,
        c => Some(c),
    }).collect()
}

impl<M: 'static> Widget for TextField<M> {
    fn role(&self) -> Role {
        Role::TextField
    }
    fn label(&self) -> Option<&str> {
        Some(&self.value)
    }
    fn interactive(&self) -> bool {
        self.on_change.is_some()
    }
    fn focusable(&self) -> bool {
        self.on_change.is_some()
    }
    fn new_state(&self) -> Box<dyn std::any::Any> {
        // Focused from the keyboard, the caret starts at the end.
        Box::new(Edit { caret: usize::MAX, anchor: None, dragging: false, scroll: Cell::new(0.0) })
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let w = self.width.unwrap_or(if c.has_bounded_width() { c.max.w } else { 160.0 });
        c.constrain(Size::new(w, cx.theme().control_height))
    }
    fn event(&self, cx: &mut EventCx, event: &Event) {
        let Some(on_change) = &self.on_change else { return };
        let len = self.len();
        let size = cx.theme().text_size;
        let rect = cx.rect();
        let (caret, selection) = {
            let st = cx.state::<Edit>();
            st.caret = st.caret.min(len);
            (st.caret, st.selection(len))
        };
        // Replaces the selection (or inserts at the caret) and reports the new value.
        let edit = |cx: &mut EventCx, text: &str| {
            let (value, at) = self.replace(selection.unwrap_or((caret, caret)), text);
            let st = cx.state::<Edit>();
            st.caret = at;
            st.anchor = None;
            if value != self.value {
                cx.emit(on_change(value));
            }
            cx.request_repaint();
            cx.set_handled();
        };
        match event {
            Event::Pointer { id, phase: Phase::Down, pos, .. } if cx.is_hot() => {
                cx.request_focus();
                cx.capture(*id);
                let st = cx.state::<Edit>();
                let at = self.index_at(rect, st.scroll.get(), size, pos.x);
                st.caret = at;
                st.anchor = Some(at);
                st.dragging = true;
                cx.set_handled();
                cx.request_repaint();
            }
            Event::Pointer { phase: Phase::Move, pos, .. } if cx.has_capture() => {
                let st = cx.state::<Edit>();
                if st.dragging {
                    st.caret = self.index_at(rect, st.scroll.get(), size, pos.x);
                    cx.request_repaint();
                }
                cx.set_handled();
            }
            Event::Pointer { phase: Phase::Up | Phase::Cancel, .. } if cx.has_capture() => {
                cx.release();
                let st = cx.state::<Edit>();
                st.dragging = false;
                if st.anchor == Some(st.caret) {
                    st.anchor = None;
                }
                cx.set_handled();
            }
            Event::Text(text) if cx.is_focused() => {
                let typed: String = text.chars().filter(|c| !c.is_control()).collect();
                if !typed.is_empty() {
                    edit(cx, &typed);
                }
            }
            Event::Paste(text) if cx.is_focused() => edit(cx, &single_line(text)),
            Event::Key { key, pressed: true, modifiers } if cx.is_focused() => {
                let shift = modifiers.shift;
                // Moves the caret; with Shift the selection follows, without it the selection goes.
                let go = |cx: &mut EventCx, to: usize| {
                    let st = cx.state::<Edit>();
                    if shift {
                        st.anchor = st.anchor.or(Some(caret));
                    } else {
                        st.anchor = None;
                    }
                    st.caret = to;
                    cx.request_repaint();
                    cx.set_handled();
                };
                match key {
                    KeyCode::Left => go(cx, match selection {
                        Some((lo, _)) if !shift => lo,
                        _ => caret.saturating_sub(1),
                    }),
                    KeyCode::Right => go(cx, match selection {
                        Some((_, hi)) if !shift => hi,
                        _ => (caret + 1).min(len),
                    }),
                    KeyCode::Home => go(cx, 0),
                    KeyCode::End => go(cx, len),
                    KeyCode::Backspace | KeyCode::Delete if selection.is_some() => edit(cx, ""),
                    KeyCode::Backspace => {
                        if caret > 0 {
                            let (value, _) = self.replace((caret - 1, caret), "");
                            cx.state::<Edit>().caret = caret - 1;
                            cx.emit(on_change(value));
                        }
                        cx.set_handled();
                    }
                    KeyCode::Delete => {
                        if caret < len {
                            let (value, _) = self.replace((caret, caret + 1), "");
                            cx.emit(on_change(value));
                        }
                        cx.set_handled();
                    }
                    KeyCode::SelectAll => {
                        let st = cx.state::<Edit>();
                        st.anchor = Some(0);
                        st.caret = len;
                        cx.request_repaint();
                        cx.set_handled();
                    }
                    KeyCode::Copy | KeyCode::Cut => {
                        if let Some((lo, hi)) = selection {
                            cx.set_clipboard(self.slice(lo, hi));
                            if *key == KeyCode::Cut {
                                edit(cx, "");
                            }
                        }
                        cx.set_handled();
                    }
                    KeyCode::Enter => {
                        if let Some(submit) = &self.on_submit {
                            cx.emit(submit());
                        }
                        cx.set_handled();
                    }
                    KeyCode::Escape => {
                        cx.state::<Edit>().anchor = None;
                        cx.blur();
                        cx.set_handled();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    fn paint(&self, cx: &mut PaintCx) {
        let t = cx.theme().clone();
        let r = cx.rect();
        let focused = cx.is_focused();
        let border = if focused { t.focus } else if cx.is_hot() { t.text_muted } else { t.border };
        cx.list().rounded_rect(r, t.radius, t.surface_pressed);
        cx.list().rounded_rect_outline(r.inset(0.5), t.radius, 1.0, border);
        let len = self.len();
        let (caret, selection) = {
            let st = cx.state::<Edit>();
            (st.caret.min(len), st.selection(len))
        };
        // Between characters: the advance is wider than a glyph, so the caret sits in the middle of the gap.
        let caret_x = |i: usize| if i == 0 { -2.0 } else { text_width(t.text_size, &self.value.chars().take(i).collect::<String>()) + t.text_size * 0.13 + 1.0 };
        // Scroll so the caret stays in view while focused; show the start otherwise.
        let visible = r.w - 2.0 * PADDING;
        let full = text_width(t.text_size, &self.value);
        let mut scroll = cx.state::<Edit>().scroll.get();
        if !focused || full <= visible {
            scroll = 0.0;
        } else {
            let x = caret_x(caret);
            if x - scroll > visible {
                scroll = x - visible;
            } else if x - scroll < 0.0 {
                scroll = x.max(0.0);
            }
            scroll = scroll.clamp(0.0, (full - visible + 4.0).max(0.0));
        }
        cx.state::<Edit>().scroll.set(scroll);
        let x0 = r.x + PADDING - scroll;
        let text_y = r.y + (r.h - t.text_size) * 0.5;
        cx.list().push_clip(r.inset(2.0));
        if let (true, Some((lo, hi))) = (focused, selection) {
            let adv = advance(t.text_size);
            let sel = Rect::new(x0 + lo as f32 * adv - 1.5, text_y - 3.0, (hi - lo) as f32 * adv, t.text_size + 6.0);
            cx.list().rect(sel, t.accent.with_alpha(0.35));
        }
        if self.value.is_empty() {
            cx.list().text(vec2(r.x + PADDING, text_y), t.text_size, &self.placeholder, t.text_muted);
        } else {
            cx.list().text(vec2(x0, text_y), t.text_size, &self.value, t.text);
        }
        if focused {
            let x = x0 + caret_x(caret);
            cx.list().line(vec2(x, text_y - 2.0), vec2(x, text_y + t.text_size + 2.0), 1.5, t.focus);
        }
        cx.list().pop_clip();
    }
}

impl<M: 'static> From<TextField<M>> for Element<M> {
    fn from(w: TextField<M>) -> Self {
        Element::new(w)
    }
}

#[cfg(test)]
mod tests {
    use super::single_line;

    #[test]
    fn pasted_text_becomes_one_line() {
        assert_eq!(single_line("pk_live_abc\n"), "pk_live_abc");
        assert_eq!(single_line("a\tb\r\nc\u{7}"), "a b  c");
    }
}
