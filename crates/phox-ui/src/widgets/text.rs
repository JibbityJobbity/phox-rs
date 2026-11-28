//! A line of text in the stroke font.

use phox_immediate::{text_width, Color};

use crate::element::{Element, LayoutCx, PaintCx, Role, Widget};
use crate::geometry::{vec2, Constraints, Size};

pub struct Text {
    text: String,
    size: Option<f32>,
    color: Option<Color>,
    muted: bool,
}

impl Text {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), size: None, color: None, muted: false }
    }
    /// Capital height in logical pixels (default: the theme's `text_size`).
    pub fn size(mut self, size: f32) -> Self {
        self.size = Some(size);
        self
    }
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
    /// The theme's secondary text colour.
    pub fn muted(mut self) -> Self {
        self.muted = true;
        self
    }
}

impl Widget for Text {
    fn role(&self) -> Role {
        Role::Text
    }
    fn label(&self) -> Option<&str> {
        Some(&self.text)
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let size = self.size.unwrap_or(cx.theme().text_size);
        c.constrain(Size::new(text_width(size, &self.text), cx.theme().line_height(size)))
    }
    fn paint(&self, cx: &mut PaintCx) {
        let theme = cx.theme();
        let size = self.size.unwrap_or(theme.text_size);
        let color = self.color.unwrap_or(if self.muted { theme.text_muted } else { theme.text });
        let r = cx.rect();
        cx.list().text(vec2(r.x, r.y + (r.h - size) * 0.5), size, &self.text, color);
    }
}

impl<M: 'static> From<Text> for Element<M> {
    fn from(t: Text) -> Self {
        Element::new(t)
    }
}

impl<M: 'static> From<&str> for Element<M> {
    fn from(s: &str) -> Self {
        Element::new(Text::new(s))
    }
}

impl<M: 'static> From<String> for Element<M> {
    fn from(s: String) -> Self {
        Element::new(Text::new(s))
    }
}
