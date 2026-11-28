//! The look every built-in widget takes its colours and metrics from. An app returns one from
//! [`crate::App::theme`]; widgets read it from their paint and layout contexts.

use phox_immediate::Color;

#[derive(Clone, Debug, PartialEq)]
pub struct Theme {
    pub background: Color,
    pub surface: Color,
    pub surface_hover: Color,
    pub surface_pressed: Color,
    pub accent: Color,
    pub on_accent: Color,
    pub text: Color,
    pub text_muted: Color,
    pub border: Color,
    pub focus: Color,
    /// Capital height of body text, in logical pixels.
    pub text_size: f32,
    pub radius: f32,
    /// The default gap between related widgets.
    pub spacing: f32,
    /// Height of buttons, sliders, fields and other single-line controls.
    pub control_height: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background: Color::hex(0x16181c),
            surface: Color::hex(0x272a31),
            surface_hover: Color::hex(0x32363f),
            surface_pressed: Color::hex(0x1e2026),
            accent: Color::hex(0x4f9dff),
            on_accent: Color::hex(0x0b1220),
            text: Color::hex(0xe6e8ec),
            text_muted: Color::hex(0x8b919c),
            border: Color::hex(0x3c414b),
            focus: Color::hex(0x8fc2ff),
            text_size: 10.0,
            radius: 6.0,
            spacing: 8.0,
            control_height: 28.0,
        }
    }
}

impl Theme {
    /// Height of a line of text at `size` (capital height plus room above and below).
    pub fn line_height(&self, size: f32) -> f32 {
        (size * 1.6).ceil()
    }
}
