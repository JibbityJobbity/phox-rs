//! Interactive controls: Button, Toggle, Slider, Knob.
//!
//! All of them capture the pointer on a press, so a drag keeps going outside the widget, and all of them are
//! focusable and work from the keyboard (Enter or Space to press; arrows, Home and End to change a value).

use phox_immediate::{text_width, Color};

use crate::element::{Element, EventCx, LayoutCx, PaintCx, Role, Widget};
use crate::event::{Event, KeyCode, Phase};
use crate::geometry::{vec2, Constraints, Rect, Size};
use crate::theme::Theme;

/// Enter and Space activate the focused control.
fn activates(event: &Event) -> bool {
    matches!(event, Event::Key { key: KeyCode::Enter | KeyCode::Space, pressed: true, .. })
}

fn focus_ring(cx: &mut PaintCx, radius: f32) {
    if cx.is_focused() {
        let (r, color) = (cx.rect(), cx.theme().focus);
        cx.list().rounded_rect_outline(r.inset(-2.0), radius + 2.0, 1.5, color);
    }
}

/// The surface colour of a control in its current state.
fn surface(cx: &PaintCx, enabled: bool) -> Color {
    let t = cx.theme();
    if !enabled {
        t.surface.with_alpha(0.5)
    } else if cx.is_active() {
        t.surface_pressed
    } else if cx.is_hot() {
        t.surface_hover
    } else {
        t.surface
    }
}

/// Shared press behaviour: a press captures the pointer, and a release over the widget activates it.
/// Returns true when the widget was activated.
fn press(cx: &mut EventCx, event: &Event) -> bool {
    match event {
        Event::Pointer { id, phase: Phase::Down, .. } if cx.is_hot() => {
            cx.capture(*id);
            cx.request_focus();
            cx.set_handled();
            false
        }
        Event::Pointer { phase: Phase::Up, pos, .. } if cx.has_capture() => {
            cx.release();
            cx.set_handled();
            cx.rect().contains(*pos)
        }
        Event::Pointer { phase: Phase::Cancel, .. } if cx.has_capture() => {
            cx.release();
            false
        }
        e if activates(e) && cx.is_focused() => {
            cx.set_handled();
            true
        }
        _ => false,
    }
}

/// A push button with a text label. Without `on_press` it is disabled.
pub struct Button<M> {
    label: String,
    on_press: Option<M>,
    primary: bool,
}

impl<M: Clone + 'static> Button<M> {
    pub fn new(label: impl Into<String>) -> Self {
        Self { label: label.into(), on_press: None, primary: false }
    }
    pub fn on_press(mut self, message: M) -> Self {
        self.on_press = Some(message);
        self
    }
    /// Drawn in the accent colour.
    pub fn primary(mut self) -> Self {
        self.primary = true;
        self
    }
}

impl<M: Clone + 'static> Widget for Button<M> {
    fn role(&self) -> Role {
        Role::Button
    }
    fn label(&self) -> Option<&str> {
        Some(&self.label)
    }
    fn interactive(&self) -> bool {
        self.on_press.is_some()
    }
    fn focusable(&self) -> bool {
        self.on_press.is_some()
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let t = cx.theme();
        c.constrain(Size::new(text_width(t.text_size, &self.label) + 2.0 * t.control_height * 0.5, t.control_height))
    }
    fn event(&self, cx: &mut EventCx, event: &Event) {
        if let Some(message) = &self.on_press
            && press(cx, event) {
                cx.emit(message.clone());
            }
    }
    fn paint(&self, cx: &mut PaintCx) {
        let enabled = self.on_press.is_some();
        let t: Theme = cx.theme().clone();
        let r = cx.rect();
        let (bg, fg) = if self.primary && enabled {
            let bg = if cx.is_active() { t.accent.lerp(t.surface_pressed, 0.3) } else if cx.is_hot() { t.accent.lerp(t.text, 0.15) } else { t.accent };
            (bg, t.on_accent)
        } else {
            (surface(cx, enabled), if enabled { t.text } else { t.text_muted })
        };
        cx.list().rounded_rect(r, t.radius, bg);
        cx.list().text_centered(r.center().x, r.y + (r.h - t.text_size) * 0.5, t.text_size, &self.label, fg);
        focus_ring(cx, t.radius);
    }
}

/// An on/off switch, with an optional label to its right.
pub struct Toggle<M> {
    value: bool,
    label: Option<String>,
    on_toggle: Option<Box<dyn Fn(bool) -> M>>,
}

const SWITCH: Size = Size::new(34.0, 18.0);

impl<M: 'static> Toggle<M> {
    pub fn new(value: bool) -> Self {
        Self { value, label: None, on_toggle: None }
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
    /// Called with the new value when the switch is flipped.
    pub fn on_toggle(mut self, f: impl Fn(bool) -> M + 'static) -> Self {
        self.on_toggle = Some(Box::new(f));
        self
    }
}

impl<M: 'static> Widget for Toggle<M> {
    fn role(&self) -> Role {
        Role::Toggle
    }
    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    fn interactive(&self) -> bool {
        self.on_toggle.is_some()
    }
    fn focusable(&self) -> bool {
        self.on_toggle.is_some()
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let t = cx.theme();
        let label = self.label.as_ref().map_or(0.0, |l| t.spacing + text_width(t.text_size, l));
        c.constrain(Size::new(SWITCH.w + label, t.control_height))
    }
    fn event(&self, cx: &mut EventCx, event: &Event) {
        if let Some(f) = &self.on_toggle
            && press(cx, event) {
                cx.emit(f(!self.value));
            }
    }
    fn paint(&self, cx: &mut PaintCx) {
        let t = cx.theme().clone();
        let r = cx.rect();
        let track = Rect::new(r.x, r.y + (r.h - SWITCH.h) * 0.5, SWITCH.w, SWITCH.h);
        let enabled = self.on_toggle.is_some();
        let color = if self.value { t.accent } else { surface(cx, enabled) };
        cx.list().rounded_rect(track, SWITCH.h * 0.5, color);
        let knob_x = if self.value { track.x + track.w - SWITCH.h * 0.5 } else { track.x + SWITCH.h * 0.5 };
        cx.list().circle(vec2(knob_x, track.center().y), SWITCH.h * 0.5 - 3.0, if self.value { t.on_accent } else { t.text });
        if let Some(label) = &self.label {
            cx.list().text(vec2(track.x + track.w + t.spacing, r.y + (r.h - t.text_size) * 0.5), t.text_size, label, t.text);
        }
        if cx.is_focused() {
            cx.list().rounded_rect_outline(track.inset(-2.0), SWITCH.h * 0.5 + 2.0, 1.5, t.focus);
        }
    }
}

/// A value in a range, keyboard steps and the callback, shared by Slider and Knob.
struct Value<M> {
    value: f32,
    min: f32,
    max: f32,
    step: Option<f32>,
    label: Option<String>,
    on_change: Option<Box<dyn Fn(f32) -> M>>,
}

impl<M: 'static> Value<M> {
    fn new(value: f32, min: f32, max: f32) -> Self {
        let (min, max) = if max > min { (min, max) } else { (min, min + 1.0) };
        Self { value: value.clamp(min, max), min, max, step: None, label: None, on_change: None }
    }

    fn fraction(&self) -> f32 {
        (self.value - self.min) / (self.max - self.min)
    }

    fn value_at_fraction(&self, f: f32) -> f32 {
        let v = self.min + f.clamp(0.0, 1.0) * (self.max - self.min);
        match self.step {
            Some(step) if step > 0.0 => (self.min + ((v - self.min) / step).round() * step).clamp(self.min, self.max),
            _ => v,
        }
    }

    fn keyboard_step(&self) -> f32 {
        self.step.unwrap_or((self.max - self.min) / 100.0)
    }

    fn emit(&self, cx: &mut EventCx, v: f32) {
        if let Some(f) = &self.on_change
            && v != self.value {
                cx.emit(f(v));
            }
    }

    /// The wheel over the control: one step per notch (a hundredth of the range without a step), up to raise.
    fn scroll(&self, cx: &mut EventCx, event: &Event) -> bool {
        let Event::Scroll { delta, .. } = event else { return false };
        if delta.y == 0.0 {
            return false;
        }
        let v = self.value - delta.y.signum() * self.keyboard_step();
        self.emit(cx, v.clamp(self.min, self.max));
        cx.set_handled();
        true
    }

    /// Arrow keys, Home and End. Returns whether the key was used.
    fn key(&self, cx: &mut EventCx, event: &Event) -> bool {
        let Event::Key { key, pressed: true, .. } = event else { return false };
        if !cx.is_focused() {
            return false;
        }
        let step = self.keyboard_step();
        let v = match key {
            KeyCode::Left | KeyCode::Down => self.value - step,
            KeyCode::Right | KeyCode::Up => self.value + step,
            KeyCode::Home => self.min,
            KeyCode::End => self.max,
            _ => return false,
        };
        self.emit(cx, v.clamp(self.min, self.max));
        cx.set_handled();
        true
    }
}

macro_rules! value_builders {
    ($ty:ident) => {
        impl<M: 'static> $ty<M> {
            /// Rounds values to multiples of `step` from the minimum (also the keyboard step).
            pub fn step(mut self, step: f32) -> Self {
                self.v.step = Some(step);
                self
            }
            pub fn label(mut self, label: impl Into<String>) -> Self {
                self.v.label = Some(label.into());
                self
            }
            /// Called with the new value while it is dragged or stepped.
            pub fn on_change(mut self, f: impl Fn(f32) -> M + 'static) -> Self {
                self.v.on_change = Some(Box::new(f));
                self
            }
        }
    };
}

/// A horizontal slider. Click or drag along the track; the value follows the pointer.
pub struct Slider<M> {
    v: Value<M>,
    width: Option<f32>,
}

impl<M: 'static> Slider<M> {
    pub fn new(value: f32, min: f32, max: f32) -> Self {
        Self { v: Value::new(value, min, max), width: None }
    }
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }
}

value_builders!(Slider);

const THUMB: f32 = 7.0;

impl<M: 'static> Slider<M> {
    fn value_at(&self, rect: Rect, x: f32) -> f32 {
        let span = (rect.w - 2.0 * THUMB).max(1.0);
        self.v.value_at_fraction((x - rect.x - THUMB) / span)
    }
}

impl<M: 'static> Widget for Slider<M> {
    fn role(&self) -> Role {
        Role::Slider
    }
    fn label(&self) -> Option<&str> {
        self.v.label.as_deref()
    }
    fn interactive(&self) -> bool {
        self.v.on_change.is_some()
    }
    fn focusable(&self) -> bool {
        self.v.on_change.is_some()
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let w = self.width.unwrap_or(if c.has_bounded_width() { c.max.w } else { 160.0 });
        c.constrain(Size::new(w, cx.theme().control_height))
    }
    fn event(&self, cx: &mut EventCx, event: &Event) {
        if self.v.on_change.is_none() {
            return;
        }
        match event {
            Event::Pointer { id, phase: Phase::Down, pos, .. } if cx.is_hot() => {
                cx.capture(*id);
                cx.request_focus();
                cx.set_handled();
                let v = self.value_at(cx.rect(), pos.x);
                self.v.emit(cx, v);
            }
            Event::Pointer { phase: Phase::Move, pos, .. } if cx.has_capture() => {
                cx.set_handled();
                let v = self.value_at(cx.rect(), pos.x);
                self.v.emit(cx, v);
            }
            Event::Pointer { phase: Phase::Up | Phase::Cancel, .. } if cx.has_capture() => {
                cx.release();
                cx.set_handled();
            }
            _ => {
                if !self.v.scroll(cx, event) {
                    self.v.key(cx, event);
                }
            }
        }
    }
    fn paint(&self, cx: &mut PaintCx) {
        let t = cx.theme().clone();
        let r = cx.rect();
        let y = r.center().y;
        let (x0, x1) = (r.x + THUMB, r.x + r.w - THUMB);
        let x = x0 + (x1 - x0) * self.v.fraction();
        let enabled = self.v.on_change.is_some();
        cx.list().line(vec2(x0, y), vec2(x1, y), 4.0, t.surface_hover);
        cx.list().line(vec2(x0, y), vec2(x, y), 4.0, if enabled { t.accent } else { t.text_muted });
        let thumb = if cx.is_active() || cx.is_hot() { t.text } else { t.text.lerp(t.text_muted, 0.3) };
        cx.list().circle(vec2(x, y), THUMB, thumb);
        if cx.is_focused() {
            cx.list().arc(vec2(x, y), THUMB + 3.0, 1.5, 0.0, std::f32::consts::TAU, t.focus);
        }
    }
}

/// A rotary knob: drag up to increase, down to decrease (200 logical pixels for the whole range).
pub struct Knob<M> {
    v: Value<M>,
    diameter: f32,
}

/// The drag that is in progress: where it started, and the value then.
#[derive(Default)]
struct KnobDrag {
    start: Option<(f32, f32)>,
}

const KNOB_DRAG_RANGE: f32 = 200.0;
/// The knob's travel: 270 degrees, starting bottom left (angles clockwise from +x on screen).
const KNOB_START: f32 = std::f32::consts::PI * 0.75;
const KNOB_SWEEP: f32 = std::f32::consts::PI * 1.5;

impl<M: 'static> Knob<M> {
    pub fn new(value: f32, min: f32, max: f32) -> Self {
        Self { v: Value::new(value, min, max), diameter: 44.0 }
    }
    pub fn diameter(mut self, diameter: f32) -> Self {
        self.diameter = diameter;
        self
    }
}

value_builders!(Knob);

impl<M: 'static> Widget for Knob<M> {
    fn role(&self) -> Role {
        Role::Knob
    }
    fn label(&self) -> Option<&str> {
        self.v.label.as_deref()
    }
    fn interactive(&self) -> bool {
        self.v.on_change.is_some()
    }
    fn focusable(&self) -> bool {
        self.v.on_change.is_some()
    }
    fn new_state(&self) -> Box<dyn std::any::Any> {
        Box::new(KnobDrag::default())
    }
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let t = cx.theme();
        let label = if self.v.label.is_some() { t.line_height(t.text_size) } else { 0.0 };
        let label_w = self.v.label.as_ref().map_or(0.0, |l| text_width(t.text_size, l));
        c.constrain(Size::new(self.diameter.max(label_w), self.diameter + label))
    }
    fn event(&self, cx: &mut EventCx, event: &Event) {
        if self.v.on_change.is_none() {
            return;
        }
        match event {
            Event::Pointer { id, phase: Phase::Down, pos, .. } if cx.is_hot() => {
                cx.capture(*id);
                cx.request_focus();
                cx.set_handled();
                cx.state::<KnobDrag>().start = Some((pos.y, self.v.value));
            }
            Event::Pointer { phase: Phase::Move, pos, .. } if cx.has_capture() => {
                cx.set_handled();
                if let Some((y0, v0)) = cx.state::<KnobDrag>().start {
                    let f = (v0 - self.v.min) / (self.v.max - self.v.min) + (y0 - pos.y) / KNOB_DRAG_RANGE;
                    let v = self.v.value_at_fraction(f);
                    self.v.emit(cx, v);
                }
            }
            Event::Pointer { phase: Phase::Up | Phase::Cancel, .. } if cx.has_capture() => {
                cx.state::<KnobDrag>().start = None;
                cx.release();
                cx.set_handled();
            }
            _ => {
                if !self.v.scroll(cx, event) {
                    self.v.key(cx, event);
                }
            }
        }
    }
    fn paint(&self, cx: &mut PaintCx) {
        let t = cx.theme().clone();
        let r = cx.rect();
        let radius = self.diameter * 0.5;
        let center = vec2(r.center().x, r.y + radius);
        let enabled = self.v.on_change.is_some();
        let body = if cx.is_active() { t.surface_pressed } else if cx.is_hot() { t.surface_hover } else { t.surface };
        cx.list().circle(center, radius - 6.0, body);
        cx.list().arc(center, radius - 2.0, 3.0, KNOB_START, KNOB_START + KNOB_SWEEP, t.surface_hover);
        let end = KNOB_START + KNOB_SWEEP * self.v.fraction();
        cx.list().arc(center, radius - 2.0, 3.0, KNOB_START, end, if enabled { t.accent } else { t.text_muted });
        let tip = vec2(center.x + end.cos() * (radius - 10.0), center.y + end.sin() * (radius - 10.0));
        cx.list().line(center, tip, 2.0, t.text);
        if let Some(label) = &self.v.label {
            cx.list().text_centered(center.x, r.y + self.diameter + (t.line_height(t.text_size) - t.text_size) * 0.5, t.text_size, label, t.text_muted);
        }
        if cx.is_focused() {
            cx.list().arc(center, radius + 1.0, 1.5, 0.0, std::f32::consts::TAU, t.focus);
        }
    }
}

macro_rules! into_element_m {
    ($($ty:ident),*) => {$(
        impl<M: 'static> From<$ty<M>> for Element<M> {
            fn from(w: $ty<M>) -> Self {
                Element::new(w)
            }
        }
    )*};
}

impl<M: Clone + 'static> From<Button<M>> for Element<M> {
    fn from(w: Button<M>) -> Self {
        Element::new(w)
    }
}

into_element_m!(Toggle, Slider, Knob);
