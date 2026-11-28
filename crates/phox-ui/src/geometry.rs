//! Sizes, box constraints, insets and alignment, in logical pixels.

pub use phox_immediate::{vec2, Rect, Vec2};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    pub w: f32,
    pub h: f32,
}

impl Size {
    pub const ZERO: Size = Size { w: 0.0, h: 0.0 };
    pub const INFINITE: Size = Size { w: f32::INFINITY, h: f32::INFINITY };

    pub const fn new(w: f32, h: f32) -> Self {
        Self { w, h }
    }
}

/// Flutter's box constraints: the size a widget may take. Constraints go down the tree, sizes come back up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraints {
    pub min: Size,
    pub max: Size,
}

impl Constraints {
    /// Exactly `size`.
    pub const fn tight(size: Size) -> Self {
        Self { min: size, max: size }
    }

    /// Anything from zero up to `max`.
    pub const fn loose(max: Size) -> Self {
        Self { min: Size::ZERO, max }
    }

    /// The same maximum, without a minimum.
    pub fn loosen(self) -> Self {
        Self { min: Size::ZERO, max: self.max }
    }

    /// The nearest size to `size` that these constraints allow.
    pub fn constrain(self, size: Size) -> Size {
        Size::new(size.w.clamp(self.min.w, self.max.w.max(self.min.w)), size.h.clamp(self.min.h, self.max.h.max(self.min.h)))
    }

    /// Shrunk by `insets` on every side (never below zero).
    pub fn deflate(self, insets: Insets) -> Self {
        let (dw, dh) = (insets.left + insets.right, insets.top + insets.bottom);
        Self {
            min: Size::new((self.min.w - dw).max(0.0), (self.min.h - dh).max(0.0)),
            max: Size::new((self.max.w - dw).max(0.0), (self.max.h - dh).max(0.0)),
        }
    }

    pub fn has_bounded_width(self) -> bool {
        self.max.w.is_finite()
    }

    pub fn has_bounded_height(self) -> bool {
        self.max.h.is_finite()
    }

    /// The largest allowed size, or the smallest where a side is unbounded.
    pub fn biggest(self) -> Size {
        Size::new(if self.max.w.is_finite() { self.max.w } else { self.min.w }, if self.max.h.is_finite() { self.max.h } else { self.min.h })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Insets {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Insets {
    pub const fn all(v: f32) -> Self {
        Self { left: v, top: v, right: v, bottom: v }
    }

    pub const fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self { left: horizontal, top: vertical, right: horizontal, bottom: vertical }
    }
}

impl From<f32> for Insets {
    fn from(v: f32) -> Self {
        Insets::all(v)
    }
}

/// Where a child sits in the space its parent gives it: -1 is the start (left or top), 0 the centre, 1 the
/// end, as Flutter's `Alignment`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alignment {
    pub x: f32,
    pub y: f32,
}

impl Alignment {
    pub const TOP_LEFT: Self = Self { x: -1.0, y: -1.0 };
    pub const TOP_CENTER: Self = Self { x: 0.0, y: -1.0 };
    pub const TOP_RIGHT: Self = Self { x: 1.0, y: -1.0 };
    pub const CENTER_LEFT: Self = Self { x: -1.0, y: 0.0 };
    pub const CENTER: Self = Self { x: 0.0, y: 0.0 };
    pub const CENTER_RIGHT: Self = Self { x: 1.0, y: 0.0 };
    pub const BOTTOM_LEFT: Self = Self { x: -1.0, y: 1.0 };
    pub const BOTTOM_CENTER: Self = Self { x: 0.0, y: 1.0 };
    pub const BOTTOM_RIGHT: Self = Self { x: 1.0, y: 1.0 };

    /// The offset of a `child` box inside a `parent` box.
    pub fn offset(self, parent: Size, child: Size) -> Vec2 {
        vec2((parent.w - child.w) * (self.x + 1.0) * 0.5, (parent.h - child.h) * (self.y + 1.0) * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraints_clamp_and_deflate() {
        let c = Constraints { min: Size::new(10.0, 10.0), max: Size::new(100.0, 50.0) };
        assert_eq!(c.constrain(Size::new(500.0, 1.0)), Size::new(100.0, 10.0));
        let d = c.deflate(Insets::all(8.0));
        assert_eq!(d.min, Size::new(0.0, 0.0));
        assert_eq!(d.max, Size::new(84.0, 34.0));
        assert_eq!(Constraints::loose(Size::INFINITE).biggest(), Size::ZERO);
    }

    #[test]
    fn alignment_places_children() {
        let (parent, child) = (Size::new(100.0, 50.0), Size::new(20.0, 10.0));
        assert_eq!(Alignment::TOP_LEFT.offset(parent, child), vec2(0.0, 0.0));
        assert_eq!(Alignment::CENTER.offset(parent, child), vec2(40.0, 20.0));
        assert_eq!(Alignment::BOTTOM_RIGHT.offset(parent, child), vec2(80.0, 40.0));
    }
}
