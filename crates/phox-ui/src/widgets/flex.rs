//! `Row` and `Column`, with Flutter's flex rules: children without flex are measured first, then `Expanded`
//! children share what is left by their flex factors.

use std::marker::PhantomData;

use crate::element::{AnyElement, Element, LayoutCx, Role, Widget};
use crate::geometry::{vec2, Constraints, Size};
use crate::widgets::layout::into_element;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MainAxis {
    #[default]
    Start,
    Center,
    End,
    /// Free space goes between the children.
    SpaceBetween,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossAxis {
    #[default]
    Start,
    Center,
    End,
    /// Children fill the cross axis (when it is bounded).
    Stretch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Horizontal,
    Vertical,
}

struct Flex {
    axis: Axis,
    children: Vec<AnyElement>,
    gap: f32,
    main: MainAxis,
    cross: CrossAxis,
    /// Takes only the space its children need on the main axis (Flutter's `MainAxisSize.min`).
    shrink: bool,
}

impl Flex {
    /// (main, cross) of a size along this axis.
    fn split(&self, s: Size) -> (f32, f32) {
        match self.axis {
            Axis::Horizontal => (s.w, s.h),
            Axis::Vertical => (s.h, s.w),
        }
    }

    fn join(&self, main: f32, cross: f32) -> Size {
        match self.axis {
            Axis::Horizontal => Size::new(main, cross),
            Axis::Vertical => Size::new(cross, main),
        }
    }

    fn constraints(&self, main: (f32, f32), cross: (f32, f32)) -> Constraints {
        Constraints { min: self.join(main.0, cross.0), max: self.join(main.1, cross.1) }
    }

    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
        let n = self.children.len();
        let (main_max, cross_max) = self.split(c.max);
        let (_, cross_min) = self.split(c.min);
        let stretch = self.cross == CrossAxis::Stretch && cross_max.is_finite();
        let cross_range = if stretch { (cross_max, cross_max) } else { (0.0, cross_max) };
        let gaps = self.gap * n.saturating_sub(1) as f32;

        let mut sizes = vec![Size::ZERO; n];
        let mut used = gaps;
        let mut total_flex = 0;
        for (i, size) in sizes.iter_mut().enumerate() {
            let flex = cx.child_flex(i);
            if flex > 0 && main_max.is_finite() {
                total_flex += flex;
                continue;
            }
            *size = cx.layout_child(i, self.constraints((0.0, f32::INFINITY), cross_range));
            used += self.split(*size).0;
        }
        if total_flex > 0 {
            let free = (main_max - used).max(0.0);
            for (i, size) in sizes.iter_mut().enumerate() {
                let flex = cx.child_flex(i);
                if flex == 0 {
                    continue;
                }
                let share = free * flex as f32 / total_flex as f32;
                *size = cx.layout_child(i, self.constraints((share, share), cross_range));
                used += self.split(*size).0;
            }
        }

        let main = if self.shrink || !main_max.is_finite() { used } else { main_max };
        let content_cross = sizes.iter().map(|s| self.split(*s).1).fold(0.0f32, f32::max);
        let cross = if stretch { cross_max } else { content_cross.max(cross_min) };
        let size = c.constrain(self.join(main, cross));
        let (main, cross) = self.split(size);

        let free = (main - used).max(0.0);
        let (mut pos, between) = match self.main {
            MainAxis::Start => (0.0, self.gap),
            MainAxis::Center => (free * 0.5, self.gap),
            MainAxis::End => (free, self.gap),
            MainAxis::SpaceBetween if n > 1 => (0.0, self.gap + free / (n - 1) as f32),
            MainAxis::SpaceBetween => (0.0, self.gap),
        };
        for (i, s) in sizes.iter().enumerate() {
            let (child_main, child_cross) = self.split(*s);
            let cross_pos = match self.cross {
                CrossAxis::Start | CrossAxis::Stretch => 0.0,
                CrossAxis::Center => (cross - child_cross) * 0.5,
                CrossAxis::End => cross - child_cross,
            };
            let offset = match self.axis {
                Axis::Horizontal => vec2(pos, cross_pos),
                Axis::Vertical => vec2(cross_pos, pos),
            };
            cx.place_child(i, offset);
            pos += child_main + between;
        }
        size
    }
}

macro_rules! flex_widget {
    ($name:ident, $axis:expr, $role:expr, $doc:literal) => {
        #[doc = $doc]
        pub struct $name<M> {
            flex: Flex,
            _m: PhantomData<fn() -> M>,
        }

        impl<M: 'static> Default for $name<M> {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<M: 'static> $name<M> {
            pub fn new() -> Self {
                Self {
                    flex: Flex { axis: $axis, children: Vec::new(), gap: 0.0, main: MainAxis::Start, cross: CrossAxis::Start, shrink: false },
                    _m: PhantomData,
                }
            }
            pub fn child(mut self, child: impl Into<Element<M>>) -> Self {
                self.flex.children.push(child.into().into_any());
                self
            }
            pub fn children<E: Into<Element<M>>>(mut self, children: impl IntoIterator<Item = E>) -> Self {
                self.flex.children.extend(children.into_iter().map(|c| c.into().into_any()));
                self
            }
            /// Space between neighbouring children.
            pub fn gap(mut self, gap: f32) -> Self {
                self.flex.gap = gap;
                self
            }
            pub fn main_axis(mut self, main: MainAxis) -> Self {
                self.flex.main = main;
                self
            }
            pub fn cross_axis(mut self, cross: CrossAxis) -> Self {
                self.flex.cross = cross;
                self
            }
            /// Only as long as the children need, instead of all the space available.
            pub fn shrink(mut self) -> Self {
                self.flex.shrink = true;
                self
            }
        }

        impl<M: 'static> Widget for $name<M> {
            fn role(&self) -> Role {
                $role
            }
            fn children(&self) -> &[AnyElement] {
                &self.flex.children
            }
            fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size {
                self.flex.layout(cx, c)
            }
        }
    };
}

flex_widget!(Row, Axis::Horizontal, Role::Row, "Children side by side, left to right.");
flex_widget!(Column, Axis::Vertical, Role::Column, "Children one above the other, top to bottom.");

into_element!(Row, Column);
