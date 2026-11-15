//! Backend-free descriptions shared by the render graph, the device and the software path.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}
