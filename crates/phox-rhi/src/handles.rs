use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// A typed index into a device's resource pool. Pools bump the generation when a slot is destroyed, so a
/// stale handle never resolves again, even after the slot is reused.
pub struct Handle<T> {
    pub index: u32,
    pub generation: u32,
    _tag: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0, _tag: PhantomData };

    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation, _tag: PhantomData }
    }

    pub const fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self::INVALID
    }
}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() { write!(f, "Handle({}v{})", self.index, self.generation) } else { write!(f, "Handle(invalid)") }
    }
}

pub enum BufferTag {}
pub enum TextureTag {}
pub enum ShaderTag {}
pub enum PipelineTag {}

pub type BufferHandle = Handle<BufferTag>;
pub type TextureHandle = Handle<TextureTag>;
pub type ShaderHandle = Handle<ShaderTag>;
pub type PipelineHandle = Handle<PipelineTag>;

/// A slot in the device's bindless texture table (see [`crate::Device::register_bindless_texture`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindlessIndex(pub u32);
