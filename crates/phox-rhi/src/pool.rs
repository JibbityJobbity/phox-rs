//! A slot pool whose handles carry a generation, for backends to store their resources in.

use crate::error::{Result, RhiError};
use crate::handles::Handle;

struct Slot<V> {
    value: Option<V>,
    generation: u32,
}

pub struct Pool<T, V> {
    kind: &'static str,
    slots: Vec<Slot<V>>,
    free: Vec<u32>,
    _tag: std::marker::PhantomData<fn() -> T>,
}

impl<T, V> Pool<T, V> {
    pub fn new(kind: &'static str) -> Self {
        Self { kind, slots: Vec::new(), free: Vec::new(), _tag: std::marker::PhantomData }
    }

    pub fn create(&mut self, value: V) -> Handle<T> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            Handle::new(index, slot.generation)
        } else {
            self.slots.push(Slot { value: Some(value), generation: 0 });
            Handle::new(self.slots.len() as u32 - 1, 0)
        }
    }

    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.slots.get(handle.index as usize).is_some_and(|s| s.generation == handle.generation && s.value.is_some())
    }

    pub fn get(&self, handle: Handle<T>) -> Result<&V> {
        self.slots
            .get(handle.index as usize)
            .filter(|s| s.generation == handle.generation)
            .and_then(|s| s.value.as_ref())
            .ok_or(RhiError::StaleHandle { kind: self.kind })
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Result<&mut V> {
        let kind = self.kind;
        self.slots
            .get_mut(handle.index as usize)
            .filter(|s| s.generation == handle.generation)
            .and_then(|s| s.value.as_mut())
            .ok_or(RhiError::StaleHandle { kind })
    }

    /// Removes and returns the value. An invalid or stale handle is a no-op (`None`).
    pub fn destroy(&mut self, handle: Handle<T>) -> Option<V> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation || slot.value.is_none() {
            return None;
        }
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        slot.value.take()
    }

    /// Every live value, for teardown.
    pub fn drain(&mut self) -> Vec<V> {
        self.free.clear();
        let mut out = Vec::new();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(v) = slot.value.take() {
                slot.generation = slot.generation.wrapping_add(1);
                out.push(v);
            }
            self.free.push(i as u32);
        }
        out
    }
}
