//! Phox renders in linear light: material colours, light colours and the clear colour are linear, and
//! the tonemap pass encodes the result for the display with the sRGB transfer function. Colours picked
//! on screen or in an image editor are sRGB-encoded; convert them with [`srgb`] first.

use crate::math::{Vec3, Vec4};

/// The sRGB electro-optical transfer function: an encoded value in [0, 1] to linear light.
pub fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 { encoded / 12.92 } else { ((encoded + 0.055) / 1.055).powf(2.4) }
}

/// The inverse of [`srgb_to_linear`] for values in [0, 1].
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.003_130_8 { linear * 12.92 } else { 1.055 * linear.powf(1.0 / 2.4) - 0.055 }
}

pub fn srgb_to_linear3(c: Vec3) -> Vec3 {
    Vec3::new(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z))
}

pub fn linear_to_srgb3(c: Vec3) -> Vec3 {
    Vec3::new(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z))
}

/// A linear RGBA colour from sRGB-encoded components (alpha is linear already).
pub fn srgb(r: f32, g: f32, b: f32, a: f32) -> Vec4 {
    srgb_to_linear3(Vec3::new(r, g, b)).extend(a)
}

/// An opaque linear colour from sRGB-encoded components.
pub fn srgb3(r: f32, g: f32, b: f32) -> Vec4 {
    srgb(r, g, b, 1.0)
}
