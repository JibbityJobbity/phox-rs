//! Rust mirrors of the data blocks the scene shaders read, as `#[repr(C)]` Pod structs with size
//! assertions. They match shaders/common.glsl; keep both in sync.

use crate::math::{Mat4, Vec4};
use crate::scene::{AlphaMode, Material};
use bytemuck::{Pod, Zeroable};

/// Per-frame data (set 0, binding 0). std140-compatible: only mat4 and vec4 members.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    /// xyz: world-space camera position.
    pub camera_position: Vec4,
    /// xyz: unit vector towards the light; w: ambient term.
    pub light_direction: Vec4,
    /// rgb.
    pub light_color: Vec4,
    /// xy: render target size in pixels; zw: 1 / size.
    pub resolution: Vec4,
    /// World space, as [`crate::meshlets::frustum_planes`].
    pub frustum_planes: [Vec4; 6],
}
const _: () = assert!(size_of::<FrameUniforms>() == 352);

/// Per-draw constants of the vertex shader path.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawPushConstants {
    pub model: Mat4,
    /// Index into the frame's `MaterialData` array.
    pub material: u32,
    pub padding: [u32; 3],
}
const _: () = assert!(size_of::<DrawPushConstants>() == 80);

/// `MaterialData` texture index of a texture the material does not have.
pub const NO_TEXTURE: u32 = 0xFFFF_FFFF;

/// `MaterialData::flags`: discard below `alpha_cutoff`.
pub const MATERIAL_ALPHA_MASK: u32 = 1 << 0;
/// `MaterialData::flags`: keep alpha for blending.
pub const MATERIAL_ALPHA_BLEND: u32 = 1 << 1;

/// One [`Material`] as the fragment shaders read it, from an array with one entry per draw of the frame
/// (std430). Texture fields index the backend's texture table, or are [`NO_TEXTURE`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MaterialData {
    pub base_color: Vec4,
    /// rgb; a unused.
    pub emissive: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
    pub flags: u32,
    pub base_color_texture: u32,
    pub metallic_roughness_texture: u32,
    pub normal_texture: u32,
    pub occlusion_texture: u32,
    pub emissive_texture: u32,
    pub padding: u32,
}
const _: () = assert!(size_of::<MaterialData>() == 80);

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: Vec4::ONE,
            emissive: Vec4::ZERO,
            metallic: 0.0,
            roughness: 0.5,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_cutoff: 0.5,
            flags: 0,
            base_color_texture: NO_TEXTURE,
            metallic_roughness_texture: NO_TEXTURE,
            normal_texture: NO_TEXTURE,
            occlusion_texture: NO_TEXTURE,
            emissive_texture: NO_TEXTURE,
            padding: 0,
        }
    }
}

/// Constants of the tonemap pass, fragment stage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TonemapPushConstants {
    pub exposure: f32,
    /// `ToneMapOperator` as an integer.
    pub op: u32,
    /// 1: apply the sRGB transfer function in the shader. 0 when the target's format encodes on write.
    pub encode_srgb: u32,
    pub padding: u32,
}
const _: () = assert!(size_of::<TonemapPushConstants>() == 16);

/// `material`'s factors and flags. Texture fields are left as [`NO_TEXTURE`] for the backend to fill in.
pub fn material_data(material: &Material) -> MaterialData {
    let flags = match material.alpha_mode {
        AlphaMode::Mask => MATERIAL_ALPHA_MASK,
        AlphaMode::Blend => MATERIAL_ALPHA_BLEND,
        AlphaMode::Opaque => 0,
    };
    MaterialData {
        base_color: material.albedo,
        emissive: material.emissive.extend(0.0),
        metallic: material.metallic,
        roughness: material.roughness,
        normal_scale: material.normal_scale,
        occlusion_strength: material.occlusion_strength,
        alpha_cutoff: material.alpha_cutoff,
        flags,
        ..MaterialData::default()
    }
}
