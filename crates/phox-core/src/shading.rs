//! The metallic-roughness BRDF of glTF 2.0, mirrored from shaders/pbr.glsl so the software rasterizer
//! shades like the GPU.

use crate::math::{Vec2, Vec3, Vec4};
use crate::scene::{AlphaMode, Material};
use crate::texture::Texture;
use std::sync::Arc;

const PI: f32 = std::f32::consts::PI;

/// A material evaluated at one point: its factors times its textures.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSample {
    /// Linear.
    pub base_color: Vec3,
    pub metallic: f32,
    /// Perceptual.
    pub roughness: f32,
    /// 1: unoccluded.
    pub occlusion: f32,
    pub emissive: Vec3,
}

impl Default for SurfaceSample {
    fn default() -> Self {
        Self { base_color: Vec3::ONE, metallic: 0.0, roughness: 0.5, occlusion: 1.0, emissive: Vec3::ZERO }
    }
}

fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

fn visibility_smith_ggx(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let ggx_l = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    let sum = ggx_v + ggx_l;
    if sum > 0.0 { 0.5 / sum } else { 0.0 }
}

fn fresnel_schlick(f0: Vec3, v_dot_h: f32) -> Vec3 {
    let f = (1.0 - v_dot_h).powi(5);
    f0 + (Vec3::ONE - f0) * f
}

/// Outgoing radiance towards the eye. `n`, `v` and `l` are unit vectors: the shading normal, towards the
/// eye and towards the light. The light colour is irradiance / pi, so a white Lambertian surface facing a
/// white light reflects 1; `ambient` lights every surface evenly.
pub fn shade_pbr(s: &SurfaceSample, n: Vec3, v: Vec3, l: Vec3, light_color: Vec3, ambient: f32) -> Vec3 {
    let roughness = s.roughness.clamp(0.045, 1.0);
    let alpha = roughness * roughness;
    let f0 = Vec3::splat(0.04).lerp(s.base_color, s.metallic);
    let diffuse_color = s.base_color * (1.0 - s.metallic);

    let h = (v + l).normalize();
    let n_dot_l = n.dot(l).clamp(0.0, 1.0);
    let n_dot_v = n.dot(v).abs().max(1e-4);
    let n_dot_h = n.dot(h).clamp(0.0, 1.0);
    let v_dot_h = v.dot(h).clamp(0.0, 1.0);

    let f = fresnel_schlick(f0, v_dot_h);
    let specular = f * (distribution_ggx(n_dot_h, alpha) * visibility_smith_ggx(n_dot_l, n_dot_v, alpha));
    let diffuse = (Vec3::ONE - f) * diffuse_color / PI;
    let direct = (diffuse + specular) * light_color * (PI * n_dot_l);
    let indirect = ambient * light_color * (diffuse_color + f0) * s.occlusion;
    direct + indirect + s.emissive
}

/// What forward.frag computes for a material at one point: the lit colour and the alpha it writes.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShadedFragment {
    pub color: Vec3,
    pub alpha: f32,
    /// Below the alpha cutoff of an `AlphaMode::Mask` material.
    pub discarded: bool,
}

fn sample_or(texture: &Option<Arc<Texture>>, uv: Vec2, fallback: Vec4) -> Vec4 {
    texture.as_ref().map_or(fallback, |t| t.sample(uv))
}

/// What forward.frag computes for `material`, given the interpolated vertex attributes (world space).
/// Textures are sampled bilinearly from their first mip, where the GPU filters between mips.
#[allow(clippy::too_many_arguments)]
pub fn shade_material(
    m: &Material,
    position: Vec3,
    normal: Vec3,
    tangent: Vec4,
    uv: Vec2,
    front_facing: bool,
    camera_position: Vec3,
    light_direction: Vec3,
    light_color: Vec3,
    ambient: f32,
) -> ShadedFragment {
    let base_color = m.albedo * sample_or(&m.base_color_texture, uv, Vec4::ONE);
    if m.alpha_mode == AlphaMode::Mask && base_color.w < m.alpha_cutoff {
        return ShadedFragment { discarded: true, alpha: 1.0, ..Default::default() };
    }

    let mut normal = normal.normalize();
    if let Some(normal_texture) = &m.normal_texture {
        let mut t = tangent.truncate() - normal * normal.dot(tangent.truncate());
        let length2 = t.dot(t);
        if length2 > 1e-12 {
            t /= length2.sqrt();
            let b = normal.cross(t) * tangent.w;
            let mut mapped = normal_texture.sample(uv).truncate() * 2.0 - Vec3::ONE;
            mapped.x *= m.normal_scale;
            mapped.y *= m.normal_scale;
            normal = (mapped.x * t + mapped.y * b + mapped.z * normal).normalize();
        }
    }
    if !front_facing {
        normal = -normal;
    }

    let metallic_roughness = sample_or(&m.metallic_roughness_texture, uv, Vec4::ONE);
    let s = SurfaceSample {
        base_color: base_color.truncate(),
        metallic: m.metallic * metallic_roughness.z,
        roughness: m.roughness * metallic_roughness.y,
        occlusion: 1.0 + m.occlusion_strength * (sample_or(&m.occlusion_texture, uv, Vec4::ONE).x - 1.0),
        emissive: m.emissive * sample_or(&m.emissive_texture, uv, Vec4::ONE).truncate(),
    };

    let view = (camera_position - position).normalize();
    ShadedFragment {
        color: shade_pbr(&s, normal, view, light_direction, light_color, ambient),
        alpha: if m.alpha_mode == AlphaMode::Blend { base_color.w } else { 1.0 },
        discarded: false,
    }
}
