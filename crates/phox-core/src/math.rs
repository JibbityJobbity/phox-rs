//! Phox's vector and matrix types, the equivalent of phox/Math.hpp: right-handed, column-major, and
//! Vulkan clip space (depth in [0, 1], y pointing down). Everything uses glam through this module.

pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

pub type Vec2f = Vec2;

/// Right-handed perspective projection into Vulkan clip space: depth 0 at `near` and 1 at `far`, and +y in
/// view space pointing up the screen (Vulkan's clip-space y points down, so it is flipped).
/// `fov_y` is the vertical field of view in radians.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let mut projection = Mat4::perspective_rh(fov_y, aspect, near, far);
    projection.y_axis.y *= -1.0;
    projection
}

/// A quaternion looking along `direction` (unit) with `up`, as glm::quatLookAtRH: the camera's -Z axis
/// ends up along `direction`.
pub fn quat_look_at(direction: Vec3, up: Vec3) -> Quat {
    let z = -direction;
    let x = up.cross(z).normalize();
    let y = z.cross(x);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

pub const PI: f32 = std::f32::consts::PI;
pub const TWO_PI: f32 = std::f32::consts::TAU;
pub const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;
