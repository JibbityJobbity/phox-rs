use crate::math::{Mat4, Quat, Vec3, perspective, quat_look_at};

/// A perspective camera. With the identity orientation it looks down -Z with +Y up.
#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub orientation: Quat,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub near_z: f32,
    pub far_z: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { position: Vec3::new(0.0, 0.0, 2.0), orientation: Quat::IDENTITY, fov_y: 60f32.to_radians(), near_z: 0.1, far_z: 100.0 }
    }
}

impl Camera {
    /// Turns the camera to face `target`. `up` must not be parallel to the view direction.
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        self.orientation = quat_look_at((target - self.position).normalize(), up);
    }

    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::new(0.0, 0.0, -1.0)
    }

    /// The inverse of the camera's world transform (rotation, then translation).
    pub fn view(&self) -> Mat4 {
        Mat4::from_quat(self.orientation.conjugate()) * Mat4::from_translation(-self.position)
    }

    /// Projection into Vulkan clip space for a target with the given width / height.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        perspective(self.fov_y, aspect, self.near_z, self.far_z)
    }
}
