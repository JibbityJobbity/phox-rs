use crate::math::{Mat3, Mat4, Vec3, Vec4};
use crate::meshlets::frustum_planes;
use crate::resources::Extent2D;
use crate::scene::{AlphaMode, Bounds, Material, Mesh, Scene, ToneMapping};

/// One mesh to draw with one material and transform.
#[derive(Clone, Debug)]
pub struct DrawItem<'a> {
    /// `Object::id()` of the object drawn.
    pub object_id: u64,
    pub mesh: &'a Mesh,
    pub material: &'a Material,
    pub model: Mat4,
    /// Distance of the mesh's bounds centre along the camera's forward axis.
    pub view_depth: f32,
    /// Back faces are culled: the material is single-sided and the transform does not mirror the mesh
    /// (which would turn its front faces into back faces). Otherwise both faces are drawn.
    pub cull_back_faces: bool,
}

/// What one frame draws from one camera, worked out from a Scene without any GPU involvement. Draw items
/// borrow from the scene, so a view cannot outlive it.
#[derive(Clone, Debug)]
pub struct RenderView<'a> {
    pub extent: Extent2D,
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub camera_position: Vec3,
    /// Unit vector towards the light.
    pub light_direction: Vec3,
    pub light_color: Vec3,
    pub ambient: f32,
    pub clear_color: Vec4,
    pub tone_mapping: ToneMapping,
    /// World space, from `view_proj` (see [`frustum_planes`]).
    pub frustum_planes: [Vec4; 6],
    /// Visible opaque and alpha-masked items, nearest first.
    pub draws: Vec<DrawItem<'a>>,
    /// Visible items with `AlphaMode::Blend`, farthest first: drawn after `draws`, blended, without depth writes.
    pub transparent: Vec<DrawItem<'a>>,
    /// Objects skipped because they are outside the view frustum.
    pub culled: usize,
}

impl RenderView<'_> {
    /// `draws.len() + transparent.len()`. Renderers number the items in that order (draws first), e.g. to
    /// index a per-frame material table.
    pub fn item_count(&self) -> usize {
        self.draws.len() + self.transparent.len()
    }
}

/// Whether a box, transformed by `mvp` into clip space, lies entirely outside one of the frustum planes.
pub fn outside_frustum(mvp: &Mat4, bounds: &Bounds) -> bool {
    let corners: [Vec4; 8] = std::array::from_fn(|i| {
        let corner = Vec3::new(
            if i & 1 != 0 { bounds.max.x } else { bounds.min.x },
            if i & 2 != 0 { bounds.max.y } else { bounds.min.y },
            if i & 4 != 0 { bounds.max.z } else { bounds.min.z },
        );
        *mvp * corner.extend(1.0)
    });
    // Clip space (Vulkan): -w <= x, y <= w and 0 <= z <= w.
    let all = |outside: fn(&Vec4) -> bool| corners.iter().all(outside);
    all(|c| c.x < -c.w)
        || all(|c| c.x > c.w)
        || all(|c| c.y < -c.w)
        || all(|c| c.y > c.w)
        || all(|c| c.z < 0.0)
        || all(|c| c.z > c.w)
}

/// The view without its items: camera matrices and frustum planes, the light, clear colour and tone
/// mapping. For renderers that keep the scene's objects themselves (the GPU-driven scene).
pub fn extract_camera(scene: &Scene, extent: Extent2D) -> RenderView<'_> {
    let camera = &scene.camera;
    let aspect = if extent.height > 0 { extent.width as f32 / extent.height as f32 } else { 1.0 };
    let view = camera.view();
    let proj = camera.projection(aspect);
    let view_proj = proj * view;
    let light = &scene.light;
    RenderView {
        extent,
        view,
        proj,
        view_proj,
        camera_position: camera.position,
        light_direction: if light.direction.length() > 0.0 { -light.direction.normalize() } else { Vec3::Z },
        light_color: light.color,
        ambient: light.ambient,
        clear_color: scene.clear_color,
        tone_mapping: scene.tone_mapping,
        frustum_planes: frustum_planes(&view_proj),
        draws: Vec::new(),
        transparent: Vec::new(),
        culled: 0,
    }
}

/// Builds the view for `scene` rendered at `extent`: camera matrices, the light, and the objects inside the
/// view frustum sorted front to back. Objects without a mesh, or with an empty one, are skipped. With `cull`
/// false, every object is listed and `draws` keep scene order; transparent items are still sorted back to
/// front, as blending needs.
pub fn extract_view(scene: &Scene, extent: Extent2D, cull: bool) -> RenderView<'_> {
    let mut view = extract_camera(scene, extent);
    let camera = &scene.camera;
    let forward = camera.forward();
    for object in scene.objects() {
        let Some(mesh) = object.mesh().map(|m| m.as_ref()).filter(|m| !m.is_empty()) else { continue };
        let model = object.transform().matrix();
        if cull && outside_frustum(&(view.view_proj * model), mesh.bounds()) {
            view.culled += 1;
            continue;
        }
        let center = (model * mesh.bounds().center().extend(1.0)).truncate();
        let cull_back_faces = !object.material().double_sided && Mat3::from_mat4(model).determinant() > 0.0;
        let item = DrawItem {
            object_id: object.id(),
            mesh,
            material: object.material(),
            model,
            view_depth: (center - camera.position).dot(forward),
            cull_back_faces,
        };
        if object.material().alpha_mode == AlphaMode::Blend {
            view.transparent.push(item);
        } else {
            view.draws.push(item);
        }
    }
    // Front to back, so depth testing rejects hidden fragments early. Stable: ties keep scene order.
    if cull {
        view.draws.sort_by(|a, b| a.view_depth.total_cmp(&b.view_depth));
    }
    // Back to front, so each blended item is composited over everything behind it.
    view.transparent.sort_by(|a, b| b.view_depth.total_cmp(&a.view_depth));
    view
}
