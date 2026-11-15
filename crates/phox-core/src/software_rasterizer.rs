use crate::cpu_image::CpuImage;
use crate::math::{Mat3, Vec2, Vec3, Vec4};
use crate::render_view::{DrawItem, RenderView};
use crate::scene::apply_tone_mapping;
use crate::shading::shade_material;

fn to_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The linear colour and depth of every pixel, before tone mapping (the GPU's HDR target and depth buffer).
struct Target {
    width: u32,
    height: u32,
    color: Vec<Vec3>,
    depth: Vec<f32>,
}

/// Rasterizes `draw` into `target`: opaque draws replace and write depth, blended ones (`blend`) are
/// composited over what is there and leave depth alone.
fn draw_item(target: &mut Target, view: &RenderView, draw: &DrawItem, blend: bool) {
    let (width, height) = (target.width as f32, target.height as f32);
    let mvp = view.view_proj * draw.model;
    let normal_matrix = Mat3::from_mat4(draw.model).inverse().transpose();
    let tangent_matrix = Mat3::from_mat4(draw.model);
    let vertices = draw.mesh.vertices();
    'triangles: for tri in draw.mesh.indices().chunks_exact(3) {
        let mut screen = [Vec3::ZERO; 3]; // framebuffer x, y and depth
        let mut positions = [Vec3::ZERO; 3]; // world space
        let mut normals = [Vec3::ZERO; 3];
        let mut tangents = [Vec4::ZERO; 3];
        let mut uvs = [Vec2::ZERO; 3];
        let mut inverse_w = [0.0f32; 3];
        for k in 0..3 {
            let v = &vertices[tri[k] as usize];
            let clip = mvp * v.position.extend(1.0);
            if clip.w <= 1e-5 {
                continue 'triangles; // no clipping: triangles crossing the near plane are skipped
            }
            let ndc = clip.truncate() / clip.w;
            screen[k] = Vec3::new((ndc.x * 0.5 + 0.5) * width, (ndc.y * 0.5 + 0.5) * height, ndc.z);
            positions[k] = (draw.model * v.position.extend(1.0)).truncate();
            normals[k] = normal_matrix * v.normal;
            tangents[k] = (tangent_matrix * v.tangent.truncate()).extend(v.tangent.w);
            uvs[k] = v.uv;
            inverse_w[k] = 1.0 / clip.w;
        }
        // Vulkan's facing rule: positive area in framebuffer coordinates is counter-clockwise.
        let area = -0.5
            * ((screen[0].x * screen[1].y - screen[1].x * screen[0].y)
                + (screen[1].x * screen[2].y - screen[2].x * screen[1].y)
                + (screen[2].x * screen[0].y - screen[0].x * screen[2].y));
        if area.abs() < 1e-8 || (area < 0.0 && draw.cull_back_faces) {
            continue;
        }
        let front_facing = area >= 0.0;

        let min_x = screen[0].x.min(screen[1].x).min(screen[2].x);
        let max_x = screen[0].x.max(screen[1].x).max(screen[2].x);
        let min_y = screen[0].y.min(screen[1].y).min(screen[2].y);
        let max_y = screen[0].y.max(screen[1].y).max(screen[2].y);
        let x0 = (min_x.floor() as i32).max(0);
        let x1 = (max_x.ceil() as i32).min(target.width as i32 - 1);
        let y0 = (min_y.floor() as i32).max(0);
        let y1 = (max_y.ceil() as i32).min(target.height as i32 - 1);
        let edge = |a: Vec3, b: Vec3, px: f32, py: f32| (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
        let total = edge(screen[0], screen[1], screen[2].x, screen[2].y);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let w0 = edge(screen[1], screen[2], px, py) / total;
                let w1 = edge(screen[2], screen[0], px, py) / total;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                // z/w is affine in screen space, so plain barycentric interpolation is exact.
                let z = w0 * screen[0].z + w1 * screen[1].z + w2 * screen[2].z;
                let index = y as usize * target.width as usize + x as usize;
                if !(0.0..=1.0).contains(&z) || z >= target.depth[index] {
                    continue;
                }
                // Attributes are interpolated perspective-correctly, as the GPU does.
                let persp = Vec3::new(w0 * inverse_w[0], w1 * inverse_w[1], w2 * inverse_w[2]);
                let b = persp / (persp.x + persp.y + persp.z);
                let fragment = shade_material(
                    draw.material,
                    b.x * positions[0] + b.y * positions[1] + b.z * positions[2],
                    b.x * normals[0] + b.y * normals[1] + b.z * normals[2],
                    b.x * tangents[0] + b.y * tangents[1] + b.z * tangents[2],
                    b.x * uvs[0] + b.y * uvs[1] + b.z * uvs[2],
                    front_facing,
                    view.camera_position,
                    view.light_direction,
                    view.light_color,
                    view.ambient,
                );
                if fragment.discarded {
                    continue;
                }
                if blend {
                    target.color[index] = target.color[index].lerp(fragment.color, fragment.alpha);
                } else {
                    target.depth[index] = z;
                    target.color[index] = fragment.color;
                }
            }
        }
    }
}

/// Renders a view on the CPU with the same projection, depth test, lighting and tone mapping as the GPU.
/// It exists so a frame can still be produced when no Vulkan device is available; it does no clipping
/// (triangles crossing the near plane are skipped) and follows the GPU only approximately on edges.
pub fn rasterize(view: &RenderView) -> CpuImage {
    let (width, height) = (view.extent.width, view.extent.height);
    let pixels = width as usize * height as usize;
    let mut target = Target { width, height, color: vec![view.clear_color.truncate(); pixels], depth: vec![1.0; pixels] };

    for draw in &view.draws {
        draw_item(&mut target, view, draw, false);
    }
    for draw in &view.transparent {
        draw_item(&mut target, view, draw, true);
    }

    // The GPU tonemaps the whole HDR image, background included, and writes opaque alpha.
    let mut rgba = Vec::with_capacity(pixels * 4);
    for color in &target.color {
        let c = apply_tone_mapping(*color, &view.tone_mapping);
        rgba.extend_from_slice(&[to_unorm8(c.x), to_unorm8(c.y), to_unorm8(c.z), 255]);
    }
    CpuImage { width, height, rgba }
}
