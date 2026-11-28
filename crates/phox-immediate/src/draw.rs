//! The draw list: shapes become triangles on the CPU as they are added, every frame.
//!
//! Anti-aliasing is geometric: each shape gets a fringe one physical pixel wide whose alpha falls to zero, so
//! edges are smooth without multisampling or textures. Coordinates are logical pixels with the origin at the
//! top left; colours are sRGB with straight alpha.

use crate::font;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

pub const fn vec2(x: f32, y: f32) -> Vec2 {
    Vec2 { x, y }
}

impl Vec2 {
    fn add(self, o: Vec2) -> Vec2 {
        vec2(self.x + o.x, self.y + o.y)
    }
    fn sub(self, o: Vec2) -> Vec2 {
        vec2(self.x - o.x, self.y - o.y)
    }
    fn scale(self, s: f32) -> Vec2 {
        vec2(self.x * s, self.y * s)
    }
    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    fn normalized(self) -> Vec2 {
        let len = self.length();
        if len > 0.0 { self.scale(1.0 / len) } else { vec2(0.0, 0.0) }
    }
    /// Rotated 90 degrees: in screen space (y down) this is the left-hand normal.
    fn perp(self) -> Vec2 {
        vec2(-self.y, self.x)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.x < self.x + self.w && p.y >= self.y && p.y < self.y + self.h
    }
    pub fn center(&self) -> Vec2 {
        vec2(self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
    pub fn inset(&self, by: f32) -> Rect {
        Rect::new(self.x + by, self.y + by, self.w - 2.0 * by, self.h - 2.0 * by)
    }
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
    /// The overlap of two rectangles; empty (zero size) when they do not overlap.
    pub fn intersect(&self, other: &Rect) -> Rect {
        let (x0, y0) = (self.x.max(other.x), self.y.max(other.y));
        let (x1, y1) = (self.right().min(other.right()), self.bottom().min(other.bottom()));
        Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
    }
    pub fn is_empty(&self) -> bool {
        self.w <= 0.0 || self.h <= 0.0
    }
}

/// sRGB colour with straight alpha, each channel 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    /// From `0xRRGGBB`.
    pub const fn hex(rgb: u32) -> Self {
        Self {
            r: ((rgb >> 16) & 0xff) as f32 / 255.0,
            g: ((rgb >> 8) & 0xff) as f32 / 255.0,
            b: (rgb & 0xff) as f32 / 255.0,
            a: 1.0,
        }
    }
    pub const fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }
    pub fn lerp(self, other: Color, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
    fn array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// A run of vertices drawn with one clip rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Batch {
    pub first_vertex: u32,
    pub vertex_count: u32,
    /// In logical pixels; `None` draws over the whole target.
    pub clip: Option<Rect>,
}

/// One frame's geometry. Reuse it across frames: [`DrawList::begin`] clears it and keeps its allocation.
#[derive(Clone, Debug)]
pub struct DrawList {
    vertices: Vec<Vertex>,
    /// Width of the anti-aliasing fringe: one physical pixel, in logical pixels.
    fringe: f32,
    scale: f32,
    size: Vec2,
    /// The clip stack; each entry is already intersected with the one below it.
    clips: Vec<Rect>,
    /// Where each clip change happened: (first vertex, clip from then on).
    clip_starts: Vec<(u32, Option<Rect>)>,
}

impl Default for DrawList {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawList {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), fringe: 1.0, scale: 1.0, size: vec2(1.0, 1.0), clips: Vec::new(), clip_starts: Vec::new() }
    }

    /// Starts a frame `width` x `height` logical pixels, shown at `scale` physical pixels per logical one.
    pub fn begin(&mut self, width: f32, height: f32, scale: f32) {
        self.vertices.clear();
        self.clips.clear();
        self.clip_starts.clear();
        self.scale = scale.max(0.01);
        self.fringe = 1.0 / self.scale;
        self.size = vec2(width, height);
    }

    /// Physical pixels per logical pixel, as given to [`DrawList::begin`].
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Clips everything drawn until the matching [`DrawList::pop_clip`] to `rect`, within any enclosing clip.
    pub fn push_clip(&mut self, rect: Rect) {
        let clip = match self.clips.last() {
            Some(outer) => outer.intersect(&rect),
            None => rect,
        };
        self.clips.push(clip);
        self.mark_clip();
    }

    /// Ends the innermost [`DrawList::push_clip`]. Unbalanced pops are ignored.
    pub fn pop_clip(&mut self) {
        if self.clips.pop().is_some() {
            self.mark_clip();
        }
    }

    /// The current clip, if any.
    pub fn clip(&self) -> Option<Rect> {
        self.clips.last().copied()
    }

    fn mark_clip(&mut self) {
        let start = self.vertices.len() as u32;
        let clip = self.clip();
        match self.clip_starts.last_mut() {
            // Nothing was drawn under the previous clip: replace it.
            Some(last) if last.0 == start => last.1 = clip,
            _ => self.clip_starts.push((start, clip)),
        }
    }

    /// The vertices split into runs that share a clip, in drawing order, without empty runs.
    pub fn batches(&self) -> Vec<Batch> {
        let total = self.vertices.len() as u32;
        let mut starts = Vec::with_capacity(self.clip_starts.len() + 1);
        if self.clip_starts.first().is_none_or(|s| s.0 > 0) {
            starts.push((0, None));
        }
        starts.extend(self.clip_starts.iter().copied());
        let mut out = Vec::new();
        for (i, &(first, clip)) in starts.iter().enumerate() {
            let end = starts.get(i + 1).map_or(total, |s| s.0);
            if end > first && clip.is_none_or(|c| !c.is_empty()) {
                out.push(Batch { first_vertex: first, vertex_count: end - first, clip });
            }
        }
        out
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// The logical size given to [`DrawList::begin`].
    pub fn size(&self) -> (f32, f32) {
        (self.size.x, self.size.y)
    }

    fn push(&mut self, p: Vec2, c: [f32; 4]) {
        self.vertices.push(Vertex { position: [p.x, p.y], color: c });
    }

    fn triangle(&mut self, a: (Vec2, [f32; 4]), b: (Vec2, [f32; 4]), c: (Vec2, [f32; 4])) {
        self.push(a.0, a.1);
        self.push(b.0, b.1);
        self.push(c.0, c.1);
    }

    fn quad(&mut self, a: (Vec2, [f32; 4]), b: (Vec2, [f32; 4]), c: (Vec2, [f32; 4]), d: (Vec2, [f32; 4])) {
        self.triangle(a, b, c);
        self.triangle(a, c, d);
    }

    /// A filled convex polygon with an anti-aliased edge. Either winding works.
    pub fn fill_convex(&mut self, points: &[Vec2], color: Color) {
        self.fill_convex_gradient(points, color, color, 0.0, 1.0);
    }

    /// As [`DrawList::fill_convex`], shaded vertically from `top` at `y0` to `bottom` at `y1`.
    pub fn fill_convex_gradient(&mut self, points: &[Vec2], top: Color, bottom: Color, y0: f32, y1: f32) {
        let n = points.len();
        if n < 3 {
            return;
        }
        let area: f32 = (0..n).map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        }).sum();
        // Outward normals point right of each edge for a clockwise (on screen) polygon, which has positive area.
        let outward = if area > 0.0 { -1.0 } else { 1.0 };
        let edge_normal = |i: usize| points[(i + 1) % n].sub(points[i]).normalized().perp().scale(outward);
        let half = self.fringe * 0.5;
        let shade = |p: Vec2| {
            let t = if y1 > y0 { ((p.y - y0) / (y1 - y0)).clamp(0.0, 1.0) } else { 0.0 };
            top.lerp(bottom, t).array()
        };

        // Each corner moves along the average of its two edge normals, lengthened so the edges move by `half`.
        let mut inner = Vec::with_capacity(n);
        let mut outer = Vec::with_capacity(n);
        for (i, &p) in points.iter().enumerate() {
            let (n0, n1) = (edge_normal((i + n - 1) % n), edge_normal(i));
            let mut m = n0.add(n1).scale(0.5);
            let len2 = (m.x * m.x + m.y * m.y).max(0.25);
            m = m.scale(1.0 / len2);
            inner.push(p.sub(m.scale(half)));
            outer.push(p.add(m.scale(half)));
        }
        for i in 1..n - 1 {
            self.triangle((inner[0], shade(points[0])), (inner[i], shade(points[i])), (inner[i + 1], shade(points[i + 1])));
        }
        for i in 0..n {
            let j = (i + 1) % n;
            let (ci, cj) = (shade(points[i]), shade(points[j]));
            let (ti, tj) = ([ci[0], ci[1], ci[2], 0.0], [cj[0], cj[1], cj[2], 0.0]);
            self.quad((inner[i], ci), (inner[j], cj), (outer[j], tj), (outer[i], ti));
        }
    }

    pub fn rect(&mut self, r: Rect, color: Color) {
        self.fill_convex(&rect_points(r), color);
    }

    pub fn rounded_rect(&mut self, r: Rect, radius: f32, color: Color) {
        self.rounded_rect_gradient(r, radius, color, color);
    }

    /// A rounded rectangle shaded from `top` to `bottom`.
    pub fn rounded_rect_gradient(&mut self, r: Rect, radius: f32, top: Color, bottom: Color) {
        let points = rounded_rect_points(r, radius);
        self.fill_convex_gradient(&points, top, bottom, r.y, r.y + r.h);
    }

    /// A rounded rectangle's outline, `thickness` wide, centred on the edge.
    pub fn rounded_rect_outline(&mut self, r: Rect, radius: f32, thickness: f32, color: Color) {
        let mut points = rounded_rect_points(r, radius);
        points.push(points[0]);
        self.polyline(&points, thickness, color);
    }

    pub fn circle(&mut self, center: Vec2, radius: f32, color: Color) {
        self.circle_gradient(center, radius, color, color);
    }

    pub fn circle_gradient(&mut self, center: Vec2, radius: f32, top: Color, bottom: Color) {
        let steps = arc_steps(radius, std::f32::consts::TAU).max(12);
        let points: Vec<Vec2> = (0..steps)
            .map(|i| {
                let a = i as f32 / steps as f32 * std::f32::consts::TAU;
                center.add(vec2(a.cos(), a.sin()).scale(radius))
            })
            .collect();
        self.fill_convex_gradient(&points, top, bottom, center.y - radius, center.y + radius);
    }

    /// A band `thickness` wide centred on a circle of `radius`, from angle `start` to `end` (radians, clockwise
    /// on screen from the +x axis).
    pub fn arc(&mut self, center: Vec2, radius: f32, thickness: f32, start: f32, end: f32, color: Color) {
        if (end - start).abs() < 1e-4 {
            return;
        }
        let (core, alpha) = self.core_half_width(thickness);
        let c = color.with_alpha(color.a * alpha).array();
        let clear = [c[0], c[1], c[2], 0.0];
        let f = self.fringe;
        let steps = arc_steps(radius + thickness, (end - start).abs()).max(2);
        let ring = |r: f32, a: f32| center.add(vec2(a.cos(), a.sin()).scale(r));
        for i in 0..steps {
            let (a0, a1) = (start + (end - start) * i as f32 / steps as f32, start + (end - start) * (i + 1) as f32 / steps as f32);
            let (ri, ro) = (radius - core, radius + core);
            if core > 0.0 {
                self.quad((ring(ri, a0), c), (ring(ri, a1), c), (ring(ro, a1), c), (ring(ro, a0), c));
            }
            self.quad((ring(ro, a0), c), (ring(ro, a1), c), (ring(ro + f, a1), clear), (ring(ro + f, a0), clear));
            self.quad((ring(ri - f, a0), clear), (ring(ri - f, a1), clear), (ring(ri, a1), c), (ring(ri, a0), c));
        }
    }

    /// Half the width of a stroke's opaque core, and the alpha to draw it with: strokes thinner than the fringe
    /// get no core and fade instead, so they keep their apparent weight.
    fn core_half_width(&self, thickness: f32) -> (f32, f32) {
        if thickness <= self.fringe { (0.0, thickness / self.fringe) } else { ((thickness - self.fringe) * 0.5, 1.0) }
    }

    /// A straight stroke with square ends extended by half the thickness, so polylines join without gaps.
    pub fn line(&mut self, a: Vec2, b: Vec2, thickness: f32, color: Color) {
        let d = b.sub(a).normalized();
        if d == vec2(0.0, 0.0) {
            return;
        }
        let (core, alpha) = self.core_half_width(thickness);
        let c = color.with_alpha(color.a * alpha).array();
        let clear = [c[0], c[1], c[2], 0.0];
        let ext = d.scale(thickness * 0.5);
        let (a, b) = (a.sub(ext), b.add(ext));
        let n = d.perp();
        let (nc, nf) = (n.scale(core), n.scale(core + self.fringe));
        if core > 0.0 {
            self.quad((a.add(nc), c), (b.add(nc), c), (b.sub(nc), c), (a.sub(nc), c));
        }
        self.quad((a.add(nc), c), (a.add(nf), clear), (b.add(nf), clear), (b.add(nc), c));
        self.quad((a.sub(nc), c), (b.sub(nc), c), (b.sub(nf), clear), (a.sub(nf), clear));
    }

    pub fn polyline(&mut self, points: &[Vec2], thickness: f32, color: Color) {
        for w in points.windows(2) {
            self.line(w[0], w[1], thickness, color);
        }
    }

    /// Draws `text` with its top-left corner at `origin`; `size` is the capital height.
    pub fn text(&mut self, origin: Vec2, size: f32, text: &str, color: Color) {
        let scale = size / font::CAP_HEIGHT;
        let thickness = (size * 0.13).max(self.fringe * 1.2);
        let mut x = origin.x;
        for ch in text.chars() {
            for stroke in font::glyph(ch) {
                let points: Vec<Vec2> = stroke.iter().map(|&(gx, gy)| vec2(x + gx * scale, origin.y + gy * scale)).collect();
                if points.len() == 1 {
                    self.circle(points[0], thickness * 0.6, color);
                } else {
                    self.polyline(&points, thickness, color);
                }
            }
            x += font::ADVANCE * scale;
        }
    }

    /// Text centred horizontally on `center_x`.
    pub fn text_centered(&mut self, center_x: f32, top: f32, size: f32, text: &str, color: Color) {
        let width = text_width(size, text);
        self.text(vec2(center_x - width * 0.5, top), size, text, color);
    }
}

/// Width of `text` drawn at `size`, without the trailing letter spacing.
pub fn text_width(size: f32, text: &str) -> f32 {
    let n = text.chars().count();
    if n == 0 {
        return 0.0;
    }
    let scale = size / font::CAP_HEIGHT;
    ((n - 1) as f32 * font::ADVANCE + font::GLYPH_WIDTH) * scale
}

fn rect_points(r: Rect) -> [Vec2; 4] {
    [vec2(r.x, r.y), vec2(r.x + r.w, r.y), vec2(r.x + r.w, r.y + r.h), vec2(r.x, r.y + r.h)]
}

fn rounded_rect_points(r: Rect, radius: f32) -> Vec<Vec2> {
    let radius = radius.min(r.w * 0.5).min(r.h * 0.5).max(0.0);
    if radius <= 0.0 {
        return rect_points(r).to_vec();
    }
    let steps = arc_steps(radius, std::f32::consts::FRAC_PI_2).max(2);
    let corners = [
        (vec2(r.x + r.w - radius, r.y + radius), -std::f32::consts::FRAC_PI_2),
        (vec2(r.x + r.w - radius, r.y + r.h - radius), 0.0),
        (vec2(r.x + radius, r.y + r.h - radius), std::f32::consts::FRAC_PI_2),
        (vec2(r.x + radius, r.y + radius), std::f32::consts::PI),
    ];
    let mut points = Vec::with_capacity(4 * (steps + 1));
    for (center, start) in corners {
        for i in 0..=steps {
            let a = start + std::f32::consts::FRAC_PI_2 * i as f32 / steps as f32;
            points.push(center.add(vec2(a.cos(), a.sin()).scale(radius)));
        }
    }
    points
}

/// Segments for an arc: about one per 2.5 logical pixels of arc length.
fn arc_steps(radius: f32, angle: f32) -> usize {
    ((radius * angle / 2.5).ceil() as usize).clamp(1, 256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_is_core_plus_fringe() {
        let mut list = DrawList::new();
        list.begin(100.0, 100.0, 1.0);
        list.rect(Rect::new(10.0, 10.0, 20.0, 20.0), Color::hex(0xff0000));
        // Two core triangles and four fringe quads.
        assert_eq!(list.vertices().len(), 2 * 3 + 4 * 6);
        let opaque = list.vertices().iter().filter(|v| v.color[3] == 1.0);
        for v in opaque {
            let [x, y] = v.position;
            assert!((10.0..=30.0).contains(&x) && (10.0..=30.0).contains(&y), "core vertex {x},{y} outside the rect");
        }
    }

    #[test]
    fn winding_does_not_matter() {
        let square = [vec2(0.0, 0.0), vec2(10.0, 0.0), vec2(10.0, 10.0), vec2(0.0, 10.0)];
        let mut reversed = square;
        reversed.reverse();
        let mut a = DrawList::new();
        a.begin(20.0, 20.0, 2.0);
        a.fill_convex(&square, Color::hex(0xffffff));
        let mut b = DrawList::new();
        b.begin(20.0, 20.0, 2.0);
        b.fill_convex(&reversed, Color::hex(0xffffff));
        let extent = |l: &DrawList| l.vertices().iter().fold(f32::MIN, |m, v| m.max(v.position[0]));
        assert_eq!(extent(&a), extent(&b));
        assert!((extent(&a) - 10.25).abs() < 1e-4, "the fringe extends half a physical pixel outside");
    }

    #[test]
    fn hairlines_fade_instead_of_thinning() {
        let mut list = DrawList::new();
        list.begin(10.0, 10.0, 1.0);
        list.line(vec2(0.0, 5.0), vec2(10.0, 5.0), 0.5, Color::hex(0xffffff));
        let max_alpha = list.vertices().iter().fold(0.0f32, |m, v| m.max(v.color[3]));
        assert!((max_alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn clips_nest_and_split_the_vertices_into_batches() {
        let mut list = DrawList::new();
        list.begin(100.0, 100.0, 2.0);
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::hex(0xffffff));
        let n = list.vertices().len() as u32;
        list.push_clip(Rect::new(10.0, 10.0, 50.0, 50.0));
        list.push_clip(Rect::new(40.0, 0.0, 50.0, 50.0)); // intersected with the outer clip
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::hex(0xffffff));
        list.pop_clip();
        list.pop_clip();
        list.pop_clip(); // unbalanced: ignored
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::hex(0xffffff));
        let batches = list.batches();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], Batch { first_vertex: 0, vertex_count: n, clip: None });
        assert_eq!(batches[1], Batch { first_vertex: n, vertex_count: n, clip: Some(Rect::new(40.0, 10.0, 20.0, 40.0)) });
        assert_eq!(batches[2], Batch { first_vertex: 2 * n, vertex_count: n, clip: None });
    }

    #[test]
    fn empty_clips_draw_nothing_and_clipless_lists_are_one_batch() {
        let mut list = DrawList::new();
        list.begin(100.0, 100.0, 1.0);
        assert!(list.batches().is_empty());
        list.push_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        list.push_clip(Rect::new(50.0, 50.0, 10.0, 10.0));
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::hex(0xffffff));
        assert!(list.batches().is_empty(), "disjoint clips leave nothing visible");
        list.begin(100.0, 100.0, 1.0);
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::hex(0xffffff));
        assert_eq!(list.batches().len(), 1);
    }

    #[test]
    fn text_width_counts_advances() {
        assert_eq!(text_width(6.0, ""), 0.0);
        let one = text_width(12.0, "A");
        let three = text_width(12.0, "ABC");
        assert!((three - one - 2.0 * font::ADVANCE * 2.0).abs() < 1e-4);
    }

    #[test]
    fn every_printable_ascii_character_has_a_glyph_or_is_blank() {
        for ch in ' '..='~' {
            let glyph = font::glyph(ch);
            if ch != ' ' {
                assert!(!glyph.is_empty(), "{ch:?} has no strokes");
            }
            for stroke in glyph {
                for &(x, y) in *stroke {
                    assert!((0.0..=font::GLYPH_WIDTH).contains(&x) && (-1.0..=font::CAP_HEIGHT + 2.0).contains(&y), "{ch:?}");
                }
            }
        }
    }
}
