use crate::color::{linear_to_srgb3, srgb_to_linear};
use crate::error::{Error, Result};
use crate::math::{Vec2, Vec3, Vec4};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub use phox_rhi::{Filter as TextureFilter, SamplerDesc, WrapMode as TextureWrap};

/// What the texels of a texture mean, which decides how they are filtered and averaged into mips.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TextureEncoding {
    /// Colours stored with the sRGB transfer function (base colour, emissive); decoded when sampled.
    Srgb,
    /// Data (metallic-roughness, occlusion).
    Linear,
    /// A tangent-space normal map: xyz = rgb * 2 - 1; mips are renormalised.
    Normal,
}

fn next_texture_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn srgb_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

fn to_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Texel `i` (wrapping or clamping) along an axis of `size` texels.
fn wrap_index(i: i64, size: i64, wrap: TextureWrap) -> i64 {
    match wrap {
        TextureWrap::Repeat => i.rem_euclid(size),
        TextureWrap::MirroredRepeat => {
            let period = 2 * size;
            let m = i.rem_euclid(period);
            if m < size { m } else { period - 1 - m }
        }
        TextureWrap::ClampToEdge => i.clamp(0, size - 1),
    }
}

/// The number of mips of a full chain for a `width` x `height` image (1 x 1 included).
pub fn mip_count_for(width: u32, height: u32) -> u32 {
    let mut count = 1;
    let mut size = width.max(height);
    while size > 1 {
        count += 1;
        size /= 2;
    }
    count
}

#[derive(Clone, Debug)]
pub struct Mip {
    pub width: u32,
    pub height: u32,
    /// width * height * 4 bytes, rows top to bottom.
    pub rgba: Vec<u8>,
}

/// An immutable 2D image with 8-bit RGBA texels and a full mip chain, built on the CPU when the texture
/// is made so that every backend (and the software rasterizer) samples the same mips. Each texture has an
/// id that is never reused, so renderers can cache GPU copies of it. Cloning gives a new id.
#[derive(Debug)]
pub struct Texture {
    id: u64,
    name: String,
    encoding: TextureEncoding,
    sampler: SamplerDesc,
    mips: Vec<Mip>,
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        Self { id: next_texture_id(), name: self.name.clone(), encoding: self.encoding, sampler: self.sampler, mips: self.mips.clone() }
    }
}

impl Texture {
    /// `rgba` is width * height * 4 bytes, rows top to bottom. With `mipmaps` the chain runs down to
    /// 1 x 1; otherwise the texture has one mip.
    pub fn new(
        name: impl Into<String>,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        encoding: TextureEncoding,
        sampler: SamplerDesc,
        mipmaps: bool,
    ) -> Result<Self> {
        let name = name.into();
        if width == 0 || height == 0 {
            return Err(Error::EmptyTexture(name));
        }
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(Error::TextureSize { name, expected, got: rgba.len() });
        }
        let mut texture = Self { id: next_texture_id(), name, encoding, sampler, mips: vec![Mip { width, height, rgba }] };
        if mipmaps {
            texture.build_mips();
        }
        Ok(texture)
    }

    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn width(&self) -> u32 {
        self.mips[0].width
    }
    pub fn height(&self) -> u32 {
        self.mips[0].height
    }
    pub fn encoding(&self) -> TextureEncoding {
        self.encoding
    }
    pub fn sampler(&self) -> &SamplerDesc {
        &self.sampler
    }
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }
    pub fn mip(&self, level: u32) -> &Mip {
        &self.mips[level as usize]
    }
    pub fn byte_size(&self) -> usize {
        self.mips.iter().map(|m| m.rgba.len()).sum()
    }

    fn build_mips(&mut self) {
        let table = srgb_table();
        while self.mips.last().is_some_and(|m| m.width > 1 || m.height > 1) {
            let src = self.mips.last().expect("a mip exists");
            let dst_w = (src.width / 2).max(1);
            let dst_h = (src.height / 2).max(1);
            let mut dst = vec![0u8; dst_w as usize * dst_h as usize * 4];
            for y in 0..dst_h {
                for x in 0..dst_w {
                    // A 2 x 2 box; an axis of size 1 contributes the same texel twice.
                    let xs = [(2 * x).min(src.width - 1), (2 * x + 1).min(src.width - 1)];
                    let ys = [(2 * y).min(src.height - 1), (2 * y + 1).min(src.height - 1)];
                    let mut sum = Vec4::ZERO;
                    for sy in ys {
                        for sx in xs {
                            let i = (sy as usize * src.width as usize + sx as usize) * 4;
                            let t = &src.rgba[i..i + 4];
                            sum += if self.encoding == TextureEncoding::Srgb {
                                Vec4::new(table[t[0] as usize], table[t[1] as usize], table[t[2] as usize], t[3] as f32 / 255.0)
                            } else {
                                Vec4::new(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32) / 255.0
                            };
                        }
                    }
                    let mut texel = sum * 0.25;
                    match self.encoding {
                        TextureEncoding::Srgb => texel = linear_to_srgb3(texel.truncate()).extend(texel.w),
                        TextureEncoding::Normal => {
                            // Averaged normals are shorter than unit; renormalise so lighting does not darken with distance.
                            let mut n = texel.truncate() * 2.0 - Vec3::ONE;
                            let length = n.length();
                            n = if length > 1e-6 { n / length } else { Vec3::Z };
                            texel = (n * 0.5 + Vec3::splat(0.5)).extend(texel.w);
                        }
                        TextureEncoding::Linear => {}
                    }
                    let o = (y as usize * dst_w as usize + x as usize) * 4;
                    for c in 0..4 {
                        dst[o + c] = to_unorm8(texel[c]);
                    }
                }
            }
            self.mips.push(Mip { width: dst_w, height: dst_h, rgba: dst });
        }
    }

    /// Bilinear sample of mip 0 at `uv` with the sampler's wrap modes, as the GPU would return it: sRGB
    /// texels decoded to linear, the others as stored (in [0, 1]). For the software rasterizer.
    pub fn sample(&self, uv: Vec2) -> Vec4 {
        let mip = &self.mips[0];
        let table = srgb_table();
        let (w, h) = (mip.width as i64, mip.height as i64);
        let texel = |x: i64, y: i64| {
            let x = wrap_index(x, w, self.sampler.wrap_u);
            let y = wrap_index(y, h, self.sampler.wrap_v);
            let i = (y as usize * mip.width as usize + x as usize) * 4;
            let t = &mip.rgba[i..i + 4];
            if self.encoding == TextureEncoding::Srgb {
                Vec4::new(table[t[0] as usize], table[t[1] as usize], table[t[2] as usize], t[3] as f32 / 255.0)
            } else {
                Vec4::new(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32) / 255.0
            }
        };
        let fx = uv.x * w as f32;
        let fy = uv.y * h as f32;
        if self.sampler.mag_filter == TextureFilter::Nearest {
            return texel(fx.floor() as i64, fy.floor() as i64);
        }
        // Texel centres sit at half-integer coordinates.
        let x = fx - 0.5;
        let y = fy - 0.5;
        let (x0, y0) = (x.floor() as i64, y.floor() as i64);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let top = texel(x0, y0).lerp(texel(x0 + 1, y0), tx);
        let bottom = texel(x0, y0 + 1).lerp(texel(x0 + 1, y0 + 1), tx);
        top.lerp(bottom, ty)
    }

    /// A `size` x `size` checkerboard of `cells` x `cells` squares, for samples and tests.
    pub fn checker(
        name: impl Into<String>,
        size: u32,
        cells: u32,
        a: Vec4,
        b: Vec4,
        encoding: TextureEncoding,
        sampler: SamplerDesc,
    ) -> Result<Self> {
        let cells = cells.max(1);
        let encode = |c: Vec4| {
            let stored = if encoding == TextureEncoding::Srgb { linear_to_srgb3(c.truncate()).extend(c.w) } else { c };
            [to_unorm8(stored.x), to_unorm8(stored.y), to_unorm8(stored.z), to_unorm8(stored.w)]
        };
        let (first, second) = (encode(a), encode(b));
        let mut rgba = Vec::with_capacity(size as usize * size as usize * 4);
        for y in 0..size {
            for x in 0..size {
                let odd = ((x * cells / size) + (y * cells / size)) % 2 != 0;
                rgba.extend_from_slice(if odd { &second } else { &first });
            }
        }
        Self::new(name, size, size, rgba, encoding, sampler, true)
    }
}

