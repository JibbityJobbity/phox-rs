use crate::error::{Error, Result};
use std::io::Write;
use std::path::Path;

/// An 8-bit RGBA image in host memory: rows top to bottom, 4 bytes per pixel.
#[derive(Clone, Debug, Default)]
pub struct CpuImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl CpuImage {
    pub fn is_empty(&self) -> bool {
        self.rgba.is_empty()
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }
}

/// Writes a binary PPM (P6). Alpha is dropped. Creates parent directories.
pub fn write_ppm(path: &Path, image: &CpuImage) -> Result<()> {
    let wrap = |source| Error::Write { path: path.to_path_buf(), source };
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(wrap)?;
    }
    let mut out = Vec::with_capacity(image.width as usize * image.height as usize * 3 + 32);
    write!(out, "P6\n{} {}\n255\n", image.width, image.height).map_err(wrap)?;
    for px in image.rgba.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, out).map_err(wrap)
}

/// Reads a binary PPM (P6, maxval 255). Alpha is set to 255.
pub fn read_ppm(path: &Path) -> Result<CpuImage> {
    let data = std::fs::read(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })?;
    let bad = |message: &str| Error::Format { path: path.to_path_buf(), message: message.to_string() };
    let mut pos = 0usize;
    // The next whitespace-separated header token, skipping '#' comments.
    let token = |pos: &mut usize| -> String {
        let mut token = String::new();
        while *pos < data.len() {
            let c = data[*pos];
            *pos += 1;
            if c == b'#' {
                while *pos < data.len() && data[*pos] != b'\n' {
                    *pos += 1;
                }
            } else if c.is_ascii_whitespace() {
                if !token.is_empty() {
                    break;
                }
            } else {
                token.push(c as char);
            }
        }
        token
    };
    if token(&mut pos) != "P6" {
        return Err(bad("is not a binary PPM (P6)"));
    }
    let width: u32 = token(&mut pos).parse().map_err(|_| bad("has a bad width"))?;
    let height: u32 = token(&mut pos).parse().map_err(|_| bad("has a bad height"))?;
    if token(&mut pos) != "255" {
        return Err(bad("only maxval 255 is supported"));
    }
    let pixels = width as usize * height as usize;
    let rgb = data.get(pos..pos + pixels * 3).ok_or_else(|| bad("is truncated"))?;
    let mut rgba = Vec::with_capacity(pixels * 4);
    for px in rgb.chunks_exact(3) {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    Ok(CpuImage { width, height, rgba })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImageDiff {
    pub size_mismatch: bool,
    /// Largest |a - b| over the RGB channels of all pixels.
    pub max_channel_delta: u32,
    /// Pixels with any RGB channel differing by more than the tolerance.
    pub pixels_over_tolerance: usize,
}

/// Compares the RGB channels of two images.
pub fn compare_images(a: &CpuImage, b: &CpuImage, tolerance: u32) -> ImageDiff {
    let mut diff = ImageDiff::default();
    if a.width != b.width || a.height != b.height || a.rgba.len() != b.rgba.len() {
        diff.size_mismatch = true;
        return diff;
    }
    for (pa, pb) in a.rgba.chunks_exact(4).zip(b.rgba.chunks_exact(4)) {
        let worst = (0..3).map(|c| (pa[c] as i32 - pb[c] as i32).unsigned_abs()).max().unwrap_or(0);
        diff.max_channel_delta = diff.max_channel_delta.max(worst);
        if worst > tolerance {
            diff.pixels_over_tolerance += 1;
        }
    }
    diff
}
