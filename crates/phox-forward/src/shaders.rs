//! The SPIR-V of shaders/*, compiled by build.rs and embedded in the binary.

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

/// The SPIR-V words of the shader file `name` (for example `"forward.vert"`).
pub fn spirv(name: &str) -> Option<Vec<u32>> {
    let (_, bytes) = embedded::ALL.iter().find(|(n, _)| *n == name)?;
    Some(bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect())
}

/// The names of every embedded shader.
pub fn names() -> impl Iterator<Item = &'static str> {
    embedded::ALL.iter().map(|(n, _)| *n)
}
