//! RHI enums to Metal's.

use objc2_metal::*;
use phox_rhi::*;

pub(crate) fn pixel_format(format: Format) -> Option<MTLPixelFormat> {
    Some(match format {
        Format::Undefined => return None,
        Format::RGBA8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::RGBA8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::BGRA8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::BGRA8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::RGBA16Float => MTLPixelFormat::RGBA16Float,
        Format::RGBA32Float => MTLPixelFormat::RGBA32Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        // Apple GPUs have no 24-bit depth; the nearest format with stencil.
        Format::D24UnormS8Uint => MTLPixelFormat::Depth32Float_Stencil8,
    })
}

pub(crate) fn texture_usage(usage: TextureUsage) -> MTLTextureUsage {
    let mut out = MTLTextureUsage::Unknown;
    if usage.contains(TextureUsage::SAMPLED) {
        out |= MTLTextureUsage::ShaderRead;
    }
    if usage.contains(TextureUsage::STORAGE) {
        out |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
    }
    if usage.intersects(TextureUsage::COLOR_ATTACHMENT | TextureUsage::DEPTH_ATTACHMENT) {
        out |= MTLTextureUsage::RenderTarget;
    }
    out
}

pub(crate) fn resource_options(memory: MemoryKind) -> MTLResourceOptions {
    match memory {
        MemoryKind::GpuOnly => MTLResourceOptions::StorageModePrivate,
        MemoryKind::Upload => MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined,
        MemoryKind::Readback => MTLResourceOptions::StorageModeShared,
    }
}

pub(crate) fn vertex_format(format: VertexFormat) -> MTLVertexFormat {
    match format {
        VertexFormat::Float32 => MTLVertexFormat::Float,
        VertexFormat::Float32x2 => MTLVertexFormat::Float2,
        VertexFormat::Float32x3 => MTLVertexFormat::Float3,
        VertexFormat::Float32x4 => MTLVertexFormat::Float4,
        VertexFormat::UInt32 => MTLVertexFormat::UInt,
    }
}

pub(crate) fn primitive(topology: Topology) -> (MTLPrimitiveType, MTLPrimitiveTopologyClass) {
    match topology {
        Topology::TriangleList => (MTLPrimitiveType::Triangle, MTLPrimitiveTopologyClass::Triangle),
        Topology::LineList => (MTLPrimitiveType::Line, MTLPrimitiveTopologyClass::Line),
        Topology::PointList => (MTLPrimitiveType::Point, MTLPrimitiveTopologyClass::Point),
    }
}

pub(crate) fn cull(mode: CullMode) -> MTLCullMode {
    match mode {
        CullMode::None => MTLCullMode::None,
        CullMode::Back => MTLCullMode::Back,
        CullMode::Front => MTLCullMode::Front,
    }
}

pub(crate) fn compare(op: CompareOp) -> MTLCompareFunction {
    match op {
        CompareOp::Less => MTLCompareFunction::Less,
        CompareOp::LessEqual => MTLCompareFunction::LessEqual,
        CompareOp::Always => MTLCompareFunction::Always,
    }
}

pub(crate) fn filter(f: Filter) -> MTLSamplerMinMagFilter {
    match f {
        Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
        Filter::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

pub(crate) fn mip_filter(f: Filter) -> MTLSamplerMipFilter {
    match f {
        Filter::Nearest => MTLSamplerMipFilter::Nearest,
        Filter::Linear => MTLSamplerMipFilter::Linear,
    }
}

pub(crate) fn wrap(mode: WrapMode) -> MTLSamplerAddressMode {
    match mode {
        WrapMode::Repeat => MTLSamplerAddressMode::Repeat,
        WrapMode::MirroredRepeat => MTLSamplerAddressMode::MirrorRepeat,
        WrapMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
    }
}
