//! RHI types to Vulkan types, and the one table that says what each [`ResourceState`] means to Vulkan.

use ash::vk;
use phox_rhi::*;

pub fn format(f: Format) -> vk::Format {
    match f {
        Format::Undefined => vk::Format::UNDEFINED,
        Format::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::RGBA8Srgb => vk::Format::R8G8B8A8_SRGB,
        Format::BGRA8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::BGRA8Srgb => vk::Format::B8G8R8A8_SRGB,
        Format::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
        Format::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::D32Float => vk::Format::D32_SFLOAT,
        Format::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
    }
}

pub fn aspect(f: Format) -> vk::ImageAspectFlags {
    match f {
        Format::D32Float => vk::ImageAspectFlags::DEPTH,
        Format::D24UnormS8Uint => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub fn image_usage(u: TextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    for (bit, vk_bit) in [
        (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
        (TextureUsage::STORAGE, vk::ImageUsageFlags::STORAGE),
        (TextureUsage::COLOR_ATTACHMENT, vk::ImageUsageFlags::COLOR_ATTACHMENT),
        (TextureUsage::DEPTH_ATTACHMENT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
        (TextureUsage::TRANSFER_SRC, vk::ImageUsageFlags::TRANSFER_SRC),
        (TextureUsage::TRANSFER_DST, vk::ImageUsageFlags::TRANSFER_DST),
    ] {
        if u.contains(bit) {
            flags |= vk_bit;
        }
    }
    flags
}

pub fn buffer_usage(u: BufferUsage, device_address: bool) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    for (bit, vk_bit) in [
        (BufferUsage::VERTEX, vk::BufferUsageFlags::VERTEX_BUFFER),
        (BufferUsage::INDEX, vk::BufferUsageFlags::INDEX_BUFFER),
        (BufferUsage::UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER),
        (BufferUsage::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
        (BufferUsage::INDIRECT, vk::BufferUsageFlags::INDIRECT_BUFFER),
        (BufferUsage::TRANSFER_SRC, vk::BufferUsageFlags::TRANSFER_SRC),
        (BufferUsage::TRANSFER_DST, vk::BufferUsageFlags::TRANSFER_DST),
    ] {
        if u.contains(bit) {
            flags |= vk_bit;
        }
    }
    if device_address {
        flags |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    flags
}

pub fn shader_stage(s: ShaderStage) -> vk::ShaderStageFlags {
    match s {
        ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
        ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
    }
}

pub fn shader_stages(s: ShaderStages) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    for (bit, stage) in [
        (ShaderStages::VERTEX, ShaderStage::Vertex),
        (ShaderStages::FRAGMENT, ShaderStage::Fragment),
        (ShaderStages::COMPUTE, ShaderStage::Compute),
        (ShaderStages::TASK, ShaderStage::Task),
        (ShaderStages::MESH, ShaderStage::Mesh),
    ] {
        if s.contains(bit) {
            flags |= shader_stage(stage);
        }
    }
    flags
}

pub fn vertex_format(f: VertexFormat) -> vk::Format {
    match f {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::UInt32 => vk::Format::R32_UINT,
    }
}

pub fn filter(f: Filter) -> vk::Filter {
    match f {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

pub fn wrap(w: WrapMode) -> vk::SamplerAddressMode {
    match w {
        WrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
        WrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        WrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    }
}

pub fn compare(op: CompareOp) -> vk::CompareOp {
    match op {
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}

/// What a texture in `state` needs: its layout, the stages that touch it and the accesses.
pub struct Access {
    pub layout: vk::ImageLayout,
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

/// The stages shaders run in (task and mesh only where the device has them).
pub fn shader_stage_mask(mesh_shaders: bool) -> vk::PipelineStageFlags2 {
    let mut stages = vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
    if mesh_shaders {
        stages |= vk::PipelineStageFlags2::TASK_SHADER_EXT | vk::PipelineStageFlags2::MESH_SHADER_EXT;
    }
    stages
}

/// The synchronization scope of a texture in `state`. A texture entering the graph (`Undefined`) has the
/// source scope ALL_COMMANDS / MEMORY_WRITE: whatever came before on the queue, including a previous user
/// of a pooled image.
pub fn texture_access(state: ResourceState, format: Format, mesh_shaders: bool) -> Access {
    let shaders = shader_stage_mask(mesh_shaders);
    let depth = format.is_depth();
    match state {
        ResourceState::Undefined => Access {
            layout: vk::ImageLayout::UNDEFINED,
            stages: vk::PipelineStageFlags2::ALL_COMMANDS,
            access: vk::AccessFlags2::MEMORY_WRITE,
        },
        ResourceState::ColorAttachment => Access {
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            stages: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        },
        ResourceState::DepthWrite => Access {
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            stages: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        },
        ResourceState::DepthRead => Access {
            layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            stages: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS | shaders,
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::SHADER_SAMPLED_READ,
        },
        ResourceState::ShaderRead | ResourceState::Uniform | ResourceState::VertexOrIndex | ResourceState::IndirectArgs => Access {
            layout: if depth { vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL } else { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL },
            stages: shaders,
            access: vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
        },
        ResourceState::ShaderWrite => Access {
            layout: vk::ImageLayout::GENERAL,
            stages: shaders,
            access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        },
        ResourceState::TransferSrc => Access {
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            stages: vk::PipelineStageFlags2::ALL_TRANSFER,
            access: vk::AccessFlags2::TRANSFER_READ,
        },
        ResourceState::TransferDst => Access {
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            stages: vk::PipelineStageFlags2::ALL_TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        },
        ResourceState::HostRead => Access {
            layout: vk::ImageLayout::GENERAL,
            stages: vk::PipelineStageFlags2::HOST,
            access: vk::AccessFlags2::HOST_READ,
        },
        ResourceState::Present => Access {
            layout: vk::ImageLayout::PRESENT_SRC_KHR,
            stages: vk::PipelineStageFlags2::NONE,
            access: vk::AccessFlags2::NONE,
        },
    }
}

/// The scope of a buffer in `state`.
pub fn buffer_access(state: ResourceState, mesh_shaders: bool) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let shaders = shader_stage_mask(mesh_shaders);
    match state {
        ResourceState::Undefined | ResourceState::ColorAttachment | ResourceState::DepthWrite | ResourceState::DepthRead | ResourceState::Present => {
            (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE)
        }
        ResourceState::ShaderRead => (
            shaders,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::UNIFORM_READ | vk::AccessFlags2::SHADER_SAMPLED_READ,
        ),
        ResourceState::Uniform => (shaders, vk::AccessFlags2::UNIFORM_READ),
        ResourceState::ShaderWrite => (shaders, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE),
        ResourceState::VertexOrIndex => (
            vk::PipelineStageFlags2::VERTEX_INPUT,
            vk::AccessFlags2::INDEX_READ | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        ),
        ResourceState::IndirectArgs => (
            vk::PipelineStageFlags2::DRAW_INDIRECT | shaders,
            vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
        ),
        ResourceState::TransferSrc => (vk::PipelineStageFlags2::ALL_TRANSFER, vk::AccessFlags2::TRANSFER_READ),
        ResourceState::TransferDst => (vk::PipelineStageFlags2::ALL_TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
        ResourceState::HostRead => (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
    }
}

/// The source side of a barrier keeps only write accesses (reads need no availability).
pub fn write_only(access: vk::AccessFlags2) -> vk::AccessFlags2 {
    access
        & (vk::AccessFlags2::MEMORY_WRITE
            | vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE
            | vk::AccessFlags2::HOST_WRITE)
}
