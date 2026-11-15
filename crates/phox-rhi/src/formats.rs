use crate::bitflags::flags;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Format {
    #[default]
    Undefined,
    RGBA8Unorm,
    RGBA8Srgb,
    BGRA8Unorm,
    BGRA8Srgb,
    RGBA16Float,
    RGBA32Float,
    R32Float,
    D32Float,
    D24UnormS8Uint,
}

impl Format {
    pub const fn is_depth(self) -> bool {
        matches!(self, Format::D32Float | Format::D24UnormS8Uint)
    }

    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Format::Undefined => 0,
            Format::RGBA16Float => 8,
            Format::RGBA32Float => 16,
            _ => 4,
        }
    }
}

flags!(TextureUsage {
    SAMPLED = 1,
    STORAGE = 2,
    COLOR_ATTACHMENT = 4,
    DEPTH_ATTACHMENT = 8,
    TRANSFER_SRC = 16,
    TRANSFER_DST = 32,
});

flags!(BufferUsage {
    VERTEX = 1,
    INDEX = 2,
    UNIFORM = 4,
    STORAGE = 8,
    INDIRECT = 16,
    TRANSFER_SRC = 32,
    TRANSFER_DST = 64,
});

/// Where a buffer lives. `Upload` is written by the CPU and read by the GPU; `Readback` the reverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MemoryKind {
    #[default]
    GpuOnly,
    Upload,
    Readback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Task,
    Mesh,
}

flags!(ShaderStages {
    VERTEX = 1,
    FRAGMENT = 2,
    COMPUTE = 4,
    TASK = 8,
    MESH = 16,
});

impl ShaderStages {
    pub const ALL_GRAPHICS: Self = Self(1 | 2 | 8 | 16);
}

impl From<ShaderStage> for ShaderStages {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
            ShaderStage::Task => Self::TASK,
            ShaderStage::Mesh => Self::MESH,
        }
    }
}
