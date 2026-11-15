use crate::formats::{BufferUsage, Format, MemoryKind, ShaderStage, ShaderStages, TextureUsage};
use crate::handles::ShaderHandle;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: MemoryKind,
    pub debug_name: String,
    /// Shaders may read the buffer through its address instead of a descriptor.
    pub device_address: bool,
}

impl BufferDesc {
    pub fn new(size: u64, usage: BufferUsage, memory: MemoryKind, name: &str) -> Self {
        Self { size, usage, memory, debug_name: name.to_string(), device_address: false }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub mips: u32,
    pub format: Format,
    pub usage: TextureUsage,
    pub debug_name: String,
}

impl TextureDesc {
    pub fn new(width: u32, height: u32, format: Format, usage: TextureUsage, name: &str) -> Self {
        Self { width, height, mips: 1, format, usage, debug_name: name.to_string() }
    }
}

#[derive(Clone, Debug)]
pub struct ShaderDesc {
    pub stage: ShaderStage,
    pub spirv: Vec<u32>,
    pub entry: String,
}

impl ShaderDesc {
    pub fn new(stage: ShaderStage, spirv: Vec<u32>) -> Self {
        Self { stage, spirv, entry: "main".into() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    Nearest,
    #[default]
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum WrapMode {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

/// How a texture is sampled (glTF's sampler).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct SamplerDesc {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    /// Between mips; `Nearest` picks the closest mip.
    pub mip_filter: Filter,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    UInt32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
}

#[derive(Clone, Debug, Default)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Topology {
    #[default]
    TriangleList,
    LineList,
    PointList,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CullMode {
    #[default]
    None,
    Back,
    Front,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompareOp {
    #[default]
    Less,
    LessEqual,
    Always,
}

/// Source-alpha blending of a colour target: `src * a + dst * (1 - a)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendDesc {
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PushConstantRange {
    pub stages: ShaderStages,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingType {
    UniformBuffer,
    StorageBuffer,
    /// A texture with a sampler (the binding's `sampler`) fixed in the layout.
    SampledTexture,
    StorageTexture,
    /// An update-after-bind, partially bound array of sampled textures; `count` is its capacity. Slots
    /// are filled with [`crate::Device::register_bindless_texture`].
    BindlessTextures,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub set: u32,
    pub binding: u32,
    pub ty: BindingType,
    pub stages: ShaderStages,
    pub count: u32,
    pub sampler: SamplerDesc,
}

impl Binding {
    pub fn new(set: u32, binding: u32, ty: BindingType, stages: ShaderStages) -> Self {
        Self { set, binding, ty, stages, count: 1, sampler: SamplerDesc::default() }
    }
}

#[derive(Clone, Debug)]
pub struct GraphicsPipelineDesc {
    /// Exactly one Vertex + one Fragment, or an optional Task + a Mesh + a Fragment.
    pub shaders: Vec<ShaderHandle>,
    /// Ignored for mesh pipelines.
    pub vertex_layout: VertexLayout,
    pub topology: Topology,
    pub cull: CullMode,
    /// Dynamic rendering: formats, not a render pass.
    pub color_formats: Vec<Format>,
    pub depth_format: Format,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: CompareOp,
    pub blend: Option<BlendDesc>,
    pub bindings: Vec<Binding>,
    pub push_constants: Vec<PushConstantRange>,
    pub debug_name: String,
}

impl Default for GraphicsPipelineDesc {
    fn default() -> Self {
        Self {
            shaders: Vec::new(),
            vertex_layout: VertexLayout::default(),
            topology: Topology::TriangleList,
            cull: CullMode::None,
            color_formats: Vec::new(),
            depth_format: Format::Undefined,
            depth_test: false,
            depth_write: false,
            depth_compare: CompareOp::Less,
            blend: None,
            bindings: Vec::new(),
            push_constants: Vec::new(),
            debug_name: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ComputePipelineDesc {
    pub shader: ShaderHandle,
    pub bindings: Vec<Binding>,
    pub push_constants: Vec<PushConstantRange>,
    pub debug_name: String,
}
