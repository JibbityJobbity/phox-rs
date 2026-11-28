//! SPIR-V to the Metal Shading Language, through SPIRV-Cross, with one fixed binding scheme for every shader
//! (so pipelines need no per-shader tables):
//!
//! | SPIR-V                          | Metal                                                   |
//! |---------------------------------|---------------------------------------------------------|
//! | `(set, binding)` buffer         | `[[buffer(set * 8 + binding)]]`, sets 0–2               |
//! | `(set, binding)` texture        | `[[texture(set * 8 + binding)]]` and `[[sampler(...)]]` |
//! | push constants                  | `[[buffer(24)]]`, written with `set*Bytes`              |
//! | vertex buffer 0                 | `[[buffer(30)]]`, through the vertex descriptor         |
//!
//! Vertex shaders are compiled with SPIR-V's y axis flipped, because Vulkan's clip space points y down and
//! Metal's points it up; the image comes out the same way up on both.

use phox_rhi::{Result, RhiError, ShaderStage};
use spirv_cross2::compile::msl::{BindTarget, CompilerOptions, MetalPlatform, MslVersion, ResourceBinding};
use spirv_cross2::reflect::{DecorationValue, ExecutionModeArguments};
use spirv_cross2::spirv::{Decoration, ExecutionMode, ExecutionModel};
use spirv_cross2::targets::Msl;
use spirv_cross2::{Compiler, Module};

/// Buffer slots each descriptor set gets.
pub(crate) const SLOTS_PER_SET: u32 = 8;
/// Descriptor sets that map to plain slots (0, 1 and 2).
pub(crate) const MAX_SETS: u32 = 3;
/// The buffer slot of the push constant block.
pub(crate) const PUSH_CONSTANT_SLOT: u64 = 24;
/// The buffer slot vertex buffer 0 is bound to.
pub(crate) const VERTEX_BUFFER_SLOT: u64 = 30;

/// The Metal slot of `(set, binding)`, for buffers, textures and samplers alike.
pub(crate) fn slot(set: u32, binding: u32) -> Option<u64> {
    (set < MAX_SETS && binding < SLOTS_PER_SET).then(|| u64::from(set * SLOTS_PER_SET + binding))
}

pub(crate) struct Translated {
    pub source: String,
    /// The entry point's name in the MSL source (SPIRV-Cross renames `main`, which MSL reserves).
    pub entry: String,
    /// The compute workgroup size, from the shader's `LocalSize`; `[1, 1, 1]` for other stages.
    pub workgroup: [u32; 3],
}

fn model(stage: ShaderStage) -> ExecutionModel {
    match stage {
        ShaderStage::Vertex => ExecutionModel::Vertex,
        ShaderStage::Fragment => ExecutionModel::Fragment,
        ShaderStage::Compute => ExecutionModel::GLCompute,
        ShaderStage::Task => ExecutionModel::TaskEXT,
        ShaderStage::Mesh => ExecutionModel::MeshEXT,
    }
}

fn cross(e: impl std::fmt::Display) -> RhiError {
    RhiError::Backend(format!("SPIRV-Cross: {e}"))
}

pub(crate) fn translate(spirv: &[u32], stage: ShaderStage, entry: &str) -> Result<Translated> {
    let model = model(stage);
    let mut compiler = Compiler::<Msl>::new(Module::from_words(spirv)).map_err(cross)?;
    compiler.set_entry_point(entry, model).map_err(cross)?;

    let resources = compiler.shader_resources().map_err(cross)?.all_resources().map_err(cross)?;
    let mut bindings = Vec::new();
    for resource in resources
        .uniform_buffers
        .iter()
        .chain(&resources.storage_buffers)
        .chain(&resources.sampled_images)
        .chain(&resources.storage_images)
        .chain(&resources.separate_images)
        .chain(&resources.separate_samplers)
    {
        let literal = |d| compiler.decoration(resource.id, d).ok().flatten().and_then(|v: DecorationValue| v.as_literal());
        let (Some(set), Some(binding)) = (literal(Decoration::DescriptorSet), literal(Decoration::Binding)) else {
            return Err(RhiError::InvalidDesc(format!("shader resource `{}` has no set and binding", resource.name)));
        };
        let index = slot(set, binding).ok_or_else(|| {
            RhiError::Unsupported(format!(
                "binding (set {set}, binding {binding}) on Metal, which maps sets 0-{} with {SLOTS_PER_SET} bindings each",
                MAX_SETS - 1
            ))
        })? as u32;
        bindings.push((set, binding, index));
    }
    for (set, binding, index) in bindings {
        let target = BindTarget { buffer: index, texture: index, sampler: index, count: None };
        compiler.add_resource_binding(model, ResourceBinding::Qualified { set, binding }, &target).map_err(cross)?;
    }
    if !resources.push_constant_buffers.is_empty() {
        let slot = PUSH_CONSTANT_SLOT as u32;
        let target = BindTarget { buffer: slot, texture: 0, sampler: 0, count: None };
        compiler.add_resource_binding(model, ResourceBinding::PushConstantBuffer, &target).map_err(cross)?;
    }

    let workgroup = match compiler.execution_mode_arguments(ExecutionMode::LocalSize).map_err(cross)? {
        Some(ExecutionModeArguments::LocalSize { x, y, z }) => [x, y, z],
        _ => [1, 1, 1],
    };

    let mut options = CompilerOptions::default();
    options.version = MslVersion::new(3, 0, 0);
    options.platform = MetalPlatform::MacOS;
    options.common.flip_vertex_y = matches!(stage, ShaderStage::Vertex | ShaderStage::Mesh);

    let artifact = compiler.compile(&options).map_err(cross)?;
    let entry = artifact
        .cleansed_entry_point_name(entry, model)
        .map_err(cross)?
        .map(|n| n.to_string())
        .ok_or_else(|| RhiError::InvalidDesc(format!("the shader has no {stage:?} entry point `{entry}`")))?;
    Ok(Translated { source: artifact.to_string(), entry, workgroup })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_follow_the_fixed_scheme() {
        assert_eq!(slot(0, 0), Some(0));
        assert_eq!(slot(1, 2), Some(10));
        assert_eq!(slot(2, 7), Some(23));
        assert_eq!(slot(3, 0), None);
        assert_eq!(slot(0, 8), None);
        assert!(slot(2, 7).unwrap() < PUSH_CONSTANT_SLOT);
    }
}
