#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Feature {
    MeshShader,
    RayTracing,
    BindlessDescriptors,
    /// `vkCmdDrawIndexedIndirectCount`: the draw count comes from a buffer. Optional (MoltenVK lacks it).
    DrawIndirectCount,
    ShaderFloat16,
    Atomic64,
    VariableRateShading,
    /// GPU timestamps ([`crate::CommandList::write_timestamp`]).
    Timestamps,
}

impl Feature {
    pub const fn name(self) -> &'static str {
        match self {
            Feature::MeshShader => "MeshShader",
            Feature::RayTracing => "RayTracing",
            Feature::BindlessDescriptors => "BindlessDescriptors",
            Feature::DrawIndirectCount => "DrawIndirectCount",
            Feature::ShaderFloat16 => "ShaderFloat16",
            Feature::Atomic64 => "Atomic64",
            Feature::VariableRateShading => "VariableRateShading",
            Feature::Timestamps => "Timestamps",
        }
    }
}

impl std::fmt::Display for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`Feature`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureSet(u32);

impl FeatureSet {
    pub fn insert(&mut self, feature: Feature) {
        self.0 |= 1 << feature as u32;
    }
    pub fn contains(&self, feature: Feature) -> bool {
        self.0 & (1 << feature as u32) != 0
    }
}

impl<const N: usize> From<[Feature; N]> for FeatureSet {
    fn from(features: [Feature; N]) -> Self {
        let mut set = FeatureSet::default();
        for f in features {
            set.insert(f);
        }
        set
    }
}

#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub device_name: String,
    pub api_major: u32,
    pub api_minor: u32,
    pub features: FeatureSet,
    pub max_texture_size_2d: u32,
    pub max_push_constant_bytes: u32,
    /// `maxDrawIndirectCount`: the largest draw count of one indirect call.
    pub max_draw_indirect_count: u32,
    /// Seconds-per-tick of GPU timestamps, in nanoseconds.
    pub timestamp_period_ns: f32,
    /// The size of the bindless texture table.
    pub bindless_texture_capacity: u32,
}

impl Capabilities {
    pub fn has(&self, feature: Feature) -> bool {
        self.features.contains(feature)
    }
}
