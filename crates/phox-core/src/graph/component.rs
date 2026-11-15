use super::*;
use phox_rhi::Capabilities;

/// A reusable piece of a graph with typed inputs and outputs.
///
/// `In` and `Out` are plain structs of [`GraphTexture`], [`GraphBuffer`], persistent `phox_rhi` handles and
/// plain values. A component may hold persistent rhi objects (pipelines, shaders): it creates them in an
/// `init` method of its own and frees them in `shutdown`, never inside `setup`. A component never inspects
/// device capabilities itself; it calls [`GraphBuilder::require`] and lets `compile()` fail.
pub trait Component {
    type In;
    type Out;

    fn name(&self) -> &str;

    /// Calls `b.require()` for each feature it needs, then adds one or more passes.
    fn setup<'g>(&mut self, b: &mut GraphBuilder<'g>, input: &Self::In) -> GraphResult<Self::Out>;
}

/// A whole frame built by chaining components. Applications that want a fallback pick another technique
/// themselves, using [`Technique::is_supported`].
pub trait Technique {
    fn name(&self) -> &str;

    /// The union of its components' requirements.
    fn requirements(&self) -> Vec<Feature>;

    /// All requirements present.
    fn is_supported(&self, caps: &Capabilities) -> bool {
        self.requirements().into_iter().all(|f| caps.has(f))
    }

    fn init(&mut self, device: &dyn Device) -> GraphResult<()>;
    fn shutdown(&mut self, device: &dyn Device);

    /// Adds this frame's passes; per-frame inputs are set through the implementor's own methods.
    fn build<'g>(&mut self, b: &mut GraphBuilder<'g>) -> GraphResult<()>;
}
