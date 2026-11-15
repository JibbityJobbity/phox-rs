//! The forward technique: forward (and blended forward), tonemap, then a readback or presentation.

use crate::passes::*;
use phox_core::graph::*;
use phox_core::resources::Extent2D;
use phox_core::scene::ToneMapping;
use phox_rhi::*;
use std::sync::Arc;

/// Everything one frame's graph needs to be built, set by the renderer before [`Technique::build`].
#[derive(Clone, Debug)]
pub struct FrameInputs {
    pub forward: ForwardIn,
    pub mapping: ToneMapping,
    pub target_format: Format,
    /// Headless: where the tonemapped image is copied.
    pub readback: Option<BufferHandle>,
    /// Windowed: the swapchain image the tonemapped image is drawn into (ending in `Present`).
    pub backbuffer: Option<Backbuffer>,
}

/// Forward rendering into HDR, then tone mapping into the target.
pub struct ForwardTechnique {
    device: Arc<dyn Device>,
    forward: ForwardPass,
    tonemap: TonemapPass,
    readback: ReadbackPass,
    frame: Option<FrameInputs>,
    options: ForwardOptions,
}

impl ForwardTechnique {
    pub fn new(device: Arc<dyn Device>, options: ForwardOptions) -> Self {
        Self { device, forward: ForwardPass::default(), tonemap: TonemapPass::default(), readback: ReadbackPass, frame: None, options }
    }

    /// The inputs of the next `build`.
    pub fn set_frame(&mut self, frame: FrameInputs) {
        self.frame = Some(frame);
    }

    pub fn extent(&self) -> Option<Extent2D> {
        self.frame.as_ref().map(|f| f.forward.extent)
    }
}

impl Technique for ForwardTechnique {
    fn name(&self) -> &str {
        "forward"
    }

    fn requirements(&self) -> Vec<Feature> {
        vec![Feature::BindlessDescriptors]
    }

    fn init(&mut self, device: &dyn Device) -> GraphResult<()> {
        self.forward.init(device, self.options)?;
        self.tonemap.init(device).map_err(GraphError::from)?;
        Ok(())
    }

    fn shutdown(&mut self, device: &dyn Device) {
        self.forward.shutdown(device);
        self.tonemap.shutdown(device);
    }

    fn build<'g>(&mut self, b: &mut GraphBuilder<'g>) -> GraphResult<()> {
        let frame = self.frame.clone().ok_or_else(|| GraphError::Declaration("ForwardTechnique::build needs set_frame() first".into()))?;
        let forward = self.forward.setup(b, &frame.forward)?;
        let device = self.device.clone();
        let target = frame.backbuffer.map(|bb| b.import_texture(bb.texture, ResourceState::Undefined, ResourceState::Present)).transpose()?;
        let tonemapped = self.tonemap.add(
            b,
            device.as_ref(),
            &TonemapIn { hdr: forward.hdr, extent: frame.forward.extent, mapping: frame.mapping, target_format: frame.target_format, target },
        )?;
        if let Some(buffer) = frame.readback {
            self.readback.setup(b, &ReadbackIn { source: tonemapped.display, buffer })?;
        }
        Ok(())
    }
}
