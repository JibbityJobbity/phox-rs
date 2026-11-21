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

/// An application's own work in a frame of the forward renderer ([`crate::ForwardRenderer::render_with`],
/// [`crate::ForwardRenderer::render_window_with`]): passes added to the frame's graph ahead of the forward
/// pass, such as compute that simulates something and builds triangles of it, and the triangles they made
/// for the forward pass to draw ([`GeneratedDraw`]), or task and mesh shaders that make them as they are drawn
/// ([`GeneratedTasks`]).
///
/// The passes are ordinary graph passes: they declare what they read and write, and buffers the extension
/// keeps across frames are imported each frame (from `Undefined`, which orders them after whatever the
/// queue did before). Create pipelines and buffers on the renderer's device, outside `build`.
pub trait FrameExtension {
    /// Adds this frame's passes and returns what the forward pass should draw.
    fn build(&mut self, b: &mut GraphBuilder<'_>) -> GraphResult<GeneratedGeometry>;
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

    /// [`Technique::build`], with `extension`'s passes first and its triangles drawn by the forward pass.
    pub fn build_with<'g>(&mut self, b: &mut GraphBuilder<'g>, extension: Option<&mut dyn FrameExtension>) -> GraphResult<()> {
        let mut frame = self.frame.clone().ok_or_else(|| GraphError::Declaration("ForwardTechnique::build needs set_frame() first".into()))?;
        if let Some(extension) = extension {
            let geometry = extension.build(b)?;
            frame.forward.generated = geometry.draws;
            frame.forward.generated_tasks = geometry.tasks;
        }
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
        self.build_with(b, None)
    }
}
