//! The nih-plug [`Editor`]: opens a `phox-ui-host` child view in the window the host gives the plugin.

use std::any::Any;
use std::sync::Arc;

use nih_plug::prelude::{Editor, GuiContext, ParentWindowHandle};
use phox_ui_host::HostedApp;

type MakePanel<P> = dyn Fn(Arc<dyn GuiContext>) -> P + Send + Sync;

pub struct PhoxEditor<P: HostedApp> {
    title: &'static str,
    size: (f32, f32),
    make: Arc<MakePanel<P>>,
}

impl<P: HostedApp> PhoxEditor<P> {
    /// An editor of `size` (logical pixels) whose panel `make` builds when the host opens the window.
    pub fn new(title: &'static str, size: (f32, f32), make: impl Fn(Arc<dyn GuiContext>) -> P + Send + Sync + 'static) -> Self {
        Self { title, size, make: Arc::new(make) }
    }
}

impl<P: HostedApp> Editor for PhoxEditor<P> {
    fn spawn(&self, parent: ParentWindowHandle, context: Arc<dyn GuiContext>) -> Box<dyn Any + Send> {
        let make = self.make.clone();
        Box::new(phox_ui_host::open_parented(&parent, self.title, self.size, move || make(context)))
    }

    fn size(&self) -> (u32, u32) {
        (self.size.0 as u32, self.size.1 as u32)
    }

    fn set_scale_factor(&self, _factor: f32) -> bool {
        // macOS reports the backing scale to the view itself; the panel's logical size never changes.
        false
    }

    // Panels compare the parameters every frame (`HostedApp::changed` or `poll`) and redraw when they change.
    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {}
    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {}
    fn param_values_changed(&self) {}
}
