//! phox-ui as an nih-plug editor: [`PhoxEditor`] opens the plugin window (through `phox-ui-host`), [`Binding`]
//! links a panel to its plugin's parameters, and [`knob`], [`switch`] and [`choices`] are controls bound to a
//! parameter. [`Look`] is a plugin's palette and section style, and [`offscreen`] draws a panel to a PNG
//! without a host. macOS only for now, like the host.

#![cfg(target_os = "macos")]

mod editor;

pub use editor::PhoxEditor;
pub use phox_immediate;
pub use phox_ui;
pub use phox_ui_host::{Command, HostedApp};

/// A panel a plugin window can show (the name the plugins use for a [`HostedApp`]).
pub use phox_ui_host::HostedApp as PluginPanel;

use std::cell::Cell;
use std::sync::Arc;

use nih_plug::prelude::{BoolParam, Enum, EnumParam, FloatParam, GuiContext, Param, ParamPtr, Params};
use phox_ui::prelude::*;

pub const KNOB: f32 = 32.0;

#[derive(Clone)]
pub enum Msg {
    /// A parameter's plain value (an enum's index, or 0 / 1 for a switch).
    Plain(ParamPtr, f32),
    /// A parameter's normalized value.
    Norm(ParamPtr, f32),
}

/// A plugin's colours.
#[derive(Clone, Copy)]
pub struct Look {
    pub body: u32,
    pub panel: u32,
    pub screen: u32,
    pub ink: u32,
    pub accent: u32,
    /// Section stripes.
    pub stripes: [u32; 4],
}

impl Look {
    pub fn theme(&self) -> Theme {
        Theme {
            background: Color::hex(self.body),
            surface: Color::hex(0x3e4046),
            surface_hover: Color::hex(0x4b4e55),
            surface_pressed: Color::hex(0x26272b),
            accent: Color::hex(self.accent),
            on_accent: Color::hex(0x0d0e10),
            text_size: 8.0,
            control_height: 22.0,
            spacing: 6.0,
            ..Theme::default()
        }
    }

    /// A titled box with a coloured stripe.
    pub fn section(&self, title: &str, stripe: usize, body: impl Into<Element<Msg>>) -> Element<Msg> {
        Container::new()
            .padding(8.0)
            .radius(4.0)
            .background(Color::hex(self.panel))
            .child(
                Column::new()
                    .gap(6.0)
                    .child(Row::new().gap(6.0).cross_axis(CrossAxis::Center).child(Container::new().background(Color::hex(self.stripes[stripe % 4])).radius(1.0).child(SizedBox::new(14.0, 3.0))).child(Text::new(title).muted()))
                    .child(body),
            )
            .into()
    }

    /// The plugin's name over its stripes and a caption.
    pub fn logo(&self, name: &str, caption: &str) -> Element<Msg> {
        let mut stripes = Row::new().gap(3.0);
        for c in self.stripes {
            stripes = stripes.child(Container::new().background(Color::hex(c)).child(SizedBox::new(22.0, 6.0)));
        }
        Column::new().gap(6.0).child(Text::new(name).size(20.0)).child(Row::new().gap(8.0).cross_axis(CrossAxis::Center).child(stripes).child(Text::new(caption).muted())).into()
    }
}

/// A panel's link to its plugin: sends changes to the host and notices changes from it.
pub struct Binding {
    params: Arc<dyn Params>,
    context: Arc<dyn GuiContext>,
    /// A digest of every parameter value.
    signature: Cell<u64>,
}

impl Binding {
    pub fn new(params: Arc<dyn Params>, context: Arc<dyn GuiContext>) -> Self {
        let binding = Self { params, context, signature: Cell::new(0) };
        binding.signature.set(binding.digest());
        binding
    }

    fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for (_, ptr, _) in self.params.param_map() {
            // SAFETY: the pointers come from the plugin's own parameters, which outlive the editor.
            let v = unsafe { ptr.unmodulated_normalized_value() };
            h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Whether any parameter changed since the last call (from the host, or from the panel).
    pub fn changed(&self) -> bool {
        let d = self.digest();
        d != self.signature.replace(d)
    }

    /// Applies a control's message.
    pub fn apply(&self, message: Msg) {
        match message {
            Msg::Plain(ptr, plain) => {
                // SAFETY: as above.
                let normalized = unsafe { ptr.preview_normalized(plain) };
                self.set_normalized(ptr, normalized);
            }
            Msg::Norm(ptr, v) => self.set_normalized(ptr, v.clamp(0.0, 1.0)),
        }
    }

    fn set_normalized(&self, ptr: ParamPtr, normalized: f32) {
        // SAFETY: as above; the context outlives the editor too.
        unsafe {
            self.context.raw_begin_set_parameter(ptr);
            self.context.raw_set_parameter_normalized(ptr, normalized);
            self.context.raw_end_set_parameter(ptr);
        }
    }
}

/// A knob with its name and value under it.
pub fn knob(label: &str, p: &FloatParam) -> Element<Msg> {
    let ptr = p.as_ptr();
    Column::new()
        .gap(3.0)
        .cross_axis(CrossAxis::Center)
        .child(Knob::new(p.unmodulated_normalized_value(), 0.0, 1.0).diameter(KNOB).on_change(move |v| Msg::Norm(ptr, v)))
        .child(Text::new(label))
        .child(Text::new(p.to_string().to_uppercase()).muted())
        .into()
}

pub fn switch(label: &str, p: &BoolParam) -> Element<Msg> {
    let ptr = p.as_ptr();
    Toggle::new(p.value()).label(label).on_toggle(move |on| Msg::Plain(ptr, if on { 1.0 } else { 0.0 })).into()
}

fn choice_buttons<T: Enum + PartialEq + 'static>(p: &EnumParam<T>, width: f32) -> Vec<Element<Msg>> {
    let ptr = p.as_ptr();
    let current = p.value().to_index();
    T::variants()
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let b = Button::new(name.to_uppercase()).on_press(Msg::Plain(ptr, i as f32));
            SizedBox::width(width).child(if i == current { b.primary() } else { b }).into()
        })
        .collect()
}

/// A row of buttons, one per variant of an enum parameter, the current one lit.
pub fn choices<T: Enum + PartialEq + 'static>(p: &EnumParam<T>, width: f32) -> Element<Msg> {
    Row::new().gap(3.0).children(choice_buttons(p, width)).into()
}

/// The same, stacked.
pub fn choices_stacked<T: Enum + PartialEq + 'static>(p: &EnumParam<T>, width: f32) -> Element<Msg> {
    Column::new().gap(3.0).children(choice_buttons(p, width)).into()
}

pub fn row(items: Vec<Element<Msg>>) -> Element<Msg> {
    Row::new().gap(10.0).cross_axis(CrossAxis::Start).children(items).into()
}

/// A labelled line of choice buttons.
pub fn labelled(label: &str, body: Element<Msg>) -> Element<Msg> {
    Column::new().gap(4.0).child(Text::new(label).muted()).child(body).into()
}

/// For drawing a panel without a host (the plugins' `render_ui` examples).
pub mod offscreen {
    use std::sync::Arc;

    use nih_plug::prelude::{GuiContext, ParamPtr, PluginApi, PluginState};
    use phox_immediate::ImmediateRenderer;
    use phox_metal::{MetalConfig, MetalDevice};
    use phox_ui::testing::Tester;
    use phox_ui::{App, Size};

    /// A host that ignores parameter changes.
    pub struct NoHost;

    impl GuiContext for NoHost {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Standalone
        }
        fn request_resize(&self) -> bool {
            false
        }
        unsafe fn raw_begin_set_parameter(&self, _: ParamPtr) {}
        unsafe fn raw_set_parameter_normalized(&self, _: ParamPtr, _: f32) {}
        unsafe fn raw_end_set_parameter(&self, _: ParamPtr) {}
        fn get_state(&self) -> PluginState {
            unimplemented!("not needed to draw")
        }
        fn set_state(&self, _: PluginState) {}
    }

    pub fn context() -> Arc<dyn GuiContext> {
        Arc::new(NoHost)
    }

    /// Draws `app` at `size` (logical pixels, 2× scale) on Metal and writes it as a PNG.
    pub fn write_png<A: App>(app: A, (w, h): (f32, f32), path: &std::path::Path) {
        let device = MetalDevice::new(&MetalConfig::default()).expect("a Metal device");
        let mut renderer = ImmediateRenderer::new(Arc::new(device)).expect("the renderer");
        let mut t = Tester::new(app, Size::new(w, h));
        t.runtime().set_size(Size::new(w, h), 2.0);
        let (w, h, pixels) = t.render(&mut renderer).expect("render");
        let file = std::fs::File::create(path).expect("create the PNG");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().and_then(|mut wr| wr.write_image_data(&pixels)).expect("write the PNG");
        println!("wrote {}", path.display());
    }
}
