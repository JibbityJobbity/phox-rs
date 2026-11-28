//! A phox-ui app drawn on a real device (Metal on macOS, Vulkan elsewhere; `PHOX_BACKEND` chooses), checked
//! by its pixels. Skips without a device unless `PHOX_REQUIRE_GPU=1`. `PHOX_UI_SAVE=<dir>` writes the frame
//! there as a PNG, to look at.

use std::sync::Arc;

use phox_immediate::ImmediateRenderer;
use phox_rhi::Device;
use phox_ui::prelude::*;
use phox_ui::testing::{find, Tester};
use serial_test::serial;

fn device() -> Option<Arc<dyn Device>> {
    let require = std::env::var("PHOX_REQUIRE_GPU").is_ok_and(|v| v == "1");
    let backend = std::env::var("PHOX_BACKEND").unwrap_or_else(|_| if cfg!(target_os = "macos") { "metal".into() } else { "vulkan".into() });
    let result: Result<Arc<dyn Device>, String> = match backend.as_str() {
        #[cfg(target_os = "macos")]
        "metal" => phox_metal::MetalDevice::new(&phox_metal::MetalConfig { validation: true }).map(|d| Arc::new(d) as Arc<dyn Device>).map_err(|e| e.to_string()),
        "vulkan" => phox_vulkan::VulkanDevice::new(&phox_vulkan::VulkanConfig { mesh_shaders: false, ..Default::default() })
            .map(|d| Arc::new(d) as Arc<dyn Device>)
            .map_err(|e| e.to_string()),
        other => panic!("PHOX_BACKEND={other}: expected metal or vulkan"),
    };
    match result {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(!require, "no device: {e}");
            eprintln!("skipping: no device ({e})");
            None
        }
    }
}

/// A small effect-style panel: knobs, a slider, a toggle and a field, in a header and a body.
struct Panel {
    drive: f32,
    tone: f32,
    mix: f32,
    bypass: bool,
    preset: String,
}

#[derive(Clone)]
enum Msg {
    Drive(f32),
    Tone(f32),
    Mix(f32),
    Bypass(bool),
    Preset(String),
}

impl App for Panel {
    type Message = Msg;
    fn update(&mut self, m: Msg, _: &mut UpdateCx<Msg>) {
        match m {
            Msg::Drive(v) => self.drive = v,
            Msg::Tone(v) => self.tone = v,
            Msg::Mix(v) => self.mix = v,
            Msg::Bypass(v) => self.bypass = v,
            Msg::Preset(s) => self.preset = s,
        }
    }
    fn view(&self) -> Element<Msg> {
        let theme = Theme::default();
        let header = Container::new()
            .padding(Insets::symmetric(12.0, 8.0))
            .background(theme.surface)
            .child(
                Row::new()
                    .cross_axis(CrossAxis::Center)
                    .main_axis(MainAxis::SpaceBetween)
                    .child(Text::new("PHOX DRIVE").size(12.0))
                    .child(Toggle::new(self.bypass).label("BYPASS").on_toggle(Msg::Bypass)),
            );
        let knobs = Row::new()
            .gap(20.0)
            .main_axis(MainAxis::Center)
            .child(Element::from(Knob::new(self.drive, 0.0, 1.0).label("DRIVE").on_change(Msg::Drive)).key("drive"))
            .child(Knob::new(self.tone, 0.0, 1.0).label("TONE").on_change(Msg::Tone));
        let body = Container::new().padding(12.0).child(
            Column::new()
                .gap(12.0)
                .cross_axis(CrossAxis::Stretch)
                .child(knobs)
                .child(Row::new().gap(8.0).cross_axis(CrossAxis::Center).child(Text::new("MIX").muted()).child(Expanded::new(Slider::new(self.mix, 0.0, 1.0).on_change(Msg::Mix))))
                .child(TextField::new(self.preset.clone()).placeholder("PRESET NAME").on_change(Msg::Preset)),
        );
        Column::new().cross_axis(CrossAxis::Stretch).child(header).child(body).into()
    }
}

fn save_png(path: &std::path::Path, w: u32, h: u32, pixels: &[u8]) {
    let file = std::fs::File::create(path).expect("create the PNG");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().and_then(|mut w| w.write_image_data(pixels)).expect("write the PNG");
}

#[test]
#[serial]
fn a_panel_renders_on_the_gpu() {
    let Some(device) = device() else { return };
    let mut renderer = ImmediateRenderer::new(device).unwrap();
    let mut t = Tester::new(Panel { drive: 0.0, tone: 0.65, mix: 0.4, bypass: false, preset: String::new() }, Size::new(260.0, 200.0));
    t.runtime().set_size(Size::new(260.0, 200.0), 2.0);
    t.drag(find::key("drive"), vec2(0.0, -140.0)); // to 0.7
    t.tap(find::text("BYPASS"));
    assert!((t.app().drive - 0.7).abs() < 1e-3 && t.app().bypass);

    let (w, h, pixels) = t.render(&mut renderer).unwrap();
    assert_eq!((w, h), (520, 400));
    if let Ok(dir) = std::env::var("PHOX_UI_SAVE") {
        let path = std::path::Path::new(&dir).join("phox-ui-panel.png");
        save_png(&path, w, h, &pixels);
        eprintln!("saved {}", path.display());
    }

    let at = |x: f32, y: f32| {
        let (x, y) = ((x * 2.0) as usize, (y * 2.0) as usize);
        let i = (y * w as usize + x) * 4;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let rgb = |c: Color| [(c.r * 255.0).round() as u8, (c.g * 255.0).round() as u8, (c.b * 255.0).round() as u8];
    let theme = Theme::default();
    assert_eq!(at(4.0, 100.0), rgb(theme.background), "the body background");
    assert_eq!(at(3.0, 3.0), rgb(theme.surface), "the header");
    // The bypass switch is on: its track is the accent colour.
    let toggle = t.rect(find::text("BYPASS"));
    assert_eq!(at(toggle.x + 4.0, toggle.center().y), rgb(theme.accent), "the switch track");

}
