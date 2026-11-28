//! Draws through a real device and checks the pixels: Metal on macOS, Vulkan elsewhere (`PHOX_BACKEND=metal`
//! or `vulkan` chooses). Skips without a device unless `PHOX_REQUIRE_GPU=1`; on Vulkan, `PHOX_VALIDATION=1`
//! fails on any validation message.

use std::sync::Arc;

use phox_immediate::{Color, DrawList, ImmediateRenderer, Rect, vec2};
use phox_rhi::Device;
use serial_test::serial;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// The device under test, and a check to run when the test is done.
struct TestDevice {
    device: Arc<dyn Device>,
    finish: Box<dyn Fn()>,
}

fn skip_or_panic(e: impl std::fmt::Display) -> Option<TestDevice> {
    assert!(!flag("PHOX_REQUIRE_GPU"), "no device: {e}");
    eprintln!("skipping: no device ({e})");
    None
}

fn backend() -> String {
    std::env::var("PHOX_BACKEND").unwrap_or_else(|_| if cfg!(target_os = "macos") { "metal".into() } else { "vulkan".into() })
}

fn device() -> Option<TestDevice> {
    match backend().as_str() {
        #[cfg(target_os = "macos")]
        "metal" => match phox_metal::MetalDevice::new(&phox_metal::MetalConfig { validation: true }) {
            Ok(d) => Some(TestDevice { device: Arc::new(d), finish: Box::new(|| {}) }),
            Err(e) => skip_or_panic(e),
        },
        "vulkan" => {
            use phox_vulkan::{VulkanConfig, VulkanDevice};
            let config = VulkanConfig { validation: flag("PHOX_VALIDATION"), mesh_shaders: false, ..VulkanConfig::default() };
            match VulkanDevice::new(&config) {
                Ok(d) => {
                    let check = d.clone();
                    let finish = Box::new(move || {
                        let (errors, warnings) = check.validation_counts();
                        assert_eq!((errors, warnings), (0, 0), "validation reported messages");
                    });
                    Some(TestDevice { device: Arc::new(d), finish })
                }
                Err(e) => skip_or_panic(e),
            }
        }
        other => panic!("PHOX_BACKEND={other}: expected metal or vulkan"),
    }
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

#[test]
#[serial]
fn shapes_land_where_they_were_drawn() {
    let Some(t) = device() else { return };
    let mut renderer = ImmediateRenderer::new(t.device.clone()).unwrap();
    let mut list = DrawList::new();
    // 64 x 48 logical at scale 2: a 128 x 96 image.
    list.begin(64.0, 48.0, 2.0);
    list.rect(Rect::new(8.0, 8.0, 16.0, 16.0), Color::hex(0xff0000));
    list.circle(vec2(48.0, 24.0), 10.0, Color::hex(0x00ff00));
    list.rect(Rect::new(8.0, 32.0, 16.0, 8.0), Color::hex(0x0000ff).with_alpha(0.5));
    let pixels = renderer.render_image(&list, 128, 96, Color::hex(0x000000)).unwrap();

    assert_eq!(pixel(&pixels, 128, 32, 32), [255, 0, 0, 255], "inside the red square");
    assert_eq!(pixel(&pixels, 128, 96, 48), [0, 255, 0, 255], "the green circle's centre");
    assert_eq!(pixel(&pixels, 128, 4, 4), [0, 0, 0, 255], "background");
    let blended = pixel(&pixels, 128, 32, 72);
    assert!((blended[2] as i32 - 128).abs() <= 2 && blended[0] == 0, "half-transparent blue over black: {blended:?}");
    (t.finish)();
}

#[test]
#[serial]
fn edges_are_anti_aliased() {
    let Some(t) = device() else { return };
    let mut renderer = ImmediateRenderer::new(t.device.clone()).unwrap();
    let mut list = DrawList::new();
    list.begin(32.0, 32.0, 1.0);
    // The right edge sits halfway across pixel column 20.
    list.rect(Rect::new(4.0, 4.0, 16.5, 24.0), Color::hex(0xffffff));
    let pixels = renderer.render_image(&list, 32, 32, Color::hex(0x000000)).unwrap();
    let edge = pixel(&pixels, 32, 20, 16)[0];
    assert!((100..=160).contains(&edge), "a half-covered pixel is about half bright: {edge}");
    assert_eq!(pixel(&pixels, 32, 19, 16)[0], 255);
    assert_eq!(pixel(&pixels, 32, 21, 16)[0], 0);
    (t.finish)();
}

#[test]
#[serial]
fn many_frames_reuse_the_slots_and_an_empty_list_just_clears() {
    let Some(t) = device() else { return };
    let mut renderer = ImmediateRenderer::new(t.device.clone()).unwrap();
    let mut list = DrawList::new();
    for frame in 0..6 {
        list.begin(16.0, 16.0, 1.0);
        // Growing geometry forces the slot buffers to be replaced.
        for i in 0..(frame * 2000) {
            list.rect(Rect::new((i % 16) as f32, 0.0, 1.0, 1.0), Color::hex(0x808080));
        }
        renderer.render_image(&list, 16, 16, Color::hex(0x204060)).unwrap();
    }
    list.begin(16.0, 16.0, 1.0);
    let pixels = renderer.render_image(&list, 16, 16, Color::hex(0x204060)).unwrap();
    assert_eq!(pixel(&pixels, 16, 8, 8), [0x20, 0x40, 0x60, 255]);
    (t.finish)();
}

#[test]
#[serial]
fn clips_cut_shapes_at_the_scissor() {
    let Some(t) = device() else { return };
    let mut renderer = ImmediateRenderer::new(t.device.clone()).unwrap();
    let mut list = DrawList::new();
    // 32 x 32 logical at scale 2: a 64 x 64 image. A full-size square clipped to the left half.
    list.begin(32.0, 32.0, 2.0);
    list.push_clip(Rect::new(0.0, 0.0, 16.0, 32.0));
    list.rect(Rect::new(0.0, 0.0, 32.0, 32.0), Color::hex(0xffffff));
    list.pop_clip();
    list.rect(Rect::new(24.0, 24.0, 4.0, 4.0), Color::hex(0xff0000)); // unclipped again
    let pixels = renderer.render_image(&list, 64, 64, Color::hex(0x000000)).unwrap();
    assert_eq!(pixel(&pixels, 64, 10, 32), [255, 255, 255, 255], "inside the clip");
    assert_eq!(pixel(&pixels, 64, 40, 10), [0, 0, 0, 255], "outside the clip");
    assert_eq!(pixel(&pixels, 64, 52, 52), [255, 0, 0, 255], "after pop_clip, drawing is unclipped");
    (t.finish)();
}
