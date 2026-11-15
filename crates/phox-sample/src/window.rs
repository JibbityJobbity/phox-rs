//! Windowed rendering: a winit window, a Vulkan device that presents to it, and the forward renderer drawing
//! the sample scene, animated by elapsed time, until the window is closed (or `--frames` frames are shown).

use phox_core::sample_scenes::SampleScene;
use phox_forward::{ForwardRenderer, PassTiming, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// What a windowed run reports when it ends.
pub struct WindowReport {
    pub frames: u64,
    pub seconds: f64,
    pub timings: Vec<(String, f64)>,
    pub timed_frames: u64,
    pub validation: (u32, u32),
    pub device_name: String,
}

/// The run's state. Declared in drop order: the renderer goes before the device, the device before the window
/// its surface belongs to.
struct App {
    renderer: Option<ForwardRenderer>,
    device: Option<VulkanDevice>,
    window: Option<Arc<Window>>,
    sample: SampleScene,
    vulkan: VulkanConfig,
    config: RendererConfig,
    max_frames: u64,
    start: Instant,
    frames: u64,
    timings: Vec<(String, f64)>,
    timed_frames: u64,
    error: Option<String>,
}

const WARMUP_FRAMES: u64 = 3;

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let attributes = Window::default_attributes().with_title("Phox").with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attributes).map_err(|e| format!("cannot create a window: {e}"))?);
        let display = window.display_handle().map_err(|e| format!("no display handle: {e}"))?.as_raw();
        let handle = window.window_handle().map_err(|e| format!("no window handle: {e}"))?.as_raw();
        // SAFETY: `window` is kept in `self.window`, which is dropped after the device.
        let device = unsafe { VulkanDevice::new_for_window(&self.vulkan, display, handle) }.map_err(|e| format!("no Vulkan device can present to the window: {e}"))?;
        let renderer = ForwardRenderer::new(Arc::new(device.clone()), self.config.clone()).map_err(|e| e.to_string())?;
        self.device = Some(device);
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start = Instant::now();
        Ok(())
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(renderer), Some(window)) = (self.renderer.as_mut(), self.window.as_ref()) else { return };
        let size = window.inner_size();
        // Animate by elapsed time, not frame count, so the speed does not depend on the frame rate.
        (self.sample.animate)(&mut self.sample.scene, self.start.elapsed().as_secs_f32());
        match renderer.render_window(&self.sample.scene, size.width, size.height) {
            Ok(true) => {
                self.frames += 1;
                if self.frames > WARMUP_FRAMES && !renderer.gpu_timings().is_empty() {
                    for PassTiming { pass, milliseconds } in renderer.gpu_timings() {
                        match self.timings.iter_mut().find(|(name, _)| name == pass) {
                            Some((_, total)) => *total += milliseconds,
                            None => self.timings.push((pass.clone(), *milliseconds)),
                        }
                    }
                    self.timed_frames += 1;
                }
                if self.max_frames > 0 && self.frames >= self.max_frames {
                    event_loop.exit();
                }
            }
            Ok(false) => {} // minimized: nothing to draw into
            Err(e) => {
                self.error = Some(e.to_string());
                event_loop.exit();
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            if let Err(e) = self.init(event_loop) {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {} // a resize is noticed by the next acquire
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Shows `sample` in a window until it is closed, or for `max_frames` frames (0: until closed).
pub fn run(sample: SampleScene, vulkan: VulkanConfig, config: RendererConfig, max_frames: u64) -> Result<WindowReport, String> {
    let event_loop = EventLoop::new().map_err(|e| format!("cannot open a window system connection: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        renderer: None,
        device: None,
        window: None,
        sample,
        vulkan,
        config,
        max_frames,
        start: Instant::now(),
        frames: 0,
        timings: Vec::new(),
        timed_frames: 0,
        error: None,
    };
    event_loop.run_app(&mut app).map_err(|e| format!("the event loop failed: {e}"))?;
    if let Some(error) = app.error.take() {
        return Err(error);
    }
    let seconds = app.start.elapsed().as_secs_f64();
    let device = app.device.clone().ok_or("the window never opened")?;
    // Drop the renderer first: it frees its GPU objects through the device.
    app.renderer = None;
    Ok(WindowReport {
        frames: app.frames,
        seconds,
        timings: std::mem::take(&mut app.timings),
        timed_frames: app.timed_frames,
        validation: device.validation_counts(),
        device_name: device.info().name.clone(),
    })
}
