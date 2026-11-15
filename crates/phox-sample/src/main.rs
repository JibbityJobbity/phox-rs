//! `phox-sample`: renders a Phox sample scene, headless or in a window.
//!
//! Headless rendering goes through the forward technique on Vulkan (vertex shaders, or task and mesh shaders with
//! meshlet and occlusion culling), and falls back to the software rasterizer when there is no usable device.
//! Windowed mode and cluster LOD arrive with later phases: their options are parsed (so the interface is
//! final) and reported as ignored until then.

use phox_core::cpu_image::write_ppm;
use phox_core::mesh_loader::load_mesh;
use phox_core::render_view::extract_view;
use phox_core::resources::Extent2D;
use phox_core::sample_scenes::*;
use phox_core::scene::ToneMapOperator;
use phox_core::software_rasterizer::rasterize;
use phox_forward::{ForwardRenderer, RendererConfig};
use phox_vulkan::{VulkanConfig, VulkanDevice};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

/// An option that takes a value, or a flag.
struct Opt {
    name: &'static str,
    value: Option<&'static str>, // None: a flag; Some(default or ""): takes a value
    help: &'static str,
}

const OPTIONS: &[Opt] = &[
    Opt { name: "scene", value: Some("triangle"), help: "sample scene: triangle, cubes, tori, materials, field, crowd" },
    Opt { name: "mesh", value: Some(""), help: "show a model file on a turntable instead of a sample scene: glTF 2.0 (.gltf, .glb) with its materials, or OBJ" },
    Opt { name: "window", value: None, help: "present to a window until it is closed" },
    Opt { name: "frames", value: Some("0"), help: "with --window: stop after this many frames; headless: render this many frames (one image) and report average GPU times" },
    Opt { name: "time", value: Some("0"), help: "headless: the animation time to render, in seconds" },
    Opt { name: "animate", value: None, help: "headless with --frames: advance the animation by 1/60 s per frame instead of repeating one pose" },
    Opt { name: "size", value: Some("640x480"), help: "headless: image size" },
    Opt { name: "output", value: Some(""), help: "headless: where to write the image (default build/rendered_<scene>.ppm)" },
    Opt { name: "backend", value: Some("vulkan"), help: "GPU backend: vulkan or metal (Apple platforms)" },
    Opt { name: "debug", value: None, help: "enable validation (the Vulkan validation layer) and debug logging" },
    Opt { name: "device", value: Some(""), help: "use the GPU whose name contains this text" },
    Opt { name: "tonemap", value: Some("clamp"), help: "tone mapping: clamp, reinhard or aces" },
    Opt { name: "exposure", value: Some("1"), help: "scale scene colours by this before tone mapping" },
    Opt { name: "lod", value: Some("on"), help: "with mesh shaders: cluster LOD (on or off)" },
    Opt { name: "lod-error", value: Some("1"), help: "cluster LOD: the error, in pixels, the picture may show" },
    Opt { name: "stats", value: None, help: "print what each frame drew (triangles before and after cluster LOD) and its CPU time" },
    Opt { name: "mesh-shaders", value: Some("on"), help: "draw with task and mesh shaders where available: on, off, noocclusion (frustum and cone culling only) or nocull (no meshlet culling)" },
    Opt { name: "dump-graph", value: Some(""), help: "write the first frame's render graph (Graphviz DOT) to this file" },
];

enum CliError {
    /// A usage error: printed with the help, exit status 2.
    Usage(String),
    /// A failure while running: exit status 1.
    Failed(String),
}

impl From<phox_core::Error> for CliError {
    fn from(e: phox_core::Error) -> Self {
        CliError::Failed(e.to_string())
    }
}

fn help() -> String {
    let mut text = String::from("Renders a Phox sample scene, headless or in a window.\n\nUsage: phox-sample [OPTIONS]\n\nOptions:\n");
    for o in OPTIONS {
        let value = match o.value {
            Some("") => " <VALUE>".to_string(),
            Some(default) => format!(" <VALUE> (default: {default})"),
            None => String::new(),
        };
        text += &format!("      --{}{}\n          {}\n", o.name, value, o.help);
    }
    text + "  -h, --help\n          show this help\n"
}

struct Args {
    values: HashMap<&'static str, String>,
    flags: Vec<&'static str>,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Option<Args>, CliError> {
        let mut parsed = Args { values: HashMap::new(), flags: Vec::new() };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            if arg == "-h" || arg == "--help" {
                return Ok(None);
            }
            let Some(rest) = arg.strip_prefix("--") else {
                return Err(CliError::Usage(format!("unexpected argument '{arg}'")));
            };
            let (name, inline) = match rest.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (rest, None),
            };
            let opt = OPTIONS.iter().find(|o| o.name == name).ok_or_else(|| CliError::Usage(format!("unrecognised option '--{name}'")))?;
            if opt.value.is_none() {
                if inline.is_some() {
                    return Err(CliError::Usage(format!("option '--{name}' does not take a value")));
                }
                parsed.flags.push(opt.name);
            } else {
                let value = match inline {
                    Some(v) => v,
                    None => args.next().ok_or_else(|| CliError::Usage(format!("option '--{name}' requires an argument")))?,
                };
                parsed.values.insert(opt.name, value);
            }
        }
        Ok(Some(parsed))
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(&name)
    }

    fn value(&self, name: &str) -> &str {
        self.values.get(name).map(String::as_str).unwrap_or_else(|| OPTIONS.iter().find(|o| o.name == name).and_then(|o| o.value).unwrap_or(""))
    }

    fn has(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<T, CliError> {
        self.value(name).parse().map_err(|_| CliError::Usage(format!("option '--{name}' expects a number, got '{}'", self.value(name))))
    }
}

fn parse_tone_map(name: &str) -> Result<ToneMapOperator, CliError> {
    match name {
        "clamp" => Ok(ToneMapOperator::Clamp),
        "reinhard" => Ok(ToneMapOperator::Reinhard),
        "aces" => Ok(ToneMapOperator::Aces),
        _ => Err(CliError::Usage("--tonemap expects clamp, reinhard or aces".into())),
    }
}

fn parse_size(text: &str) -> Result<Extent2D, CliError> {
    let bad = || CliError::Usage("--size expects WIDTHxHEIGHT, e.g. 1280x720".into());
    let (w, h) = text.split_once('x').ok_or_else(bad)?;
    let (width, height): (u32, u32) = (w.parse().map_err(|_| bad())?, h.parse().map_err(|_| bad())?);
    if width == 0 || height == 0 {
        return Err(bad());
    }
    Ok(Extent2D::new(width, height))
}

fn scene_list() -> String {
    SAMPLE_SCENES.iter().map(|&id| sample_scene_name(id)).collect::<Vec<_>>().join(", ")
}

fn run(args: &Args) -> Result<i32, CliError> {
    let mut sample = if args.has("mesh") {
        let path = PathBuf::from(args.value("mesh"));
        let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if extension == "gltf" || extension == "glb" {
            make_gltf_viewer_scene(&path)?
        } else {
            make_mesh_viewer_scene(Arc::new(load_mesh(&path)?), path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())?
        }
    } else {
        let name = args.value("scene");
        let id = parse_sample_scene(name)
            .ok_or_else(|| CliError::Usage(format!("unknown sample scene '{name}' (try: {})", scene_list())))?;
        make_sample_scene(id)?
    };

    sample.scene.tone_mapping.op = parse_tone_map(args.value("tonemap"))?;
    sample.scene.tone_mapping.exposure = args.number("exposure")?;
    let extent = parse_size(args.value("size"))?;

    match args.value("backend") {
        "vulkan" | "metal" => {}
        _ => return Err(CliError::Usage("--backend expects vulkan or metal".into())),
    }
    if !matches!(args.value("mesh-shaders"), "on" | "off" | "noocclusion" | "nocull") {
        return Err(CliError::Usage("--mesh-shaders expects on, off, noocclusion or nocull".into()));
    }
    if !matches!(args.value("lod"), "on" | "off") {
        return Err(CliError::Usage("--lod expects on or off".into()));
    }
    let _lod_error: f32 = args.number("lod-error")?;
    let frames: i64 = args.number("frames")?;
    let time: f32 = args.number("time")?;

    if args.flag("debug") {
        phox_core::log_setup::enable_debug();
    }
    if args.flag("window") {
        return Err(CliError::Failed("--window needs the GPU renderer, which is not implemented yet".into()));
    }
    let output = if args.has("output") { PathBuf::from(args.value("output")) } else { PathBuf::from(format!("build/rendered_{}.ppm", sample.name)) };
    let frames = frames.max(1);
    let animate = args.flag("animate");
    // Without --animate every frame shows the same pose, so the GPU does the same work each time.
    (sample.animate)(&mut sample.scene, time);

    let gpu = match VulkanDevice::new(&VulkanConfig {
        validation: args.flag("debug"),
        preferred_device: args.value("device").to_string(),
        mesh_shaders: args.value("mesh-shaders") != "off",
        draw_indirect_count: true,
    }) {
        Ok(device) => Some(device),
        Err(e) => {
            log::warn!("no usable Vulkan device ({e}); using the software rasterizer");
            None
        }
    };
    let Some(device) = gpu else {
        println!("Rendering on the CPU with the software rasterizer");
        let mut image = None;
        for i in 0..frames {
            if animate {
                (sample.animate)(&mut sample.scene, time + i as f32 / 60.0);
            }
            image = Some(rasterize(&extract_view(&sample.scene, extent, true)));
        }
        write_ppm(&output, &image.expect("at least one frame is rendered"))?;
        println!("Wrote {} (software rasterizer)", output.display());
        return Ok(0);
    };

    let mode = args.value("mesh-shaders");
    let config = RendererConfig {
        extent,
        capture_graph: args.has("dump-graph"),
        timestamps: true,
        mesh_shaders: mode != "off",
        meshlet_culling: mode != "nocull",
        occlusion_culling: mode == "on",
    };
    let mut renderer = ForwardRenderer::new(Arc::new(device.clone()), config).map_err(|e| CliError::Failed(e.to_string()))?;
    println!(
        "Rendering on {} with {}",
        device.info().name,
        if !renderer.uses_mesh_shaders() {
            "vertex shaders"
        } else if renderer.uses_occlusion_culling() {
            "mesh shaders, frustum, cone and occlusion culling"
        } else if mode != "nocull" {
            "mesh shaders, frustum and cone culling"
        } else {
            "mesh shaders, no meshlet culling"
        }
    );
    // The first frames upload meshes and create pooled images, so they are left out of the averages.
    const WARMUP_FRAMES: i64 = 3;
    let mut timings = TimingAverages::default();
    let mut graph_written = false;
    let (mut cpu_ms, mut cpu_frames) = (0.0, 0);
    for i in 0..frames {
        if animate {
            (sample.animate)(&mut sample.scene, time + i as f32 / 60.0);
        }
        renderer.render(&sample.scene).map_err(|e| CliError::Failed(e.to_string()))?;
        if !graph_written && !renderer.frame_graph_dot().is_empty() {
            let path = PathBuf::from(args.value("dump-graph"));
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|e| CliError::Failed(e.to_string()))?;
            }
            std::fs::write(&path, renderer.frame_graph_dot()).map_err(|e| CliError::Failed(format!("cannot write {}: {e}", path.display())))?;
            println!("Wrote the frame graph to {}", path.display());
            graph_written = true;
        }
        if i >= WARMUP_FRAMES || frames <= WARMUP_FRAMES {
            timings.add(renderer.gpu_timings());
            cpu_ms += renderer.cpu_milliseconds();
            cpu_frames += 1;
        }
    }
    if frames > 1 {
        timings.print();
    }
    if args.flag("stats") {
        println!("CPU time per frame: {:.3} ms (recording and submitting)", cpu_ms / f64::from(cpu_frames.max(1)));
    }
    write_ppm(&output, renderer.last_frame())?;
    println!("Wrote {} ({})", output.display(), device.info().name);
    drop(renderer);
    let (errors, warnings) = device.validation_counts();
    if errors > 0 || warnings > 0 {
        eprintln!("Validation reported {errors} error(s) and {warnings} warning(s)");
        return Ok(3);
    }
    Ok(0)
}

/// Sums GPU pass times over frames, to print their averages.
#[derive(Default)]
struct TimingAverages {
    totals: Vec<(String, f64)>,
    frames: u32,
}

impl TimingAverages {
    fn add(&mut self, timings: &[phox_forward::PassTiming]) {
        if timings.is_empty() {
            return;
        }
        for t in timings {
            match self.totals.iter_mut().find(|(name, _)| *name == t.pass) {
                Some((_, total)) => *total += t.milliseconds,
                None => self.totals.push((t.pass.clone(), t.milliseconds)),
            }
        }
        self.frames += 1;
    }

    fn print(&self) {
        if self.frames == 0 {
            return;
        }
        let n = f64::from(self.frames);
        let mut line = format!("GPU time per frame, average of {} frame(s):", self.frames);
        for (pass, total) in &self.totals {
            line += &format!(" {pass} {:.3} ms,", total / n);
        }
        println!("{line} total {:.3} ms", self.totals.iter().map(|(_, t)| t).sum::<f64>() / n);
    }
}

fn main() -> ExitCode {
    phox_core::log_setup::init();
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print!("{}", help());
            return ExitCode::SUCCESS;
        }
        Err(CliError::Usage(message)) => {
            eprintln!("{message}\n\n{}", help());
            return ExitCode::from(2);
        }
        Err(CliError::Failed(message)) => {
            eprintln!("phox-sample failed: {message}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(status) => ExitCode::from(status as u8),
        Err(CliError::Usage(message)) => {
            eprintln!("{message}\n\n{}", help());
            ExitCode::from(2)
        }
        Err(CliError::Failed(message)) => {
            eprintln!("phox-sample failed: {message}");
            ExitCode::FAILURE
        }
    }
}
