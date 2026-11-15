//! The `phox-sample` binary end to end: options, exit codes, the files it writes, the software fallback and
//! (opt in) a window.
//!
//! Honours PHOX_VALIDATION-style behaviour through `--debug`; the windowed test only runs with
//! PHOX_TEST_WINDOW=1 (CI sets it under xvfb-run), so a developer's own desktop is left alone.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_phox-sample"));
    command.args(args);
    for (k, v) in env {
        command.env(k, v);
    }
    command.output().expect("the binary runs")
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("phox_rs_cli_{}_{name}", std::process::id()))
}

fn ppm_size(path: &Path) -> (u32, u32) {
    let image = phox_core::cpu_image::read_ppm(path).expect("a readable PPM");
    (image.width, image.height)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

#[test]
fn help_lists_every_option() {
    let out = run(&["--help"], &[]);
    assert!(out.status.success());
    let help = text(&out.stdout);
    for option in [
        "--scene", "--mesh", "--window", "--frames", "--time", "--animate", "--size", "--output", "--backend", "--debug", "--device", "--tonemap",
        "--exposure", "--lod", "--lod-error", "--stats", "--mesh-shaders", "--dump-graph",
    ] {
        assert!(help.contains(option), "--help lacks {option}");
    }
}

#[test]
fn usage_errors_exit_with_status_2() {
    for args in [&["--scene", "nope"][..], &["--tonemap", "sepia"], &["--size", "10"], &["--lod", "maybe"], &["--no-such-option"], &["--mesh-shaders", "x"]] {
        let out = run(args, &[]);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", text(&out.stderr));
    }
}

#[test]
fn the_metal_backend_is_refused_in_this_build() {
    let out = run(&["--backend", "metal"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("no metal backend"), "{}", text(&out.stderr));
}

#[test]
fn renders_the_triangle_headless_and_validates() {
    let output = temp("triangle.ppm");
    let out = run(&["--debug", "--size", "96x64", "--output", output.to_str().unwrap()], &[]);
    let stdout = text(&out.stdout);
    // Exit status 3 would be validation messages; 0 on a GPU or on the software fallback.
    assert_eq!(out.status.code(), Some(0), "{stdout}\n{}", text(&out.stderr));
    assert_eq!(ppm_size(&output), (96, 64));
    std::fs::remove_file(&output).unwrap();
}

#[test]
fn falls_back_to_the_software_rasterizer_without_a_vulkan_driver() {
    let output = temp("software.ppm");
    let out = run(&["--scene", "cubes", "--size", "80x60", "--output", output.to_str().unwrap()], &[("VK_DRIVER_FILES", "/nonexistent/icd.json")]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("software rasterizer"), "{}", text(&out.stdout));
    assert_eq!(ppm_size(&output), (80, 60));
    std::fs::remove_file(&output).unwrap();
}

#[test]
fn the_gpu_and_the_software_rasterizer_agree() {
    let (gpu, cpu) = (temp("agree_gpu.ppm"), temp("agree_cpu.ppm"));
    let args = |path: &Path| vec!["--scene".to_string(), "cubes".into(), "--time".into(), "0.7".into(), "--size".into(), "160x120".into(), "--output".into(), path.to_str().unwrap().to_string()];
    let run_with = |path: &Path, env: &[(&str, &str)]| {
        let a = args(path);
        let refs: Vec<&str> = a.iter().map(String::as_str).collect();
        run(&refs, env)
    };
    let out = run_with(&gpu, &[]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    if !text(&out.stdout).contains("Rendering on the CPU") {
        assert!(run_with(&cpu, &[("VK_DRIVER_FILES", "/nonexistent/icd.json")]).status.success());
        let diff = phox_core::cpu_image::compare_images(&phox_core::cpu_image::read_ppm(&gpu).unwrap(), &phox_core::cpu_image::read_ppm(&cpu).unwrap(), 2);
        assert!(diff.pixels_over_tolerance <= 160 * 120 / 100, "{diff:?}");
    }
    let _ = std::fs::remove_file(&gpu);
    let _ = std::fs::remove_file(&cpu);
}

#[test]
fn dumps_the_frame_graph_and_prints_timings_and_statistics() {
    let (dot, image) = (temp("graph.dot"), temp("stats.ppm"));
    let out = run(&["--scene", "tori", "--size", "96x72", "--frames", "6", "--stats", "--dump-graph", dot.to_str().unwrap(), "--output", image.to_str().unwrap()], &[]);
    let stdout = text(&out.stdout);
    assert!(out.status.success(), "{stdout}\n{}", text(&out.stderr));
    if stdout.contains("Rendering on the CPU") {
        return; // no GPU: nothing to dump
    }
    let graph = std::fs::read_to_string(&dot).unwrap();
    assert!(graph.starts_with("digraph") && graph.contains("forward") && graph.contains("tonemap"), "{graph}");
    assert!(stdout.contains("GPU time per frame") && stdout.contains("CPU time per frame"), "{stdout}");
    assert!(stdout.contains("triangles after cluster LOD") || stdout.contains("Drew"), "{stdout}");
    let _ = std::fs::remove_file(&dot);
    let _ = std::fs::remove_file(&image);
}

#[test]
fn every_mesh_shader_mode_renders() {
    for mode in ["on", "off", "noocclusion", "nocull"] {
        let output = temp(&format!("mode_{mode}.ppm"));
        let out = run(&["--scene", "cubes", "--size", "64x48", "--mesh-shaders", mode, "--lod", "off", "--output", output.to_str().unwrap()], &[]);
        assert!(out.status.success(), "{mode}: {}", text(&out.stderr));
        std::fs::remove_file(&output).unwrap();
    }
}

#[test]
fn shows_a_window_when_asked_to() {
    if !flag("PHOX_TEST_WINDOW") {
        eprintln!("skipping: set PHOX_TEST_WINDOW=1 (with a display, e.g. under xvfb-run) to open a window");
        return;
    }
    let out = run(&["--window", "--debug", "--frames", "20", "--scene", "cubes"], &[]);
    let stdout = text(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}\n{}", text(&out.stderr));
    assert!(stdout.contains("Presented 20 frame(s)"), "{stdout}");
}
