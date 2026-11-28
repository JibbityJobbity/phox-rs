// Compiles shaders/*.{vert,frag} to SPIR-V with glslangValidator into OUT_DIR, embedded by src/renderer.rs.
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    for name in ["immediate.vert", "immediate.frag"] {
        println!("cargo:rerun-if-changed={}", shader_dir.join(name).display());
        let output = Command::new("glslangValidator")
            .current_dir(&shader_dir)
            .args(["-V", name, "-o"])
            .arg(out_dir.join(format!("{name}.spv")))
            .output()
            .unwrap_or_else(|e| panic!("cannot run glslangValidator ({e}); it is in the nix dev shell: `nix develop`"));
        if !output.status.success() {
            panic!(
                "glslangValidator failed on {name}:\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
