// Compiles shaders/*.{vert,frag,comp,task,mesh} to SPIR-V with glslangValidator and generates `shaders.rs`
// in OUT_DIR, which embeds each module with include_bytes!. Task and mesh shaders (VK_EXT_mesh_shader) need
// SPIR-V 1.4 or later, so they are compiled for Vulkan 1.3.
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", shader_dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| ["vert", "frag", "comp", "task", "mesh"].contains(&e.to_str().unwrap_or(""))))
        .collect();
    sources.sort();

    let mut table = String::from("pub const ALL: &[(&str, &[u8])] = &[\n");
    for source in &sources {
        let name = source.file_name().and_then(|n| n.to_str()).expect("shader names are UTF-8");
        let spv = out_dir.join(format!("{name}.spv"));
        let mut command = Command::new("glslangValidator");
        command.current_dir(&shader_dir).arg("-V");
        if name.ends_with(".task") || name.ends_with(".mesh") {
            command.args(["--target-env", "vulkan1.3"]);
        }
        let output = command.arg(name).arg("-o").arg(&spv).output().unwrap_or_else(|e| {
            panic!("cannot run glslangValidator ({e}); it is in the nix dev shell: `nix develop`")
        });
        if !output.status.success() {
            panic!(
                "glslangValidator failed on {name}:\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        table += &format!("    ({name:?}, include_bytes!({:?})),\n", spv.display().to_string());
    }
    table += "];\n";
    std::fs::write(out_dir.join("shaders.rs"), table).expect("write shaders.rs");
}
