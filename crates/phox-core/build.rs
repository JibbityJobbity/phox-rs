// Builds meshoptimizer v1.3 (vendor/meshoptimizer, the tag the C++ Phox uses) so the meshlet and cluster
// LOD output, which the .lod.ppm goldens depend on, is the same.
fn main() {
    let dir = std::path::Path::new("../../vendor/meshoptimizer/src");
    let mut build = cc::Build::new();
    build.cpp(true).opt_level(2).warnings(false).include(dir);
    let mut sources = Vec::new();
    for entry in std::fs::read_dir(dir).expect("vendor/meshoptimizer/src is missing") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "cpp") {
            sources.push(path);
        }
    }
    sources.sort();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
        build.file(source);
    }
    println!("cargo:rerun-if-changed=../../vendor/meshoptimizer/src/meshoptimizer.h");
    build.compile("meshoptimizer");
}
