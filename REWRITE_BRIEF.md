# Phox: Rust rewrite

Rewrite the C++20 Vulkan renderer at /home/jibbity/dev/Projects/phox-sample in Rust, in this
directory. Read that repo's README.md, CLAUDE.md and docs/architecture.md first; they describe the
features, options, build environment and design. Do not modify the C++ repo.

## Decisions already made

- Port the RHI architecture (include/phox/rhi, include/phox/graph, backends/, techniques/), not the
  older phox::vk / phox::mtl layer where pass bodies record Vulkan directly. rhi::Device becomes a
  trait with typed handles; graph::RenderGraph sits on top of it; passes are written once. Bring the
  RHI up to full feature coverage during the port: mesh shaders, indirect draws with optional
  drawIndirectCount, timeline-semaphore submit tickets, the bindless texture table, compute passes,
  GPU timestamps. The feature set to reach is everything README.md lists and every PhoxSample option.
- Cargo workspace, Rust edition 2024. Crates: phox-core (scene, math, render graph, view extraction,
  software rasterizer, loaders), phox-rhi (trait + descs), phox-vulkan (ash), phox-forward (the
  technique and passes), phox-sample (the CLI), plus a null backend for graph tests. Leave a Metal
  backend crate stubbed; it is not part of this pass.
- Dependencies: ash (Vulkan 1.3, loaded dynamically), gpu-allocator, glam behind a math module that
  pins the Vulkan depth range and conventions of phox/Math.hpp, winit + raw-window-handle +
  ash-window for the window, gltf + image for glTF, tobj for OBJ, bytemuck for GPU structs,
  thiserror for errors, log + env_logger honouring PHOX_LOG, serial_test for GPU tests.
- meshoptimizer: the .lod.ppm goldens depend on its exact output. Build the same version the C++
  uses (CPM tag meshoptimizer@1.3) via the cc crate with hand-written FFI for the functions
  Meshlets.cpp calls, rather than the meshopt crate's bundled copy. If the output still differs,
  regenerate the LOD goldens on lavapipe and show me the before/after images.
- Shaders: copy shaders/*.glsl, *.vert, *.frag, *.comp, *.task, *.mesh unchanged. A build.rs
  compiles them with glslangValidator (task/mesh need --target-env vulkan1.3 / SPIR-V 1.4) and the
  binary embeds the SPIR-V with include_bytes!. Shader interface structs are #[repr(C)] + Pod with
  const size assertions matching common.glsl.
- Errors are Result, not panics; the graph's missing-feature error carries the feature list.
- Scene change log: object mutation goes through *_mut() accessors returning a guard that records
  the change on drop, replacing the C++ "non-const accessor" convention.
- Mesh and Object: Clone allocates a fresh id; moves keep it.
- Tests: port render_graph, render_view, scene and golden tests to cargo test. Copy test/golden/*.ppm
  unchanged and keep the tolerance (per-channel difference of 2) from test/source/ImageCompare.hpp.
  GPU tests are serialised. Honour PHOX_VALIDATION, PHOX_REQUIRE_GPU, PHOX_REQUIRE_MESH_SHADERS,
  PHOX_UPDATE_GOLDEN, PHOX_NO_DRAW_INDIRECT_COUNT and PHOX_DEVICE exactly as the C++ does.
- Build environment: write a flake.nix based on the C++ one with rustc, cargo, rust-analyzer,
  clippy, glslang, vulkan-loader, vulkan-validation-layers and lavapipe (PHOX_LAVAPIPE_ICD). Add an
  .envrc with `use flake`. Run everything through `nix develop -c`. Goldens are checked on lavapipe:
  VK_DRIVER_FILES=$PHOX_LAVAPIPE_ICD.
- CI: a GitHub Actions workflow on Ubuntu 24.04 that installs the toolchain and lavapipe, runs
  cargo clippy, cargo test with PHOX_VALIDATION=1 PHOX_REQUIRE_GPU=1, and renders the sample
  scenes with the CLI.

## Order of work

Work in phases, each ending with `cargo clippy -- -D warnings`, `cargo test` on lavapipe with
validation, and one git commit (initialise the repo first). Phases:

1. Workspace, flake, math, scene, camera, view extraction, software rasterizer, loaders; port the
   render_view and scene unit tests. The CLI renders the triangle scene headless through the
   software rasterizer.
2. RHI trait, null backend, render graph with culling, barriers, transient pool and DOT dump; port
   the render_graph tests.
3. Vulkan backend: device selection with the descriptor-indexing and indirect-draw requirements from
   CLAUDE.md, frames in flight, upload arena, swapchain, forward + tonemap + readback through the
   graph, vertex path only. Vertex-path goldens must match.
4. Materials, textures, bindless table, glTF scenes, PBR shading mirrored in the software rasterizer.
5. Mesh shading with meshlet culling and two-phase occlusion culling against the depth pyramid.
6. GPU-driven rendering: persistent GPU scene fed by the change log, compute culling, indirect draws.
7. Cluster LOD; the .lod.ppm goldens.
8. Windowed mode, --frames timings, --stats, --dump-graph, remaining CLI options, README, docs.

If a phase cannot be finished, say exactly what is missing and why before moving on. Never mark a
golden as passing by raising the tolerance.
