# Phox (Rust)

A Rust port of the C++20 Vulkan renderer Phox, built around a render graph over a small RHI. It draws a
`Scene` (meshes, materials, transforms, a camera and a directional light) either to a window or headless into
an image, and falls back to a software rasterizer when there is no usable GPU.

- **RHI**: `phox-rhi` is a `Device` / `CommandList` trait pair with typed, generation-checked handles. Backends
  implement it: Vulkan 1.3 (`phox-vulkan`), a null backend that records calls (`phox-null`) and a Metal stub.
  Everything above it is written once.
- **Render graph**: passes declare how they use transient and imported resources; the graph validates the
  declarations, culls unused work, derives usages and every barrier (synchronization2), pools transients and
  can dump itself as Graphviz DOT. A missing device feature is an error that lists the features.
- **Frames in flight**: two frame slots with their own upload buffers; the CPU never waits for the device
  inside a frame. Submissions return timeline-semaphore tickets.
- **Physically based materials**: glTF 2.0's metallic-roughness model (GGX, height-correlated Smith,
  Schlick) with base colour, metallic-roughness, normal, occlusion and emissive textures, alpha masking and
  blending, in linear light. Textures are mipmapped on the CPU and indexed from one bindless texture table.
- **glTF 2.0 loading** (`.gltf` and `.glb`, via the `gltf` crate): node hierarchies, shared meshes and
  textures, samplers, PNG and JPEG images. OBJ files load through `tobj`.
- **Forward rendering into HDR** with depth testing, then tone mapping (clamp, Reinhard or ACES) and sRGB
  encoding straight into the swapchain image.
- **GPU-driven rendering**: the scene's objects and materials stay on the GPU and the CPU uploads only what
  changed, found through the scene's change log. Compute passes cull objects against the frustum (and the
  depth pyramid), then build the draw lists that a handful of indirect draws consume.
- **Mesh shading** where the device has `VK_EXT_mesh_shader`: meshes are split into meshlets with
  meshoptimizer (v1.3, built from `vendor/`), and a task shader skips meshlets outside the view frustum, facing
  away from the camera, or hidden behind what is already drawn (two-phase occlusion culling against a depth
  pyramid). Elsewhere the same images come from the vertex pipeline.
- **Cluster LOD** (as in Nanite): each mesh is built into a hierarchy of meshlets at successive levels of
  detail, and the task shader picks, per part of the mesh, the coarsest meshlets whose error stays under a
  pixel, without cracks.
- **Verified output**: golden images rendered on lavapipe, checked with the Khronos validation layer
  (including synchronization validation) on every CI run.

## Build and run

The nix flake provides the Rust toolchain, glslang, the Vulkan loader, validation layers, a CPU Vulkan driver
(lavapipe), Graphviz and Xvfb: `nix develop` (or `direnv allow`). Run everything through it:

```bash
nix develop -c cargo build --release
nix develop -c cargo run --release -p phox-sample                                    # one frame of the triangle -> build/rendered_triangle.ppm
nix develop -c cargo run --release -p phox-sample -- --scene cubes --window          # animated, until the window is closed
nix develop -c cargo run --release -p phox-sample -- --scene materials --window      # the material model: metals, plastics, textures
nix develop -c cargo run --release -p phox-sample -- --mesh model.glb --window       # a glTF (or OBJ) file on a turntable
nix develop -c cargo run --release -p phox-sample -- --help
```

Useful options:

| Option | Effect |
| --- | --- |
| `--scene triangle\|cubes\|tori\|materials\|field\|crowd` | the sample scene to show (`field` is a 37.7-million-triangle benchmark, `crowd` a 32,768-object one) |
| `--mesh FILE` | show a glTF 2.0 (`.gltf`, `.glb`) or OBJ file on a turntable instead |
| `--window`, `--frames N` | present to a window (for N frames) instead of rendering one image |
| `--frames N` (headless) | render N frames of one pose (`--animate`: advancing 1/60 s each) and print the average GPU time per pass |
| `--mesh-shaders on\|off\|noocclusion\|nocull` | draw with task and mesh shaders where available; `noocclusion` keeps frustum and cone culling only, `nocull` turns meshlet culling off |
| `--lod on\|off`, `--lod-error PIXELS` | cluster LOD with mesh shaders, and the error it may show (default 1 pixel) |
| `--stats` | print how many triangles cluster LOD selected out of the full detail, and the CPU time per frame |
| `--time S`, `--size WxH`, `--output FILE` | headless: animation time, image size, output path |
| `--tonemap clamp\|reinhard\|aces`, `--exposure X` | tone mapping |
| `--backend vulkan\|metal` | the GPU backend (the Metal backend is a stub in this version: `metal` is refused) |
| `--debug` | validation (the Vulkan validation layer, with synchronization validation) and debug logging; exits with status 3 if validation reported anything |
| `--device NAME` | use the GPU whose name contains NAME (also `PHOX_DEVICE`) |
| `--dump-graph FILE` | write the frame's render graph as Graphviz DOT (`dot -Tsvg FILE -o graph.svg`) |

To run on lavapipe instead of your GPU, as CI does: `VK_DRIVER_FILES=$PHOX_LAVAPIPE_ICD cargo run -p phox-sample`.
Logging goes to stderr; `PHOX_LOG=debug|info|warn|error|off` sets the level. Without a usable Vulkan device
the CLI renders headless images with the software rasterizer.

To try windowed mode without a screen: `Xvfb :99 -screen 0 1280x720x24 & DISPLAY=:99 cargo run -p phox-sample -- --window --frames 60`.

## Tests

```bash
nix develop -c bash -c 'VK_DRIVER_FILES=$PHOX_LAVAPIPE_ICD PHOX_VALIDATION=1 PHOX_REQUIRE_GPU=1 cargo test --workspace'
```

| Test | What it checks | GPU |
| --- | --- | --- |
| `phox-core/tests/render_graph.rs` | culling, barriers, hazards, imports, pool reuse, capability gating, validation of declarations, DOT export | no (null device) |
| `phox-core/tests/render_view.rs` | projection, camera, transforms, meshes and tangents, frustum culling, sorting, meshlets and their culling tests, cluster LOD, sphere projection, textures and mips, materials, blending and masking in the software rasterizer, OBJ and glTF loading | no |
| `phox-core/tests/scene.rs` | the scene's change log, the sample scenes | no |
| `phox-core/tests/golden_software.rs` | the software rasterizer against `test/golden/*.ppm` | no |
| `phox-null/tests/null_device.rs` | handle pools, descriptions, recording, feature gating, the bindless table | no |
| `phox-vulkan/tests/smoke.rs` | buffers, barriers, copies, rendering, readback and timestamps on a real device | yes |
| `phox-forward/tests/golden.rs` | renders the sample scenes with vertex shaders (with and without a draw count), with mesh shaders and occlusion culling, with mesh shaders and frustum and cone culling only, and with cluster LOD (against `<case>.lod.ppm`), and compares each with `test/golden/*.ppm` | yes (skips without one) |
| `phox-forward/tests/scene_changes.rs` | a renderer reused across scene changes draws what a fresh one does | yes |
| `phox-forward/tests/mesh_paths.rs`, `lod.rs`, `gltf.rs` | two-phase occlusion culling against the vertex path, LOD triangle counts, a glTF scene on the GPU | yes |
| `phox-forward/tests/technique_graph.rs` | the technique's passes and required features on the null device | no |
| `phox-sample/tests/cli.rs` | options, exit codes, files written, the software fallback, a window (opt in) | optional |

The tests read a few environment variables, exactly as the C++ renderer's do:

- `PHOX_VALIDATION=1` runs with the validation layer and fails on any message.
- `PHOX_REQUIRE_GPU=1` fails instead of skipping when no Vulkan device is available.
- `PHOX_REQUIRE_MESH_SHADERS=1` fails instead of skipping the mesh shader runs when the device lacks them.
- `PHOX_UPDATE_GOLDEN=1` (or a comma-separated list of case names) rewrites the goldens: the vertex run writes
  `<case>.ppm` and the cluster LOD run `<case>.lod.ppm`. Regenerate them on lavapipe, which is deterministic:
  `VK_DRIVER_FILES=$PHOX_LAVAPIPE_ICD PHOX_UPDATE_GOLDEN=1 cargo test -p phox-forward --test golden`.
  Mismatching images are written to `golden_out/`.
- `PHOX_NO_DRAW_INDIRECT_COUNT=1` makes the device act as if it lacked `drawIndirectCount` (the golden test also
  checks that path explicitly).
- `PHOX_DEVICE=<name>` picks the device whose name contains `<name>`.
- `PHOX_TEST_WINDOW=1` lets the CLI test open a window (use it under `xvfb-run`).

The comparison tolerance is the C++ one: a per-channel difference of 2 on all but 0.5% of the pixels (2% for the
textured `materials` case, whose mips the software rasterizer cannot match). It is never raised to make a golden pass.

CI (`.github/workflows/ci.yml`) runs clippy and every test on Ubuntu 24.04 with lavapipe and validation, then
renders the sample scenes with the CLI.

## How a frame is made

```
Scene ──change log──▶ GpuScene ──updates──▶ RenderGraph ──compile──▶ rhi::Device ──▶ GPU
 (data)               (what changed,        (passes declare           (Vulkan, or the
                       mesh table)           what they touch)          null backend)
```

| Crate | Contents |
| --- | --- |
| `phox-core` | scene, math, camera, view extraction, software rasterizer, glTF/OBJ loaders, meshlets and cluster LOD (over meshoptimizer), the render graph |
| `phox-rhi` | the `Device` and `CommandList` traits, handles, descriptions, capabilities |
| `phox-vulkan` | the Vulkan 1.3 backend (ash, gpu-allocator): device, resources, commands, pipelines, swapchain |
| `phox-null` | a backend that records calls, for tests without a GPU |
| `phox-metal` | the Metal backend, stubbed (not part of this pass) |
| `phox-forward` | the forward technique's passes (graph components), the GPU scene, mesh and texture caches, the renderer |
| `phox-sample` | the `phox-sample` CLI |
| `shaders/` | GLSL, compiled to SPIR-V by `phox-forward`'s `build.rs` and embedded; `common.glsl` mirrors the `#[repr(C)]` structs |
| `vendor/meshoptimizer` | meshoptimizer v1.3, built by the `cc` crate (the `.lod.ppm` goldens depend on its exact output) |
| `docs/architecture.md` | the design and its rationale |

## Render graph

A frame's graph is declared on a builder made by `RenderGraph::begin`; the `RenderGraph` itself only holds the
transient pool, so keep one per device:

```rust
let mut graph = RenderGraph::new(device.clone());
let mut b = graph.begin();
let hdr = b.create_texture(&TextureDesc::new(w, h, Format::RGBA16Float, TextureUsage::NONE, "hdr"));
b.add_pass(
    "forward",
    move |p| {
        p.color_target(hdr)?;            // declares: renders to hdr, clearing it
        Ok(())
    },
    move |c| {
        // c.cmd records commands; rendering was begun from the declared attachments
        Ok(())
    },
)?;
b.add_pass(
    "tonemap",
    move |p| {
        p.read_texture(hdr, TextureAccess::ShaderRead)?;
        p.color_target(backbuffer)?;     // an imported swapchain image
        Ok(())
    },
    |_| Ok(()),
)?;
b.compile()?;
let ticket = b.execute()?;
```

- Passes run in the order added. A pass is kept if it has `side_effect()`, writes an imported resource, or
  produces something a kept pass reads.
- Barriers come from how states change between passes (and write hazards), batched per pass.
- A transient's usage is derived from what the passes do with it; transients come from a pool, so steady-state
  frames create nothing.
- Invalid declarations (reading before anything writes, loading undefined contents, one resource used two
  ways in a pass, targets of different sizes, ...) are errors that name the pass and the resource.

Derived from the C++ Phox, itself derived from [ModernCppStarter](https://github.com/TheLartians/ModernCppStarter).
