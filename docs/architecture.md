# Phox architecture (Rust)

This describes the Rust port as built, why it is shaped this way, and where it departs from the C++ renderer it
was ported from. The C++ design lives in that repository's `docs/architecture.md`; section numbers below follow
its topics where useful.

## 1. Shape

```
  Scene (plain data: objects, meshes, materials, camera, light, tone mapping)       phox-core
      |  change log (Scene::changes), extract_camera()
      v
  GpuScene  (a GpuInstance + MaterialData per object slot, on the GPU)               phox-forward
      |  what changed, the mesh table
      v
  ForwardTechnique: scene update > cull > args > forward (early/late) > tonemap      phox-forward
      |  graph components declare their passes
      v
  RenderGraph (validate, cull, derive usages and barriers, pool transients)          phox-core
      |  record()
      v
  rhi::Device / CommandList (traits)                                                 phox-rhi
      |
      +-- phox-vulkan  Vulkan 1.3: dynamic rendering, synchronization2, timeline semaphore
      +-- phox-null    records call names; for tests without a GPU
      +-- phox-metal   stub
```

Two rules keep the design small:

1. **The graph tracks resources whose state changes inside a frame**: transient targets, the swapchain image, the
   readback buffer, and the scene buffers the passes write. Resources that stay in one state once uploaded (mesh
   buffers, textures) live in caches and get one barrier at upload.
2. **The graph is rebuilt every frame.** It is bookkeeping over cached objects (pipelines, pooled images, uploaded
   meshes), so rebuilding costs little and there is no frame-to-frame graph state to get wrong.

Passes are written once, against `phox_rhi`; nothing above `phox-vulkan` names Vulkan. The backend is chosen by
whoever creates the `Device` (the CLI, a test).

## 2. Decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Abstraction boundary | `Device` / `CommandList` traits with typed handles; pass bodies record through them | Passes are written once and run on Vulkan or the null backend; the graph is testable without a GPU |
| 2 | Device ownership | Every `Device` method takes `&self`; state is behind locks; command lists are owned boxes handed to `submit` | A pass can use the device while a list is being recorded; Rust's borrow rules stay out of the way |
| 3 | Errors | `Result` everywhere; the graph's missing-feature error lists the features | No panics on bad input or missing capabilities |
| 4 | Misuse in a command list | Remembered and reported by `submit` | Recording calls stay infallible; a mesh draw without the feature fails at submit, not by crashing |
| 5 | Scene mutation | `*_mut()` accessors return guards that bump the revision and log the change when dropped | There is no way to change an object without the renderer hearing of it |
| 6 | Ids | `Clone` allocates a fresh id; moves keep it | Renderers key per-object state by id |
| 7 | Math | glam behind `phox_core::math`, Vulkan depth range and the C++ conventions | Quaternions, frustums and clip-space conventions are easy to get wrong by hand |
| 8 | Shaders | `shaders/*` unchanged; `build.rs` compiles them with glslangValidator and embeds the SPIR-V | One source of truth shared with the C++ renderer |
| 9 | Verification | Golden images from lavapipe, checked under synchronization validation | Lavapipe is deterministic and runs the real code path in CI |

## 3. Render graph (`phox-core::graph`)

### 3.1 Resources and handles

Textures and buffers, each either transient (`create_texture`, `create_buffer`) or imported (`import_texture`,
`import_buffer`, with a state they are in when the graph starts and the state they must end in; `Undefined`
as the end state means "leave it"). Handles (`GraphTexture`, `GraphBuffer`) are indices valid for one frame's
builder. A transient's `usage` is ignored on creation: it is the union of what live passes do with it.

### 3.2 States

`ResourceState` (`phox-rhi`) is the one table that says what a resource is doing between passes:
`ColorAttachment`, `DepthWrite`, `DepthRead`, `ShaderRead`, `ShaderWrite`, `TransferSrc/Dst`, `VertexOrIndex`,
`Uniform`, `IndirectArgs`, `HostRead`, `Present`. `phox-vulkan::convert` maps each to a layout, pipeline stages and
accesses. A state a resource entered the graph in (and the first use of any transient) has the source scope
ALL_COMMANDS / MEMORY_WRITE: whatever came before on the queue, including a previous user of a pooled image or
the semaphore wait for a swapchain image. Two consecutive accesses in a write state still get a barrier, because
the second must wait for the first's writes.

### 3.3 Declaring passes

`GraphBuilder::add_pass(name, setup, execute)`. `setup` runs at once and declares through a `PassBuilder`
(`color_target`, `depth_target`, `depth_read`, `read_texture`, `write_texture`, `read_buffer`, `write_buffer`,
`side_effect`); `execute` runs during recording with a `PassContext` (the command list, the device, and the real
handles of what the pass declared). A failing `setup` leaves the graph as it was. Resources that a shader reads
through a device address must still be declared, or no barrier would protect them.

### 3.4 Compile and execute

`compile()` checks the device's features (one error naming every missing one, once each, in declaration order),
validates (reads need an earlier write unless imported, a pass may not read and write one resource, attachments
share a size, no two states of one resource in one pass), culls (roots are `side_effect` passes and writers of
imports; a live pass makes the producers of what it reads live), derives usages, takes transients from the pool
(retired after three unused frames) and plans barriers. `record()` writes every live pass, with its barriers as
one batch and a debug label (and GPU timestamps when asked), into one command list; `execute()` submits it.
`dump()` is a text form for tests and `to_dot()` Graphviz.

### 3.5 Departures from the C++ graph

The C++ repository has two graphs; the RHI one is ported. The older graph's *versioned handles* (a write returns
a new version, which makes culling exact when a pass overwrites an image nothing read) and *relative extents* are
not ported, nor is *aliasing*. Without versions, a pass that is fully overwritten before anyone reads is still
kept if something later reads the resource.

## 4. The RHI (`phox-rhi`)

Typed `Handle<T>` (index and generation; a stale handle is an error, never a crash), descriptions
(`BufferDesc`, `TextureDesc`, `GraphicsPipelineDesc`, ...), `Capabilities` with a `Feature` set, and the two
traits. It covers what the renderer needs: mesh-shader draws, indirect draws with an optional count buffer,
timeline-semaphore tickets, a bindless texture table, compute dispatches (also indirect), buffer fills and
copies, per-mip storage-image binding, GPU timestamps and presenting to a window (`acquire_backbuffer`,
`submit_for_present`). `write_buffer` / `read_buffer` replace the C++ `map()`: no mapped slice crosses the
trait.

## 5. The Vulkan backend (`phox-vulkan`)

- **Instance and device.** `Instance` loads Vulkan dynamically, enables the Khronos validation layer with
  synchronization validation through `VK_EXT_layer_settings` when asked, and counts the messages it reports.
  Device selection requires Vulkan 1.3 with synchronization2, dynamic rendering, timeline semaphores, buffer device
  addresses, descriptor indexing (the bindless table) and `multiDrawIndirect` with `drawIndirectFirstInstance`;
  `drawIndirectCount` and mesh shaders are optional (`PHOX_NO_DRAW_INDIRECT_COUNT=1` /
  `VulkanConfig::draw_indirect_count` turn the first off). Discrete beats integrated beats virtual beats CPU;
  `PHOX_DEVICE` / `VulkanConfig::preferred_device` pick by name.
- **Resources and destruction.** Pools of resources behind one lock; gpu-allocator for memory. Destruction is
  deferred behind the timeline semaphore (a resource is freed once the submission being recorded and every earlier
  one are done).
- **Commands.** One command pool per list, recycled when its ticket completes. Descriptor sets are written lazily at
  draw or dispatch from what the list bound (`bind_uniform_buffer` and friends), from per-list pools; the bindless
  table is one set bound by `bind_bindless_table`. Misuse is recorded and reported by `submit`.
- **Synchronization.** `convert::texture_access` / `buffer_access` turn states into barriers; the source side
  keeps only write accesses.
- **Swapchain.** `acquire_backbuffer` creates and recreates the swapchain (the window's size changed, or it was out
  of date); `submit_for_present` waits on the acquire semaphore, signals the render-finished one and presents.
  The images are textures in the pool whose backing the swapchain owns.

## 6. The forward technique (`phox-forward`)

### 6.1 The GPU scene

`GpuScene` keeps a `GpuInstance` and a `MaterialData` per object slot (`ObjectId::index`) in device-local buffers.
Each frame it reads `Scene::changes` (falling back to a full scan when the log cannot say), compares the objects'
transform and material revisions, and hands only the changed slots to the **scene update** compute pass. Only the
mesh table (a `GpuMesh` per distinct mesh) is written every frame. It holds the meshes and textures its objects use
and asks the caches to keep them resident. Slot state the GPU keeps across frames, the occlusion history and the
meshlet visibility bits, follows the slot and starts over (cleared, twice the size) when the bits run out; both are
hints, so stale ones cost time, never correctness. Blended objects are sorted and written by the CPU each frame, in
runs that share a pipeline, and drawn with vertex shaders after the opaque ones.

### 6.2 Passes

`scene update` > `clear draw counters` > `cull` (one invocation per slot: frustum test, then append a `GpuDraw` and
its work with atomics) > `draw args` (meshlet path: turns the work count into a task dispatch) > `forward` >
`forward blend` > `tonemap` > `readback` or present. Opaque draws are indirect: task and mesh dispatches whose
arguments the GPU wrote, or indexed draws with the cull pass's count (or, without `drawIndirectCount`, a cleared
command buffer drawn whole, unused commands drawing nothing). The shaders fetch their vertices through device
addresses, so one pipeline draws every mesh.

### 6.3 Mesh shading, occlusion culling and cluster LOD

With mesh shaders each mesh is split into meshlets (64 vertices, 124 triangles; `meshopt_buildMeshlets`, cone
weight 0.25). The task shader drops meshlets outside the frustum or, for single-sided meshes, facing away. With
occlusion culling the opaque pass runs twice: **forward early** draws the meshlets visible last frame, a **depth
pyramid** compute pass (R32F, each texel the farthest depth it covers) is built from its depth, **cull late** and
**forward late** test everything else against the pyramid, draw what is newly visible and record the visibility
bits. This is exact: a meshlet with a visible pixel is nearer there than whatever the early pass drew. With cluster
LOD the meshlets form a hierarchy built by `buildMeshletLod` (groups of about 16 simplified to half with shared
borders locked) and the task shader's `lodSelected` draws, per part of the mesh, the coarsest meshlets whose error
stays under the budget, without cracks. The hierarchy is built by meshoptimizer v1.3 compiled from `vendor/`, and
its output matches the C++ renderer's, which the `.lod.ppm` goldens verify.

### 6.4 Caches

`MeshCache` uploads each mesh once per version: a vertex buffer (read by address), a slice of a shared index pool,
and its meshlets; unused for 120 frames it is evicted. `TextureCache` creates each texture with its CPU-built
mip chain (sRGB textures get sRGB formats), registers it in the bindless table at once and uploads the mips in one
submission ahead of the frame. Both submit their copies ahead of the frame's graph, with the barriers that make them
visible; staging buffers are destroyed at once (the device defers the free).

## 7. Software rasterizer and verification

`phox_core::software_rasterizer::rasterize` draws a `RenderView` with the GPU's projection, depth test, lighting
(`shade_pbr` mirrors `shaders/pbr.glsl`) and tone mapping. It is the fallback when there is no device and a check
on the GPU's output: it matches every golden within the C++ tolerance. The GPU paths are compared with
`test/golden/*.ppm` (rendered by the C++ renderer on lavapipe), exactly: the maximum channel delta is 0 on every
case and path except the LOD ones (1, and 92 on 33 pixels of `materials`).

## 8. Departures from the C++ renderer, and what is missing

- **Metal** is a stub crate that refuses to create a device.
- **`Mesh` behind an `Arc` cannot be edited in place**, so a changed mesh is always a different `Mesh`; the C++
  scene test's "shared mesh given new geometry" step is covered by retargeting. The GPU scene's version check
  remains for completeness.
- **glTF**: files that require `KHR_mesh_quantization` are refused (the `gltf` crate has no support), and an image
  that cannot be decoded fails the load instead of being skipped with a warning.
- **Blended objects** always use the vertex path, even when the opaque ones use mesh shaders.
- **The graph** has no versioned handles, relative extents or aliasing (section 3.5).
- **Windowed mode** was verified on lavapipe under Xvfb, including validation; swapchain recreation on resize is
  implemented but was not exercised by a test.
- **Dependencies**: no `clap` (the CLI parses its options by hand).

## 9. Next

Image-based lighting, shadows, GPU-sorted transparency, aliasing of transients, async compute and a Metal backend,
as in the C++ renderer's list; plus the remaining Nanite pieces (hierarchical cluster culling, a visibility buffer,
streaming).
