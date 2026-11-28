# phox-metal: plan

The Metal backend implements `phox_rhi::Device` and `CommandList` the same way `phox-vulkan` does. It targets
Metal 3 on Apple Silicon only (no Intel Macs), and assumes macOS 14 or later. It uses `objc2-metal`,
`objc2-quartz-core` (for `CAMetalLayer`) and `objc2-app-kit`. Shaders are translated with SPIRV-Cross.

Status: **stage 1 is done** (`crates/phox-metal`). `MetalDevice::new` (offscreen) and
`MetalDevice::new_for_view` (a `CAMetalLayer` in an `NSView`) pass `tests/smoke.rs` on Apple Silicon, also
under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, and `phox-immediate`'s and `phox-ui`'s render tests run on
Metal by default on macOS. Not yet checked: presenting to a real window (no host exists yet; see
`docs/ui.md`). Stages 2 and 3 are next.

Out of scope for now: golden images on Metal, and macOS CI. It's also unknown whether the nix flake works on
macOS. Until that's checked, Metal builds with plain `cargo` and the Xcode command-line tools.

## Stages

| Stage | Goal | Done when |
|---|---|---|
| 1 | Everything `phox-immediate` needs | `phox-immediate`'s render tests pass on Metal, and `ImmediateRenderer::render_window` draws into a `CAMetalLayer` |
| 2 | Basic forward rendering | `ForwardRenderer` draws the sample scenes on the vertex pipeline, with no bindless textures, device addresses or indirect count |
| 3 | GPU-driven rendering, then mesh shaders, then cluster LOD | Each feature passes its existing tests on Metal |

`capabilities()` reports only what the current stage supports. Anything above the backend then fails with the
render graph's "missing features" error, never by drawing something wrong.

## Shaders: SPIR-V in, MSL out

`ShaderDesc` hands the backend SPIR-V words at runtime, so translation happens in `create_shader`
(`src/shader.rs`):

1. SPIRV-Cross, through the `spirv-cross2` crate (vendored and built with `cc`), compiles the SPIR-V to the
   Metal Shading Language with fixed binding rules (below).
2. `MTLDevice::newLibraryWithSource` compiles the result. Libraries are cached by a hash of the SPIR-V.
3. Building `.metallib` files ahead of time in the build step is an optimisation for later, if startup time
   matters. Nothing above the backend changes either way.

**Binding rules** (the same for every shader, so pipelines need no per-shader tables):

| Vulkan | Metal |
|---|---|
| `(set, binding)` uniform or storage buffer | buffer index `set * 8 + binding` (sets 0–2) |
| `(set, binding)` sampled or storage texture | texture index `set * 8 + binding`, plus a sampler at the same index |
| Push constants | buffer index 24, set with `setVertexBytes` / `setFragmentBytes` / `setBytes` |
| Vertex buffer 0 | buffer index 30 (`stage_in` through an `MTLVertexDescriptor`) |
| Bindless set (stage 3) | Argument buffer, at a buffer index of its own |

## How Vulkan concepts map to Metal

| RHI | Metal |
|---|---|
| `Device` | `MTLDevice` plus one `MTLCommandQueue` |
| Timeline semaphore and `SubmitTicket` | `MTLSharedEvent`. Each submit encodes a signal of the next value. `is_complete` reads `signaledValue`, and `wait` waits on it with a listener or `waitUntilSignaledValue`. |
| `MemoryKind::Upload` / `Readback` | `MTLStorageModeShared`; `write_buffer` and `read_buffer` copy through `contents()` |
| `MemoryKind::Device` | `MTLStorageModePrivate` |
| Textures | `MTLTexture`. The formats map one to one (RGBA8, BGRA8, sRGB variants, D32F, RGBA16F). |
| `barrier()` | In stages 1–2, resources use Metal's automatic hazard tracking, so `barrier` only checks that the call is valid. Stage 3 may switch to untracked resources with `MTLFence` and `memoryBarrier` for performance. |
| `begin_rendering` / `end_rendering` | `MTLRenderPassDescriptor` and a render command encoder. Clear becomes `loadAction = Clear`. |
| Copy and fill commands | A blit encoder, opened lazily. The command list switches encoders as needed. Metal fills with one byte only, so a `u32` pattern whose bytes differ is copied from a scratch buffer. |
| `set_scissor` | `setScissorRect`, clamped to the render target (Metal rejects scissors outside it). |
| `dispatch` | A compute encoder. Threadgroup size comes from the shader's `LocalSize`, recorded when the pipeline is created. |
| Push constants | `set*Bytes` at buffer 24 |
| Debug labels | `pushDebugGroup` / `popDebugGroup` |
| Timestamps | Not supported in stage 1. Later via `MTLCounterSampleBuffer`. |
| Deferred destruction | As in Vulkan: a destroyed resource is kept in a queue until the ticket of the last submit that used it completes. |

## Windows and swapchains

`MetalConfig` takes an optional `RawWindowHandle` of kind `AppKit` (an `NSView`). The device:

- adds a `CAMetalLayer` to that view as a sublayer, which is what embedding in a host's view needs;
- in `acquire_backbuffer`, sets the layer's `drawableSize` and calls `nextDrawable`, then wraps the drawable's
  texture as an external `TextureHandle` that is released after the frame;
- in `submit_for_present`, encodes `presentDrawable` and commits.

`Present` and `Undefined` states need no work, because Metal has no image layouts.

## Validation and tests

- `PHOX_VALIDATION=1` on Metal turns on `MTL_DEBUG_LAYER` and shader validation (set as environment variables
  before the device is created). Validation errors fail the test.
- Render tests stop naming `VulkanDevice` directly. A shared helper in `phox-core` (test-only) chooses the
  backend: `PHOX_BACKEND=metal|vulkan`, defaulting to Metal on macOS and Vulkan elsewhere. Skipping when no
  device is available, and `PHOX_REQUIRE_GPU=1`, behave as they do now.
- Stage 1 also adds `phox-metal/tests/smoke.rs`, matching Vulkan's.

## Risks

- **Stage 3 on Metal:** device addresses (`gpuAddress`, which needs Metal 3), argument-buffer tiers, and how
  SPIRV-Cross handles `GL_EXT_buffer_reference` and mesh shaders. Each needs a small test on its own before the
  full pass is ported.
- **SPIRV-Cross build** on macOS without nix: `spirv-cross2` builds its vendored C++ with `cc`, so only Xcode's
  clang is needed.
- **Threading:** `Device: Send + Sync` holds, because Metal objects are thread-safe. The layer and its view must
  be touched only on the main thread, so `acquire_backbuffer` asserts it is running there.
