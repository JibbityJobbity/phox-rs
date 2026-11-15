# Phox (Rust)

Rust port of the C++20 Vulkan renderer at `/home/jibbity/dev/Projects/phox-sample` (read-only). See README.md for
features and options and docs/architecture.md for the design.

## Build environment

Everything comes from the nix flake (`.envrc` has `use flake`). Run commands inside it:
`nix develop -c bash -c '...'`. It provides rustc/cargo/clippy, glslang, the Vulkan loader, the validation layer,
lavapipe (`PHOX_LAVAPIPE_ICD`), Graphviz and Xvfb. The flake reuses the C++ repo's nixpkgs pin so lavapipe is the
build the golden images came from.

```bash
nix develop -c cargo clippy --workspace --all-targets -- -D warnings
nix develop -c bash -c 'VK_DRIVER_FILES=$PHOX_LAVAPIPE_ICD PHOX_VALIDATION=1 PHOX_REQUIRE_GPU=1 cargo test --workspace'
```

Add `PHOX_REQUIRE_MESH_SHADERS=1` to make the mesh shader paths fail rather than skip. Windowed mode: start a
private Xvfb (`Xvfb :99 -screen 0 1280x720x24 &`) and set `DISPLAY=:99 WAYLAND_DISPLAY=`; never open windows on
the real desktop. `PHOX_TEST_WINDOW=1` enables the CLI's window test.

## Notes

- Never raise a golden tolerance to make an image pass; regenerate on lavapipe with `PHOX_UPDATE_GOLDEN=1` and look
  at the before/after instead. The goldens in `test/golden/` came from the C++ renderer and match without changes.
- `shaders/*` are the C++ repo's, unchanged. Structs mirrored from `shaders/common.glsl` live in
  `phox-core::shader_interface` and `phox-forward::gpu_types` with `const` size assertions; change both together.
  Shading changes in `shaders/pbr.glsl` must be mirrored in `phox-core::shading` (the software rasterizer).
- meshoptimizer is vendored (v1.3) and built by `phox-core/build.rs`; the `.lod.ppm` goldens depend on its output.
- Objects are changed through `*_mut()` guards, which log the change when dropped; the GPU scene finds changes
  through them. Do not hold a guard across frames.
- A buffer a shader reads through a device address must still be declared in the graph, or it gets no barrier.
- A `cd` that fails in a `cd x && python3 - <<EOF` chain silently skips the script: use absolute paths.
- The validation layer's synchronization checks are the arbiter for barriers; a test that triggers one has a bug.
