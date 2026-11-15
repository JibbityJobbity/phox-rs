{
  description = "Phox (Rust): Rust toolchain, glslang, Vulkan loader, validation layers and lavapipe";

  # The same nixpkgs as the C++ Phox, so lavapipe (mesa) is the build the golden images came from.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  # macOS gets Vulkan from MoltenVK, which only reports Vulkan 1.3 (required by Phox) from 1.3.0 on.
  # Pinned separately so the Linux shell (and the lavapipe used for goldens) does not move with it.
  inputs.nixpkgs-darwin.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs, nixpkgs-darwin }:
    let
      linux = "x86_64-linux";
      darwin = "aarch64-darwin";

      common = pkgs: with pkgs; [
        rustc
        cargo
        clippy
        rustfmt
        rust-analyzer
        glslang        # glslangValidator: build.rs compiles the shaders to SPIR-V
        vulkan-loader
        vulkan-tools   # optional: vulkaninfo, vkcube
        graphviz       # renders frame graphs dumped with --dump-graph: dot -Tsvg graph.dot -o graph.svg
        pkg-config
      ];
    in {
      devShells.${linux}.default =
        let
          pkgs = import nixpkgs { system = linux; };
          # winit dlopen()s these at runtime (X11 or Wayland), ash dlopen()s libvulkan.so.1.
          runtimeLibs = with pkgs; [
            vulkan-loader
            libxkbcommon
            wayland
            xorg.libX11
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXi
            xorg.libxcb
          ];
        in pkgs.mkShell {
          name = "phox-rs";

          # xorg.xorgserver provides Xvfb, a virtual display for testing windowed mode without a screen:
          #   Xvfb :99 -screen 0 1280x720x24 & DISPLAY=:99 cargo run -p phox-sample -- --window --frames 60
          packages = common pkgs ++ [ pkgs.gcc pkgs.xorg.xorgserver ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            # Makes the validation layer available (it is only enabled with --debug / PHOX_VALIDATION=1).
            export VK_ADD_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d''${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH}
            # Mesa's CPU Vulkan driver, built against this shell's glibc (the system's drivers may not load
            # here). Use it with: VK_DRIVER_FILES=$PHOX_LAVAPIPE_ICD (this is what CI and the goldens use).
            export PHOX_LAVAPIPE_ICD=${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json
          '';
        };

      # Builds with the default (clang) stdenv. There is no lavapipe here, so goldens stay a Linux job.
      devShells.${darwin}.default =
        let
          pkgs = import nixpkgs-darwin { system = darwin; };
        in pkgs.mkShell {
          name = "phox-rs";

          packages = common pkgs ++ [ pkgs.moltenvk ];

          shellHook = ''
            # ash dlopen()s libvulkan.dylib by name. SIP strips DYLD_* when a system binary
            # (/bin/sh, /usr/bin/env) starts the program, so run it directly or from this shell.
            export DYLD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            # The loader finds MoltenVK (Vulkan on Metal) itself, through the XDG_DATA_DIRS mkShell sets.
            # A VK_LAYER_PATH from outside replaces the search, VK_ADD_LAYER_PATH included, so extend it.
            layers=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
            if [ -n "''${VK_LAYER_PATH:-}" ]; then
              export VK_LAYER_PATH=$layers:$VK_LAYER_PATH
            else
              export VK_ADD_LAYER_PATH=$layers''${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH}
            fi
            unset layers
          '';
        };
    };
}
