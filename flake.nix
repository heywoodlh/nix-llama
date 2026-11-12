{
  description = "nix-llama: a turnkey, self-hosted llama.cpp server that tunes itself to your hardware";

  # Deliberately a single input. Every dependency this flake has is a package
  # in nixpkgs, pinned by flake.lock, which keeps the supply chain small enough
  # to audit -- and means the llama.cpp in the container is byte-identical to
  # the one the service runs.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;

      version = lib.fileContents ./VERSION;

      # x86_64-darwin is absent on purpose: nixpkgs 26.11 dropped support for
      # it, and listing it makes `nix flake show` and `nix flake check` fail
      # rather than skip it. Intel Macs can still run the container.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forAllSystems = f: lib.genAttrs systems f;

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ self.overlays.default ];
        };

      # CUDA needs its own nixpkgs instance: the components are unfree, and
      # enabling that globally would surprise anyone importing the overlay.
      # Only llama.cpp is taken from here, so nothing else gets rebuilt.
      cudaPkgsFor =
        system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

      eachSystem = f: forAllSystems (system: f (pkgsFor system) system);

      supportsCuda = system: lib.hasSuffix "-linux" system;
    in
    {
      overlays.default = import ./nix/overlay.nix {
        inherit version;
        self = ./.;
        nixpkgsSrc = nixpkgs;
      };

      packages = eachSystem (
        pkgs: system:
        {
          default = pkgs.nix-llama;

          inherit (pkgs)
            nix-llama
            nix-llama-cpu
            nix-llama-vulkan
            nix-llama-rocm
            ;

          # The exact llama.cpp build the default package serves with. Exposed
          # so the container build and the flake provably share it.
          llama-cpp = pkgs.nix-llama.llama-cpp;

          # --- the model and its pipeline stages -------------------------- #
          model = pkgs.nix-llama-model;
          model-small = pkgs.nix-llama-model-small;
          model-test = pkgs.nix-llama-model-test;
          model-trained = pkgs.nix-llama-model-trained;
          model-small-trained = pkgs.nix-llama-model.override {
            variant = "small";
            trained = true;
          };
          corpus = pkgs.nix-llama-corpus;
        }
        // lib.optionalAttrs (supportsCuda system) {
          nix-llama-cuda = pkgs.nix-llama.override { inherit ((cudaPkgsFor system)) llama-cpp; };
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # Container images. `container` streams the layered image (fast, no
          # intermediate tarball); `container-archive` writes one out for
          # `docker load`.
          container = pkgs.callPackage ./nix/container.nix {
            inherit version;
            inherit (pkgs) nix-llama;
            stream = true;
          };
          container-archive = pkgs.callPackage ./nix/container.nix {
            inherit version;
            inherit (pkgs) nix-llama;
            stream = false;
          };
          # The same image with the default model baked in at /model. Useful
          # where a multi-gigabyte volume is more trouble than a large image --
          # Kubernetes, in particular.
          container-model = pkgs.callPackage ./nix/container.nix {
            inherit version;
            inherit (pkgs) nix-llama;
            model = pkgs.nix-llama-model;
            stream = true;
          };
          # The runtime closure the Dockerfile copies out of its Nix stage, so
          # the Dockerfile and the flake ship the same binaries.
          container-closure = pkgs.callPackage ./nix/container.nix {
            inherit version;
            inherit (pkgs) nix-llama;
            closureOnly = true;
          };
        }
      );

      apps = eachSystem (
        pkgs: _system:
        let
          app = program: {
            type = "app";
            inherit program;
            meta.description = "nix-llama helper";
          };
        in
        {
          # `nix run github:heywoodlh/nix-llama` should serve something, not
          # ask for a model. This app carries the default model bundle; the
          # `nix-llama` app and the package deliberately do not, so building
          # the software does not pull 4.7 GB of weights.
          default = app "${
            pkgs.writeShellApplication {
              name = "nix-llama-serve";
              runtimeInputs = [ pkgs.nix-llama ];
              text = ''
                # --model/--model-dir on the command line still wins; this is
                # only the fallback for having been given nothing.
                export NIX_LLAMA_MODEL_DIR="''${NIX_LLAMA_MODEL_DIR:-${pkgs.nix-llama-model}}"
                exec nix-llama "$@"
              '';
            }
          }/bin/nix-llama-serve";

          # The bare program, with no model attached.
          nix-llama = app "${pkgs.nix-llama}/bin/nix-llama";
          probe = app "${pkgs.nix-llama}/bin/nix-llama-probe";
          rag = app "${pkgs.nix-llama}/bin/nix-llama-rag";

          update-sources = app "${
            pkgs.writeShellApplication {
              name = "nix-llama-update-sources";
              runtimeInputs = [
                pkgs.python3
                pkgs.nix
                pkgs.git
              ];
              text = ''
                exec python3 ${./nix/update-sources.py} "$@"
              '';
            }
          }/bin/nix-llama-update-sources";

          update-base-model = app "${
            pkgs.writeShellApplication {
              name = "nix-llama-update-base-model";
              runtimeInputs = [
                pkgs.python3
                pkgs.nix
              ];
              text = ''
                exec python3 ${./nix/update-base-model.py} "$@"
              '';
            }
          }/bin/nix-llama-update-base-model";
        }
      );

      checks = eachSystem (
        pkgs: system:
        import ./nix/checks.nix {
          inherit
            pkgs
            lib
            self
            system
            version
            ;
          nixosSystem = nixpkgs.lib.nixosSystem;
        }
      );

      devShells = eachSystem (
        pkgs: _system: {
          default = pkgs.mkShell {
            name = "nix-llama";
            packages = [
              pkgs.python3
              pkgs.python3Packages.pytest
              pkgs.ruff
              pkgs.nixfmt-rfc-style
              pkgs.statix
              pkgs.deadnix
              pkgs.shellcheck
              pkgs.actionlint
              pkgs.gitleaks
              pkgs.kubeconform
              pkgs.hadolint
              pkgs.jq
              pkgs.curl
              pkgs.nix-llama.llama-cpp
            ];
            shellHook = ''
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              export NIX_LLAMA_VERSION="${version}"
              echo "nix-llama ${version} dev shell"
              echo "  pytest                      run the unit tests"
              echo "  python -m nix_llama.cli probe"
              echo "  nix run .#update-sources    refresh the documentation corpus"
            '';
          };
        }
      );

      # Two shapes for each platform:
      #   `nix-llama` is the bare module, for configurations that already apply
      #   the overlay (or that pin their own nixpkgs instance).
      #   `default` additionally applies the overlay, which is what most people
      #   want -- but it conflicts with a read-only nixpkgs, as inside a NixOS
      #   test, so the bare module has to exist too.
      nixosModules = {
        nix-llama = ./modules/nixos.nix;
        default = {
          imports = [ ./modules/nixos.nix ];
          nixpkgs.overlays = [ self.overlays.default ];
        };
      };

      darwinModules = {
        nix-llama = ./modules/darwin.nix;
        default = {
          imports = [ ./modules/darwin.nix ];
          nixpkgs.overlays = [ self.overlays.default ];
        };
      };

      formatter = eachSystem (pkgs: _system: pkgs.nixfmt-rfc-style);
    };
}
