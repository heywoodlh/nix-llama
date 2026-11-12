# The nix-llama overlay.
#
# Importing this gives a nixpkgs instance `nix-llama`, the model packages and
# the accelerated variants. The flake applies it for every system; downstream
# users can apply it to their own nixpkgs to get the same packages.
{
  version,
  # The flake's own source tree, and the nixpkgs flake input. Both are passed in
  # rather than read from a relative path so the overlay works when imported
  # from a store path.
  self,
  nixpkgsSrc,
}:

final: _prev:

let
  inherit (final) lib;

  # Only the files the Python build actually needs, so editing the README or a
  # workflow does not invalidate the build.
  pythonSrc = lib.fileset.toSource {
    root = self;
    fileset = lib.fileset.unions [
      (self + "/VERSION")
      (self + "/pyproject.toml")
      (self + "/README.md")
      (self + "/src")
      (self + "/tests")
    ];
  };

  mkNixLlama =
    llama-cpp:
    final.callPackage ./package.nix {
      inherit version llama-cpp;
      src = pythonSrc;
    };
in
{
  inherit version;

  nix-llama = mkNixLlama final.llama-cpp;

  # Accelerated variants. Each one is the same Python application wrapped
  # around a differently-built llama.cpp, so the service, the container and the
  # CLI all agree on behaviour and only the backend changes.
  nix-llama-cpu = mkNixLlama (
    final.llama-cpp.override {
      blasSupport = !final.stdenv.hostPlatform.isDarwin;
      metalSupport = final.stdenv.hostPlatform.isDarwin;
    }
  );

  nix-llama-vulkan = mkNixLlama (
    final.llama-cpp.override {
      vulkanSupport = true;
      blasSupport = false;
    }
  );

  nix-llama-rocm = mkNixLlama (
    final.llama-cpp.override {
      rocmSupport = true;
      blasSupport = false;
    }
  );

  # Documentation corpus, retrieval index and the model itself.
  nix-llama-sources = final.callPackage ./model/sources.nix {
    inherit nixpkgsSrc;
    declaration = ../nix/model/sources.json;
    lockfile = ../nix/model/sources.lock.json;
  };

  nix-llama-corpus = final.callPackage ./model/corpus.nix { inherit version; };

  # Fixture corpus for the test suite: same code path, three files instead of
  # every pinned repository.
  nix-llama-corpus-test = final.callPackage ./model/test-corpus.nix { inherit version; };

  nix-llama-model = final.callPackage ./model { inherit version; };

  nix-llama-model-small = final.nix-llama-model.override { variant = "small"; };

  nix-llama-model-test = final.nix-llama-model.override {
    variant = "test";
    corpus = final.nix-llama-corpus-test;
  };

  nix-llama-model-trained = final.nix-llama-model.override { trained = true; };
}
