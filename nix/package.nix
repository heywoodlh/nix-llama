{
  lib,
  python3,
  makeWrapper,
  llama-cpp,
  version,
  src,
  # Extra programs the wrapper should expose to the probe. nvidia-smi is looked
  # up at runtime and is intentionally *not* a dependency: on NixOS it comes
  # from the driver, and in a container from the nvidia container toolkit.
  extraRuntimeInputs ? [ ],
}:

python3.pkgs.buildPythonApplication {
  pname = "nix-llama";
  inherit version src;

  pyproject = true;
  build-system = [ python3.pkgs.setuptools ];

  # No Python dependencies, by design: nix-llama runs on 8 GiB machines and
  # sits in front of a model server, so the standard library is the budget.
  dependencies = [ ];

  nativeBuildInputs = [ makeWrapper ];

  nativeCheckInputs = [ python3.pkgs.pytestCheckHook ];
  pytestFlags = [ "tests" ];

  postFixup = ''
    for program in nix-llama nix-llama-probe nix-llama-rag; do
      wrapProgram "$out/bin/$program" \
        --set-default NIX_LLAMA_VERSION ${version} \
        --prefix PATH : ${lib.makeBinPath ([ llama-cpp ] ++ extraRuntimeInputs)}
    done
  '';

  passthru = {
    inherit llama-cpp;
    # Which llama.cpp backend this build can actually use. The service module
    # reads it to pick a package, and `nix-llama doctor` reports it.
    backend =
      if llama-cpp.cudaSupport or false then
        "cuda"
      else if llama-cpp.rocmSupport or false then
        "rocm"
      else if llama-cpp.vulkanSupport or false then
        "vulkan"
      else if llama-cpp.metalSupport or false then
        "metal"
      else
        "cpu";
  };

  meta = {
    description = "Turnkey llama.cpp serving with hardware-aware defaults and doc retrieval";
    longDescription = ''
      nix-llama probes the machine it is running on -- accelerators, memory,
      CPU topology, cgroup limits, storage type -- reads the model's GGUF
      header, and derives a llama-server invocation sized for that combination.
      It also fronts llama-server with a retrieval proxy so the model can cite
      current documentation instead of guessing.
    '';
    homepage = "https://github.com/heywoodlh/nix-llama";
    license = lib.licenses.mit;
    mainProgram = "nix-llama";
    platforms = lib.platforms.unix;
  };
}
