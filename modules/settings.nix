# Shared service wiring: render the JSON config and pick the right package.
#
# Imported by both the NixOS and nix-darwin modules so the two cannot drift.
# Null-valued options are dropped rather than serialised, which leaves the
# built-in defaults in `nix_llama.config` authoritative for anything the
# operator has not set -- one place where a default lives, not two.
{
  lib,
  pkgs,
  cfg,
}:

let
  inherit (lib) filterAttrs;

  dropNulls = attrs: filterAttrs (_: v: v != null) attrs;

  modelPath = toString cfg.model;
  modelIsFile = lib.hasSuffix ".gguf" modelPath;

  systemPromptValue =
    if cfg.systemPrompt == null then
      null
    else if builtins.isPath cfg.systemPrompt || lib.hasPrefix "/" (toString cfg.systemPrompt) then
      toString cfg.systemPrompt
    else
      toString (pkgs.writeText "nix-llama-system-prompt.md" cfg.systemPrompt);

  settings = {
    inherit (cfg) host port;
    model = if modelIsFile then modelPath else null;
    modelDir = if modelIsFile then null else modelPath;
    alias = "nix-llama";
    inherit (cfg) acceleration;
    apiKeyFile = if cfg.apiKeyFile == null then null else toString cfg.apiKeyFile;
    stateDir = toString cfg.stateDir;
    systemPrompt = systemPromptValue;
    llamaServer = "llama-server";

    tuning = dropNulls {
      inherit (cfg.tuning)
        contextSize
        maxContextSize
        minContextSize
        gpuLayers
        threads
        threadsBatch
        batchSize
        ubatchSize
        parallel
        kvCacheType
        promptCacheMiB
        speculative
        contextShift
        flashAttention
        corsOrigins
        metrics
        webui
        ;
      loadMode = cfg.tuning.loadMode;
    };

    diskCache = dropNulls {
      inherit (cfg.diskCache) enable minFreeGiB;
      path = if cfg.diskCache.path == null then null else toString cfg.diskCache.path;
    };

    rag = dropNulls {
      inherit (cfg.rag)
        enable
        topK
        maxChars
        minScore
        injectInto
        upstreamPort
        ;
      index = if cfg.rag.index == null then null else toString cfg.rag.index;
    };

    tls = dropNulls {
      certFile = if cfg.tls.certFile == null then null else toString cfg.tls.certFile;
      keyFile = if cfg.tls.keyFile == null then null else toString cfg.tls.keyFile;
    };

    # Values arrive as bool/int/str from the module system; the planner renders
    # `true` as a bare flag and everything else as `--flag value`.
    settings = dropNulls cfg.settings;
    inherit (cfg) extraFlags;
  };
in
rec {
  config = dropNulls settings;

  configFile = pkgs.writeText "nix-llama.json" (builtins.toJSON config);

  # The accelerated variants exposed by the overlay, keyed by the
  # `acceleration` option. "metal" is the Darwin default build, so it maps to
  # the plain package.
  packages = {
    none = pkgs.nix-llama-cpu;
    cpu = pkgs.nix-llama-cpu;
    cuda = pkgs.nix-llama-cuda or pkgs.nix-llama;
    rocm = pkgs.nix-llama-rocm;
    vulkan = pkgs.nix-llama-vulkan;
    metal = pkgs.nix-llama;
  };

  packageFor = accel: packages.${accel} or pkgs.nix-llama;

  # `acceleration = "auto"` resolves at evaluation time, so it reads the
  # configuration rather than the hardware. The runtime probe is what catches a
  # mismatch -- it asks llama-server which devices it can actually see and warns
  # when real hardware is going unused.
  #
  # `sysConfig` is the NixOS configuration: the system module passes its own,
  # Home Manager passes `osConfig`, or `{ }` when it runs standalone and there
  # is nothing to read.
  inferAcceleration =
    sysConfig:
    if pkgs.stdenv.hostPlatform.isDarwin then
      "metal"
    else if pkgs.config.cudaSupport or false then
      "cuda"
    else if sysConfig.hardware.nvidia.modesetting.enable or false then
      "cuda"
    else if lib.elem "nvidia" (sysConfig.services.xserver.videoDrivers or [ ]) then
      "cuda"
    else if pkgs.config.rocmSupport or false then
      "rocm"
    else if sysConfig.hardware.amdgpu.opencl.enable or false then
      "rocm"
    else if sysConfig.hardware.graphics.enable or false then
      "vulkan"
    else
      "cpu";

  # Environment shared by both service managers.
  environment = {
    NIX_LLAMA_CONFIG = toString configFile;
    NIX_LLAMA_VERSION = pkgs.nix-llama.version;
    # llama.cpp would otherwise try to reach Hugging Face for a missing model;
    # a packaged service has no business making network calls at startup.
    LLAMA_ARG_OFFLINE = "1";
    HOME = toString cfg.stateDir;
  }
  // cfg.environment;
}
