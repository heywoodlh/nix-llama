# NixOS module for the nix-llama service.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  inherit (lib)
    mkIf
    mkMerge
    optional
    optionals
    ;

  cfg = config.services.nix-llama;
  settings = import ./settings.nix { inherit lib pkgs cfg; };

  # `acceleration = "auto"` resolves at evaluation time, so it reads the
  # configuration rather than the hardware. The runtime probe is what catches a
  # mismatch -- it asks llama-server which devices it can actually see and warns
  # when real hardware is going unused.
  inferredAcceleration =
    if pkgs.config.cudaSupport or false then
      "cuda"
    else if config.hardware.nvidia.modesetting.enable or false then
      "cuda"
    else if lib.elem "nvidia" (config.services.xserver.videoDrivers or [ ]) then
      "cuda"
    else if pkgs.config.rocmSupport or false then
      "rocm"
    else if config.hardware.amdgpu.opencl.enable or false then
      "rocm"
    else if config.hardware.graphics.enable or false then
      "vulkan"
    else
      "cpu";

  acceleration = if cfg.acceleration == "auto" then inferredAcceleration else cfg.acceleration;
  package = if cfg.package != null then cfg.package else settings.packageFor acceleration;

  usesGpu = acceleration != "cpu" && acceleration != "none";

  # The planner may choose `mmap+mlock`, which needs the capability and the
  # rlimit; granting them unconditionally is cheaper than guessing.
  wantsMlock = cfg.tuning.loadMode == null || lib.hasInfix "mlock" (cfg.tuning.loadMode or "");

  # One place where capabilities are decided, so the hardening block below can
  # keep the bounding set equal to what is actually granted.
  ambientCapabilities =
    optionals wantsMlock [ "CAP_IPC_LOCK" ] ++ optionals (cfg.port < 1024) [ "CAP_NET_BIND_SERVICE" ];
in
{
  imports = [ ./options.nix ];

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = !cfg.openFirewall || cfg.apiKeyFile != null;
        message = ''
          services.nix-llama.openFirewall is enabled without services.nix-llama.apiKeyFile.
          An open port in front of an unauthenticated llama-server lets anyone who can
          reach it spend your GPU and read anything the model has been given. Set
          apiKeyFile, or terminate TLS and authentication in a reverse proxy and leave
          the firewall closed.
        '';
      }
      {
        assertion = (cfg.tls.certFile == null) == (cfg.tls.keyFile == null);
        message = "services.nix-llama.tls needs both certFile and keyFile, or neither.";
      }
    ];

    warnings =
      optional (cfg.host != "127.0.0.1" && cfg.host != "localhost" && cfg.apiKeyFile == null) ''
        services.nix-llama.host is ${cfg.host} with no apiKeyFile: the model server will
        accept unauthenticated requests from anything that can route to it.
      ''
      ++ optional (cfg.acceleration == "auto" && inferredAcceleration == "cpu") ''
        services.nix-llama could not infer an accelerator from your configuration and
        will build a CPU-only llama.cpp. If this machine has a GPU, set
        services.nix-llama.acceleration explicitly -- "auto" reads your NixOS
        configuration, not your PCI bus.
      '';

    users.users.${cfg.user} = mkIf (cfg.user == "nix-llama") {
      isSystemUser = true;
      inherit (cfg) group;
      home = cfg.stateDir;
      description = "nix-llama model server";
      # Access to /dev/dri and /dev/kfd is group-gated.
      extraGroups = optionals usesGpu [
        "video"
        "render"
      ];
    };

    users.groups.${cfg.group} = mkIf (cfg.group == "nix-llama") { };

    networking.firewall = mkIf cfg.openFirewall { allowedTCPPorts = [ cfg.port ]; };

    environment.systemPackages = [ package ];

    # Installed at a well-known path so `nix-llama probe`, `plan` and `doctor`
    # typed into a shell describe the configured service instead of asking for
    # a model. The service itself reads it through NIX_LLAMA_CONFIG.
    environment.etc."nix-llama/config.json".source = settings.configFile;

    systemd.services.nix-llama = {
      description = "nix-llama model server";
      documentation = [ "https://github.com/heywoodlh/nix-llama" ];
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      inherit (settings) environment;

      serviceConfig = mkMerge [
        {
          ExecStart = "${package}/bin/nix-llama serve";
          Type = "exec";
          User = cfg.user;
          Group = cfg.group;
          Restart = "on-failure";
          RestartSec = 5;
          TimeoutStopSec = 30;

          StateDirectory = mkIf (cfg.stateDir == "/var/lib/nix-llama") "nix-llama";
          StateDirectoryMode = "0750";
          WorkingDirectory = cfg.stateDir;
          ReadWritePaths = [ cfg.stateDir ];

          # Loading a multi-gigabyte model is I/O heavy but not urgent; keep it
          # from starving interactive work on a workstation.
          IOSchedulingClass = "best-effort";
          IOSchedulingPriority = 5;
          OOMPolicy = "continue";

          # --- hardening -------------------------------------------------- #
          NoNewPrivileges = true;
          PrivateTmp = true;
          PrivateMounts = true;
          ProtectSystem = "strict";
          ProtectHome = true;
          ProtectProc = "invisible";
          ProcSubset = "pid";
          ProtectClock = true;
          ProtectHostname = true;
          ProtectKernelLogs = true;
          ProtectKernelModules = true;
          ProtectKernelTunables = true;
          ProtectControlGroups = true;
          RestrictAddressFamilies = [
            "AF_INET"
            "AF_INET6"
            "AF_UNIX"
          ];
          RestrictNamespaces = true;
          RestrictRealtime = true;
          RestrictSUIDSGID = true;
          LockPersonality = true;
          SystemCallArchitectures = "native";
          # `~@resources` is deliberately absent: it denies sched_setaffinity
          # and setpriority, which llama.cpp's threadpool calls on startup, and
          # the service dies with SIGSYS the moment the model begins loading.
          # An inference server setting its own thread affinity is its job.
          SystemCallFilter = [
            "@system-service"
            "~@privileged"
          ];
          UMask = "0077";
          # GPU backends JIT-compile kernels, so W^X cannot be enforced.
          MemoryDenyWriteExecute = false;
        }
        {
          AmbientCapabilities = ambientCapabilities;
          CapabilityBoundingSet = ambientCapabilities;
        }
        (mkIf wantsMlock { LimitMEMLOCK = "infinity"; })
        (mkIf usesGpu {
          # Accelerators need their device nodes; everything else stays denied.
          PrivateDevices = false;
          DeviceAllow = [
            "/dev/dri rw"
            "/dev/nvidia0 rw"
            "/dev/nvidia1 rw"
            "/dev/nvidiactl rw"
            "/dev/nvidia-uvm rw"
            "/dev/nvidia-uvm-tools rw"
            "/dev/nvidia-modeset rw"
            "/dev/kfd rw"
          ];
          SupplementaryGroups = [
            "video"
            "render"
          ];
        })
        (mkIf (!usesGpu) { PrivateDevices = true; })
        (mkIf (cfg.apiKeyFile != null) { LoadCredential = [ "api-key:${cfg.apiKeyFile}" ]; })
      ];
    };
  };
}
