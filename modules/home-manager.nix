# Home Manager module for the nix-llama service.
#
# The same option surface as the NixOS and nix-darwin modules, run as the user
# instead of a service account: a systemd user unit on Linux, a launchd agent
# on macOS. It is for machines whose system configuration you do not control
# -- a work Mac, a non-NixOS distribution -- or where a model server belongs to
# one person rather than the host.
#
# What does not carry over:
#   - `user`, `group` and `openFirewall` are system concerns. The first two are
#     ignored; the last is asserted against rather than silently dropped.
#   - Most of systemd's sandbox needs privilege or user namespaces a user
#     manager may not have. The seccomp-based half works unprivileged and is
#     kept; the filesystem half is not.
#   - No mlock rlimit is raised. A user unit cannot exceed the hard limit its
#     manager was given, and asking for more fails the unit outright. If the
#     planner picks `mmap+mlock` and the limit is too small, llama.cpp warns and
#     carries on unpinned; set `tuning.loadMode = "mmap"` to silence it.
#
# A function of the overlay rather than a plain module: the flake passes it, so
# the module can supply its own packages without setting `nixpkgs.overlays`,
# which Home Manager rejects under `useGlobalPkgs` -- how most people run it.
{
  overlay ? null,
}:

{
  config,
  lib,
  pkgs,
  ...
}@args:

let
  inherit (lib)
    escapeShellArg
    mkDefault
    mkIf
    mkMerge
    optional
    ;

  # Reuse the caller's packages when they already carry the overlay, so that a
  # pinned or patched nix-llama is respected and nixpkgs is not instantiated
  # twice.
  llamaPkgs = if overlay == null || pkgs ? nix-llama then pkgs else pkgs.extend overlay;

  cfg = config.services.nix-llama;
  settings = import ./settings.nix {
    inherit lib cfg;
    pkgs = llamaPkgs;
  };

  inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux;

  # `osConfig` exists only when Home Manager runs as a NixOS or nix-darwin
  # module. Standalone, there is no system configuration to infer from.
  inferredAcceleration = settings.inferAcceleration (args.osConfig or { });
  acceleration = if cfg.acceleration == "auto" then inferredAcceleration else cfg.acceleration;
  package = if cfg.package != null then cfg.package else settings.packageFor acceleration;

  stateDir = toString cfg.stateDir;
  logFile = "${config.home.homeDirectory}/Library/Logs/nix-llama.log";

  # One launcher for both service managers. launchd has no ExecStartPre and
  # refuses to start a job whose WorkingDirectory is missing, and exporting the
  # environment here sidesteps two different quoting schemes for values.
  launcher = llamaPkgs.writeShellScript "nix-llama-user-service" ''
    set -eu
    ${lib.toShellVars settings.environment}
    export ${lib.concatStringsSep " " (lib.attrNames settings.environment)}
    ${llamaPkgs.coreutils}/bin/mkdir -p -m 0700 ${escapeShellArg stateDir}
    cd ${escapeShellArg stateDir}
    exec ${package}/bin/nix-llama serve
  '';
in
{
  imports = [
    # options.nix defaults `model` to `pkgs.nix-llama-model`, which has to come
    # from the overlaid package set rather than the caller's.
    (optionArgs: import ./options.nix (optionArgs // { pkgs = llamaPkgs; }))
  ];

  config = mkIf cfg.enable (mkMerge [
    {
      assertions = [
        {
          assertion = !cfg.openFirewall;
          message = ''
            services.nix-llama.openFirewall cannot be honoured from Home Manager: the
            firewall belongs to the system. Open the port in the system configuration,
            and set apiKeyFile before you do.
          '';
        }
        {
          assertion = (cfg.tls.certFile == null) == (cfg.tls.keyFile == null);
          message = "services.nix-llama.tls needs both certFile and keyFile, or neither.";
        }
        {
          assertion = isLinux || isDarwin;
          message = "services.nix-llama runs under systemd or launchd only.";
        }
      ];

      warnings =
        optional (cfg.host != "127.0.0.1" && cfg.host != "localhost" && cfg.apiKeyFile == null) ''
          services.nix-llama.host is ${cfg.host} with no apiKeyFile: the model server will
          accept unauthenticated requests from anything that can route to it.
        ''
        ++ optional (isLinux && cfg.acceleration == "auto" && inferredAcceleration == "cpu") ''
          services.nix-llama could not infer an accelerator and will build a CPU-only
          llama.cpp. If this machine has a GPU, set services.nix-llama.acceleration
          explicitly -- "auto" reads the NixOS configuration (when Home Manager runs
          inside one), not your PCI bus.
        '';

      # /var/lib is not the user's to write.
      services.nix-llama.stateDir = mkDefault "${config.xdg.stateHome}/nix-llama";

      home.packages = [ package ];

      # The CLI looks here after NIX_LLAMA_CONFIG, so `nix-llama plan` and
      # `doctor` in a shell describe this service rather than asking for a model.
      xdg.configFile."nix-llama/config.json".source = settings.configFile;
    }

    (mkIf isLinux {
      systemd.user.services.nix-llama = {
        Unit = {
          Description = "nix-llama model server";
          Documentation = [ "https://github.com/heywoodlh/nix-llama" ];
        };

        Service = {
          ExecStart = "${launcher}";
          Type = "exec";
          Restart = "on-failure";
          RestartSec = 5;
          TimeoutStopSec = 30;

          # Loading a multi-gigabyte model is I/O heavy but not urgent; keep it
          # from starving the desktop session it shares a user with.
          IOSchedulingClass = "best-effort";
          IOSchedulingPriority = 5;
          OOMPolicy = "continue";

          # --- hardening -------------------------------------------------- #
          # Only what is enforced by seccomp, which an unprivileged manager can
          # apply. ProtectSystem/ProtectHome and friends need user namespaces
          # and would, in any case, hide the home directory stateDir lives in.
          NoNewPrivileges = true;
          RestrictAddressFamilies = [
            "AF_INET"
            "AF_INET6"
            "AF_UNIX"
          ];
          RestrictRealtime = true;
          RestrictSUIDSGID = true;
          LockPersonality = true;
          SystemCallArchitectures = "native";
          # `~@resources` is deliberately absent, as in the system unit: llama.cpp's
          # threadpool calls sched_setaffinity and setpriority while loading.
          SystemCallFilter = [
            "@system-service"
            "~@privileged"
          ];
          UMask = "0077";
        };

        Install.WantedBy = [ "default.target" ];
      };
    })

    (mkIf isDarwin {
      launchd.agents.nix-llama = {
        enable = true;
        config = {
          ProgramArguments = [ "${launcher}" ];
          RunAtLoad = true;
          KeepAlive = {
            SuccessfulExit = false;
            Crashed = true;
          };
          ThrottleInterval = 5;
          StandardOutPath = logFile;
          StandardErrorPath = logFile;
          ProcessType = "Interactive";
          # Metal maps the weights into unified memory; the default
          # file-descriptor limit is too tight for a multi-gigabyte model.
          SoftResourceLimits.NumberOfFiles = 8192;
        };
      };
    })
  ]);
}
