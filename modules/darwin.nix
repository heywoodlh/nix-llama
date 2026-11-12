# nix-darwin module for the nix-llama service.
#
# Same options as the NixOS module; launchd instead of systemd. The two
# meaningful differences are that launchd has no ExecStartPre (so the state
# directory is created by an activation script) and that nix-darwin needs a uid
# to create a service account.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  inherit (lib)
    mkIf
    mkOption
    optional
    types
    ;

  cfg = config.services.nix-llama;
  settings = import ./settings.nix { inherit lib pkgs cfg; };

  # Every Mac that can run this has a GPU Metal can address, and nixpkgs builds
  # llama.cpp with Metal on Darwin by default.
  acceleration = if cfg.acceleration == "auto" then "metal" else cfg.acceleration;
  package = if cfg.package != null then cfg.package else settings.packageFor acceleration;

  logFile = "/var/log/nix-llama.log";
in
{
  imports = [ ./options.nix ];

  options.services.nix-llama = {
    uid = mkOption {
      type = types.int;
      default = 541;
      description = ''
        uid for the service account. nix-darwin cannot allocate one
        automatically, so it has to be stated; pick anything free in the
        system range (500-999) that no other nix-darwin service is using.
      '';
    };

    gid = mkOption {
      type = types.int;
      default = 541;
      description = "gid for the service group.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = (cfg.tls.certFile == null) == (cfg.tls.keyFile == null);
        message = "services.nix-llama.tls needs both certFile and keyFile, or neither.";
      }
      {
        assertion = cfg.user != "root";
        message = ''
          services.nix-llama.user must not be root. The service parses untrusted
          HTTP input and loads multi-gigabyte files from disk; it does not need,
          and should not have, system privileges.
        '';
      }
    ];

    warnings =
      optional (cfg.host != "127.0.0.1" && cfg.host != "localhost" && cfg.apiKeyFile == null)
        ''
          services.nix-llama.host is ${cfg.host} with no apiKeyFile: the model server will
          accept unauthenticated requests from anything on your network.
        '';

    users = {
      # nix-darwin only manages accounts it has been told it owns.
      knownUsers = [ cfg.user ];
      knownGroups = [ cfg.group ];

      users.${cfg.user} = {
        inherit (cfg) uid gid;
        home = cfg.stateDir;
        createHome = false;
        description = "nix-llama model server";
        shell = "/usr/bin/false";
      };

      groups.${cfg.group} = {
        inherit (cfg) gid;
        description = "nix-llama model server";
      };
    };

    environment.systemPackages = [ package ];

    # Installed at a well-known path so `nix-llama probe`, `plan` and `doctor`
    # typed into a shell describe the configured service instead of asking for
    # a model. The service itself reads it through NIX_LLAMA_CONFIG.
    environment.etc."nix-llama/config.json".source = settings.configFile;

    # launchd has no equivalent of ExecStartPre, so the state directory and the
    # log file are created here, at activation, while we are still root.
    system.activationScripts.nix-llama.text = ''
      echo "setting up nix-llama state directory..." >&2
      mkdir -p '${cfg.stateDir}'
      chown ${toString cfg.uid}:${toString cfg.gid} '${cfg.stateDir}'
      chmod 0750 '${cfg.stateDir}'
      touch '${logFile}'
      chown ${toString cfg.uid}:${toString cfg.gid} '${logFile}'
      chmod 0640 '${logFile}'
    '';

    launchd.daemons.nix-llama = {
      script = ''
        exec ${package}/bin/nix-llama serve
      '';

      serviceConfig = {
        Label = "org.nixos.nix-llama";
        RunAtLoad = true;
        KeepAlive = {
          SuccessfulExit = false;
          Crashed = true;
        };
        ThrottleInterval = 5;
        StandardOutPath = logFile;
        StandardErrorPath = logFile;
        UserName = cfg.user;
        GroupName = cfg.group;
        WorkingDirectory = toString cfg.stateDir;
        EnvironmentVariables = settings.environment;
        ProcessType = "Interactive";
        # Metal maps the weights into unified memory; the default file-descriptor
        # and memory limits are too tight for a multi-gigabyte model.
        SoftResourceLimits = {
          NumberOfFiles = 8192;
        };
      };
    };
  };
}
