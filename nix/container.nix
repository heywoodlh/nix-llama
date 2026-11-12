# Container images.
#
# The container runs the *same* binaries the service does: both come from this
# flake's nixpkgs, so there is no second llama.cpp build to drift. Three shapes:
#
#   stream = true    `streamLayeredImage` -- pipe straight into `docker load`
#   stream = false   a loadable tarball, for CI artefacts
#   closureOnly      the runtime closure plus a prepared /etc, which is what
#                    docker/Dockerfile copies out of its Nix build stage
{
  lib,
  dockerTools,
  buildEnv,
  closureInfo,
  runCommand,
  bashInteractive,
  coreutils,
  cacert,
  tzdata,
  nix-llama,
  version,
  stream ? true,
  closureOnly ? false,
  # Bake a model directory into the image at /model. Convenient for Kubernetes
  # and anywhere a 5 GB volume is more trouble than a 5 GB image; leave it null
  # to mount the model at runtime instead.
  model ? null,
}:

let
  name = if model != null then "nix-llama-model" else "nix-llama";

  # bash and coreutils are not needed to serve, but a container you cannot
  # `docker exec` into is a container you cannot debug. llama-server is on
  # PATH for the same reason -- and because `llama-server --version` inside the
  # image is how CI proves the image and the flake carry the same binary.
  rootEnv = buildEnv {
    name = "nix-llama-root";
    paths = [
      nix-llama
      nix-llama.llama-cpp
      bashInteractive
      coreutils
      cacert
    ];
    pathsToLink = [
      "/bin"
      "/etc"
      "/share"
    ];
  };

  # A minimal /etc. The server runs unprivileged, and glibc's resolver wants
  # nsswitch.conf to exist before it will look at /etc/hosts.
  etc = runCommand "nix-llama-container-etc" { } ''
    mkdir -p $out/etc $out/tmp $out/var/lib/nix-llama
    cat > $out/etc/passwd <<'EOF'
    root:x:0:0:root:/root:/bin/bash
    nix-llama:x:1000:1000:nix-llama:/var/lib/nix-llama:/bin/bash
    nobody:x:65534:65534:nobody:/nonexistent:/bin/false
    EOF
    cat > $out/etc/group <<'EOF'
    root:x:0:
    nix-llama:x:1000:
    nogroup:x:65534:
    EOF
    cat > $out/etc/nsswitch.conf <<'EOF'
    hosts: files dns
    EOF
    echo 'nix-llama' > $out/etc/hostname
    ln -s ${cacert}/etc/ssl/certs/ca-bundle.crt $out/etc/ssl-ca-bundle.crt
    chmod 1777 $out/tmp
  '';

  config = {
    Entrypoint = [ "${nix-llama}/bin/nix-llama" ];
    Cmd = [ "serve" ];
    WorkingDir = "/var/lib/nix-llama";
    User = "1000:1000";
    ExposedPorts."8080/tcp" = { };
    Env = [
      "PATH=${rootEnv}/bin"
      # Listen on all interfaces: inside a container the loopback-only default
      # would make the published port useless. Port publishing is the
      # operator's explicit act, which is what makes this safe here and not in
      # the service module.
      "NIX_LLAMA_HOST=0.0.0.0"
      "NIX_LLAMA_STATE_DIR=/var/lib/nix-llama"
      "NIX_LLAMA_MODEL_DIR=/model"
      "SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt"
      "TZDIR=${tzdata}/share/zoneinfo"
      "HOME=/var/lib/nix-llama"
      # llama.cpp must not reach out for weights at startup.
      "LLAMA_ARG_OFFLINE=1"
    ];
    Labels = {
      "org.opencontainers.image.title" = "nix-llama";
      "org.opencontainers.image.description" =
        "Turnkey llama.cpp server with hardware-aware defaults and documentation retrieval";
      "org.opencontainers.image.version" = version;
      "org.opencontainers.image.source" = "https://github.com/heywoodlh/nix-llama";
      "org.opencontainers.image.licenses" = "MIT";
    };
  };

  bakedModel =
    if model == null then
      null
    else
      runCommand "nix-llama-container-model" { } ''
        mkdir -p $out
        cp -r ${model} $out/model
      '';

  imageArgs = {
    inherit name config;
    tag = version;
    contents = [
      rootEnv
      etc
    ]
    ++ lib.optional (bakedModel != null) bakedModel;
    maxLayers = 64;
    # No timestamps, so the same inputs give the same image digest.
    created = "1970-01-01T00:00:01Z";
  };

  closure = closureInfo {
    rootPaths = [
      rootEnv
      etc
    ];
  };
in
if closureOnly then
  runCommand "nix-llama-container-closure-${version}" { } ''
    mkdir -p $out
    cp ${closure}/store-paths $out/store-paths
    ln -s ${rootEnv} $out/root
    ln -s ${etc} $out/etc
    ln -s ${nix-llama} $out/nix-llama
    cat > $out/entrypoint <<EOF
    ${nix-llama}/bin/nix-llama
    EOF
    cat > $out/env <<EOF
    ${lib.concatStringsSep "\n" config.Env}
    EOF
  ''
else if stream then
  dockerTools.streamLayeredImage imageArgs
else
  dockerTools.buildLayeredImage imageArgs
