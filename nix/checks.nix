# `nix flake check` -- everything CI gates a pull request on.
#
# Checks are cheap-first: lint and unit tests fail in seconds, the smoke test
# that actually boots llama-server takes about a minute, and the NixOS VM test
# is the slowest. Nothing here reaches the network, so a check that passes
# locally passes in CI.
{
  pkgs,
  lib,
  self,
  system,
  version,
  nixosSystem,
}:

let
  inherit (pkgs) runCommand;

  inherit (pkgs.stdenv.hostPlatform) isDarwin isLinux;

  # The flake source is already free of .git and build outputs, so the lint
  # checks can take it as-is.
  treeSrc = self;

  check =
    name: deps: script:
    runCommand "nix-llama-check-${name}"
      {
        nativeBuildInputs = deps;
        src = treeSrc;
      }
      ''
        cp -r $src/. .
        chmod -R u+w .
        ${script}
        touch $out
      '';

  testModel = pkgs.nix-llama-model-test;
in
{
  # Unit tests run inside the package build (pytestCheckHook), so depending on
  # the package is the same as running them.
  package = pkgs.nix-llama;

  python-lint = check "python-lint" [ pkgs.ruff ] ''
    ruff check src tests training nix
    ruff format --check --diff src tests training nix
  '';

  nix-lint =
    check "nix-lint"
      [
        pkgs.nixfmt-rfc-style
        pkgs.statix
        pkgs.deadnix
      ]
      ''
        nixfmt --check $(find . -name '*.nix' -not -path './.git/*')
        statix check .
        deadnix --fail .
      '';

  shell-lint = check "shell-lint" [ pkgs.shellcheck ] ''
    scripts=$(find . -name '*.sh' -not -path './.git/*')
    if [ -n "$scripts" ]; then
      # shellcheck disable=SC2086 # deliberate word splitting over the file list
      shellcheck --severity=style $scripts
    else
      echo "no shell scripts to check"
    fi
  '';

  workflow-lint = check "workflow-lint" [ pkgs.actionlint ] ''
    actionlint -color .github/workflows/*.yml
  '';

  dockerfile-lint = check "dockerfile-lint" [ pkgs.hadolint ] ''
    hadolint docker/Dockerfile
  '';

  secret-scan = check "secret-scan" [ pkgs.gitleaks ] ''
    gitleaks detect --no-git --source . --redact --verbose \
      --config .gitleaks.toml --exit-code 1
  '';

  # Policy checks that would otherwise only be caught by a human reading a
  # diff. Each one encodes a decision we do not want silently reversed.
  security-policy = check "security-policy" [ pkgs.ripgrep ] ''
    fail() { echo "SECURITY POLICY: $1" >&2; exit 1; }

    # llama.cpp's built-in agent tools include exec_shell_command. Enabling
    # them by default would turn every nix-llama deployment into remote code
    # execution for anyone who can reach the port.
    # -g excludes this file: the patterns below would otherwise match their
    # own definitions.
    if rg -n -g '!checks.nix' -- '--tools|--agent|-ag\b|--ui-mcp-proxy|--mcp-servers-json' \
         src/ modules/ nix/ docker/ 2>/dev/null; then
      fail "llama.cpp agent/tool flags must never be set by nix-llama itself"
    fi

    # The service must not listen beyond loopback unless the operator says so.
    rg -q '"host": "127.0.0.1"' src/nix_llama/config.py \
      || fail "the default listen address must stay 127.0.0.1"
    rg -q 'default = "127.0.0.1"' modules/options.nix \
      || fail "the module default listen address must stay 127.0.0.1"

    # A CORS wildcard with credentials reflected back is reachable from any
    # page in the user's browser.
    rg -q '"corsOrigins": "localhost"' src/nix_llama/config.py \
      || fail "the default CORS origin must stay localhost"

    # openFirewall without authentication is asserted against in the module.
    rg -q 'assertion = !cfg.openFirewall \|\| cfg.apiKeyFile != null' modules/nixos.nix \
      || fail "the openFirewall/apiKeyFile assertion must not be removed"

    # Secrets belong in files read at runtime, never in the store.
    if rg -n -g '!checks.nix' 'apiKey\s*=\s*mkOption' modules/ 2>/dev/null; then
      fail "expose apiKeyFile, not an inline apiKey: option values land in the Nix store"
    fi

    # shell=True anywhere in a request path is a command-injection surface.
    if rg -n -g '!checks.nix' 'shell\s*=\s*True' src/ 2>/dev/null; then
      fail "subprocess calls must not use shell=True"
    fi

    echo "security policy checks passed"
  '';

  # The version in VERSION, the flake and the changelog must agree, or a
  # release tags artefacts that disagree about what they are.
  version-consistency = check "version-consistency" [ pkgs.ripgrep ] ''
    file_version="$(cat VERSION)"
    if [ "$file_version" != "${version}" ]; then
      echo "VERSION says $file_version but the flake computed ${version}" >&2
      exit 1
    fi
    rg -q "^## \[$file_version\]" CHANGELOG.md \
      || { echo "CHANGELOG.md has no section for $file_version" >&2; exit 1; }
    case "$file_version" in
      [0-9]*.[0-9]*.[0-9]*) ;;
      *) echo "VERSION must be semantic (MAJOR.MINOR.PATCH), got $file_version" >&2; exit 1 ;;
    esac
    echo "version $file_version is consistent"
  '';

  # The Dockerfile and nix/container.nix both describe the runtime environment.
  # They are separate files, so they can disagree -- this makes disagreeing a
  # CI failure instead of a support ticket.
  container-parity =
    let
      closure = pkgs.callPackage ./container.nix {
        inherit version;
        inherit (pkgs) nix-llama;
        closureOnly = true;
      };
    in
    check "container-parity" [ pkgs.ripgrep ] ''
      # Environment variables the Dockerfile sets, excluding PATH (which the
      # image layout defines rather than the Nix expression).
      rg -o '^\s+(NIX_LLAMA_[A-Z_]+|HOME|LLAMA_ARG_[A-Z_]+)=\S+' docker/Dockerfile \
        | tr -d ' \\' | sort -u > dockerfile-env.txt

      grep -oE '^(NIX_LLAMA_[A-Z_]+|HOME|LLAMA_ARG_[A-Z_]+)=.*' ${closure}/env \
        | sort -u > nix-env.txt

      if ! diff -u nix-env.txt dockerfile-env.txt; then
        echo "" >&2
        echo "docker/Dockerfile and nix/container.nix disagree about the runtime" >&2
        echo "environment. Update both, or the container will not behave like the" >&2
        echo "service does." >&2
        exit 1
      fi

      # The entrypoint must be the same program, not a lookalike.
      rg -q 'ENTRYPOINT \["/bin/nix-llama"\]' docker/Dockerfile \
        || { echo "the Dockerfile entrypoint must be /bin/nix-llama" >&2; exit 1; }
      rg -q 'USER 1000:1000' docker/Dockerfile \
        || { echo "the container must not run as root" >&2; exit 1; }

      echo "Dockerfile and container.nix agree"
    '';

  # Structural validation only: kubeconform fetches JSON schemas over the
  # network, which a Nix build cannot do. The CI workflow runs the full
  # schema check, where network access is available.
  kubernetes-manifests =
    check "kubernetes-manifests"
      [
        pkgs.yq-go
        pkgs.jq
      ]
      ''
        yq -o=json '.' examples/kubernetes.yaml > manifests.json

        jq -se '
          length == 2
          and (map(.kind) | sort) == ["Deployment", "Service"]
        ' manifests.json > /dev/null || { echo "expected one Service and one Deployment" >&2; exit 1; }

        jq -se '
          (map(select(.kind == "Deployment")) | first) as $d
          | $d.spec.template.spec.securityContext.runAsNonRoot == true
          and ($d.spec.template.spec.containers | length) == 1
          and ($d.spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation == false)
          and ($d.spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem == true)
          and ($d.spec.template.spec.containers[0].resources.limits.memory != null)
          and ($d.spec.template.spec.containers[0].startupProbe != null)
        ' manifests.json > /dev/null \
          || { echo "the example Deployment lost a hardening or sizing field" >&2; exit 1; }

        # The example must stay CPU-only: accelerators in Kubernetes depend on the
        # cluster, and a half-explained GPU example is worse than none.
        if jq -e 'map(.. | objects | select(has("nvidia.com/gpu"))) | length > 0' manifests.json; then
          echo "the Kubernetes example must not claim accelerator resources" >&2
          exit 1
        fi

        echo "Kubernetes example validated"
      '';

  # Licence policy: every documentation source must carry a licence the
  # allowlist accepts, and share-alike sources must not reach training data.
  # This is the check that keeps "never train on material we lack permission
  # for" true as sources are added.
  corpus-licenses = check "corpus-licenses" [ pkgs.python3 ] ''
    PYTHONPATH=src python3 - <<'PY'
    import json
    from nix_llama.rag.corpus import LICENSE_ALLOWLIST, SHARE_ALIKE

    declared = json.load(open("nix/model/sources.json"))["sources"]
    problems = []
    for source in declared:
        name, license_id = source["name"], source["license"].upper()
        usage = source.get("usage", ["retrieval"])
        if license_id not in LICENSE_ALLOWLIST:
            problems.append(f"{name}: licence {license_id} is not on the allowlist")
        if license_id in SHARE_ALIKE and "training" in usage:
            problems.append(
                f"{name}: {license_id} is share-alike and must be retrieval-only"
            )
        if not usage:
            problems.append(f"{name}: declares no usage")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"{len(declared)} documentation sources pass the licence policy")
    PY
  '';

  # Boot the real thing: probe, plan, llama-server, the retrieval proxy, and
  # the OpenAI-compatible surface a harness actually calls. Uses the 1 MB test
  # model so this costs a minute rather than a download.
  serve-smoke =
    runCommand "nix-llama-check-serve-smoke"
      {
        nativeBuildInputs = [
          pkgs.nix-llama
          pkgs.curl
          pkgs.jq
        ];
      }
      ''
        export HOME="$TMPDIR"
        export NIX_LLAMA_STATE_DIR="$TMPDIR/state"
        mkdir -p "$NIX_LLAMA_STATE_DIR"
        port=18080

        echo "== probe"
        nix-llama probe
        nix-llama probe --format json | jq -e '.cpu.available >= 1' > /dev/null

        echo "== plan"
        nix-llama plan --model-dir ${testModel} --port "$port"
        nix-llama plan --model-dir ${testModel} --format json \
          | jq -e '.argv | index("--model") != null' > /dev/null

        echo "== doctor"
        nix-llama doctor --model-dir ${testModel}

        echo "== serve"
        # No --ctx-size override: the fixture model's trained context is 2048,
        # and the planner should arrive there on its own.
        nix-llama serve --model-dir ${testModel} --port "$port" \
          > "$TMPDIR/server.log" 2>&1 &
        server=$!
        # Dump the server log on any failure: a smoke test that fails without
        # showing why costs more time than it saves.
        trap 'status=$?; kill $server 2>/dev/null || true;
              if [ $status -ne 0 ]; then
                echo "--- server log ---" >&2; cat "$TMPDIR/server.log" >&2 || true;
              fi' EXIT

        for _ in $(seq 1 120); do
          if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then break; fi
          if ! kill -0 $server 2>/dev/null; then
            echo "server died:"; cat "$TMPDIR/server.log"; exit 1
          fi
          sleep 1
        done

        curl -sf "http://127.0.0.1:$port/health" | jq -e '.status == "ok"' > /dev/null
        echo "   health ok"

        curl -sf "http://127.0.0.1:$port/v1/models" | jq -e '.models | length >= 1' > /dev/null
        echo "   /v1/models relayed"

        curl -sf "http://127.0.0.1:$port/v1/rag/stats" | jq -e '.chunks > 0' > /dev/null
        echo "   retrieval index loaded"

        curl -sf "http://127.0.0.1:$port/v1/rag/search?q=nixos+option+enable" \
          | jq -e '.results | length >= 1' > /dev/null
        echo "   retrieval returns hits"

        curl -sf -X POST "http://127.0.0.1:$port/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d '{"messages":[{"role":"user","content":"how do I enable a service in NixOS?"}],"max_tokens":4}' \
          | jq -e '.choices[0].message.content != null' > /dev/null
        echo "   chat completion through the proxy"

        curl -sfN -X POST "http://127.0.0.1:$port/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}' \
          | grep -q '^data: ' 
        echo "   streaming relayed"

        grep -q 'augmented request with' "$TMPDIR/server.log" \
          || { echo "retrieval never fired"; cat "$TMPDIR/server.log"; exit 1; }
        echo "   request was augmented with retrieved documentation"

        kill $server 2>/dev/null || true
        touch $out
      '';

  # The MCP transport a harness speaks, exercised over real stdio.
  mcp-protocol =
    runCommand "nix-llama-check-mcp"
      {
        nativeBuildInputs = [
          pkgs.nix-llama
          pkgs.jq
        ];
      }
      ''
        cat > requests.jsonl <<'EOF'
        {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}
        {"jsonrpc":"2.0","method":"notifications/initialized"}
        {"jsonrpc":"2.0","id":2,"method":"tools/list"}
        {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_docs","arguments":{"query":"enable a nixos service"}}}
        {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"list_doc_sources","arguments":{}}}
        EOF

        nix-llama-rag --index ${testModel}/rag/index.json.gz mcp < requests.jsonl > responses.jsonl

        test "$(wc -l < responses.jsonl)" -eq 4 \
          || { echo "expected 4 responses, got $(wc -l < responses.jsonl)"; cat responses.jsonl; exit 1; }
        jq -se '.[0].result.serverInfo.name == "nix-llama-docs"' responses.jsonl > /dev/null
        jq -se '.[1].result.tools | length == 2' responses.jsonl > /dev/null
        jq -se '.[2].result.content[0].text | length > 0' responses.jsonl > /dev/null
        jq -se '.[3].result.content[0].text | test("chunks")' responses.jsonl > /dev/null
        echo "MCP initialize/tools-list/tools-call all behaved"
        touch $out
      '';

  # Home Manager is not a flake input -- there is exactly one -- so the module
  # is evaluated against stubs of the handful of Home Manager options it sets.
  # That is enough to catch a broken option path, a lost assertion or a unit
  # that no longer starts the right thing; real Home Manager types are
  # exercised by anyone importing the module.
  #
  # Bare nixpkgs, deliberately: the overlay-less path, where the module extends
  # `pkgs` itself, is the one most users of `homeManagerModules.default` take.
  home-manager-module-eval =
    let
      inherit (lib) mkOption types;
      anything = mkOption { type = types.anything; };
      attrs = mkOption {
        type = types.attrsOf types.anything;
        default = { };
      };
      list = mkOption {
        type = types.listOf types.anything;
        default = [ ];
      };

      evaluate =
        extra:
        lib.evalModules {
          modules = [
            self.homeManagerModules.default
            {
              options = {
                assertions = list;
                warnings = list;
                home.homeDirectory = anything;
                home.packages = list;
                xdg.stateHome = anything;
                xdg.configFile = attrs;
                systemd.user.services = attrs;
                launchd.agents = attrs;
              };
              config = {
                _module.args.pkgs = import pkgs.path { inherit system; };
                home.homeDirectory = "/home/alice";
                xdg.stateHome = "/home/alice/.local/state";
                services.nix-llama = {
                  enable = true;
                  # A path string, so the check does not pull the model bundle.
                  model = "/home/alice/models/example.gguf";
                  rag.topK = 8;
                  # A stand-in: this checks the wiring, and building llama.cpp
                  # just to grep a store path out of a script would be waste.
                  package = stubPackage;
                };
              };
            }
            extra
          ];
        };

      stubPackage = pkgs.writeShellScriptBin "nix-llama" "exit 0";

      hm = (evaluate { }).config;
      failedAssertions = cfg: map (a: a.message) (lib.filter (a: !a.assertion) cfg.assertions);

      unit = hm.systemd.user.services.nix-llama;
      agent = hm.launchd.agents.nix-llama;
      launcher = if isLinux then unit.Service.ExecStart else lib.head agent.config.ProgramArguments;
    in
    assert lib.assertMsg (failedAssertions hm == [ ]) (lib.concatLines (failedAssertions hm));
    assert lib.assertMsg (
      failedAssertions (evaluate { services.nix-llama.openFirewall = true; }).config != [ ]
    ) "openFirewall must be rejected under Home Manager";
    assert lib.assertMsg (
      isLinux -> unit.Install.WantedBy == [ "default.target" ] && unit.Service.Restart == "on-failure"
    ) "the systemd user unit is no longer started with the session";
    # The system unit's trap applies here too: denying @resources kills
    # llama.cpp with SIGSYS while the model loads.
    assert lib.assertMsg (
      isLinux -> !(lib.elem "~@resources" unit.Service.SystemCallFilter)
    ) "the user unit's SystemCallFilter denies @resources";
    assert lib.assertMsg (
      isDarwin -> agent.enable && agent.config.KeepAlive.Crashed && hm.systemd.user.services == { }
    ) "the launchd agent is not the one defined on macOS";
    runCommand "nix-llama-check-home-manager-eval" { nativeBuildInputs = [ pkgs.jq ]; } ''
      cp ${hm.xdg.configFile."nix-llama/config.json".source} config.json
      cp ${launcher} launcher.sh

      jq -e '.model == "/home/alice/models/example.gguf"' config.json > /dev/null
      jq -e '.rag.topK == 8' config.json > /dev/null
      # /var/lib is not writable by a user service, so the default must move.
      jq -e '.stateDir == "/home/alice/.local/state/nix-llama"' config.json > /dev/null

      grep -qF "mkdir -p -m 0700 /home/alice/.local/state/nix-llama" launcher.sh \
        || { echo "the launcher no longer creates stateDir" >&2; exit 1; }
      grep -qE '^export .*NIX_LLAMA_CONFIG' launcher.sh \
        || { echo "the launcher no longer exports NIX_LLAMA_CONFIG" >&2; exit 1; }
      grep -qF '${stubPackage}/bin/nix-llama serve' launcher.sh \
        || { echo "the launcher no longer runs nix-llama serve" >&2; exit 1; }

      echo "home-manager module evaluation checks passed"
      touch $out
    '';
}
// lib.optionalAttrs isLinux {
  # Full-system test: the module, the systemd unit and its hardening, the
  # service account, and a request from a second machine on the network.
  nixos-module = pkgs.testers.runNixOSTest (import ../tests/nixos-service.nix { inherit self; });

  # Cheap module evaluation for the non-default paths the VM test does not
  # cover: GPU acceleration, firewall, API keys, overrides.
  nixos-module-eval =
    let
      evaluated = nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.default
          (
            { lib, ... }:
            {
              # Enough of a system to evaluate; nothing is ever built from it.
              boot.loader.grub.enable = false;
              fileSystems."/" = {
                device = "/dev/null";
                fsType = "ext4";
              };
              system.stateVersion = "25.05";
              nixpkgs.hostPlatform = system;

              services.nix-llama = {
                enable = true;
                acceleration = "vulkan";
                host = "0.0.0.0";
                port = 9090;
                openFirewall = true;
                apiKeyFile = "/run/secrets/key";
                model = "/var/lib/models/example.gguf";
                tuning = {
                  contextSize = 16384;
                  threads = 6;
                  kvCacheType = "q8_0";
                  speculative = "off";
                  loadMode = "mmap";
                };
                diskCache.enable = false;
                rag = {
                  topK = 8;
                  injectInto = "user";
                };
                settings."--temp" = 0;
                extraFlags = [ "--props" ];
                environment.GGML_VK_VISIBLE_DEVICES = "0";
              };
              boot.loader.systemd-boot.enable = lib.mkForce false;
            }
          )
        ];
      };
      unit = evaluated.config.systemd.units."nix-llama.service".text;
      configJson = evaluated.config.systemd.services.nix-llama.environment.NIX_LLAMA_CONFIG;
    in
    runCommand "nix-llama-check-module-eval" { nativeBuildInputs = [ pkgs.jq ]; } ''
      cat > unit.txt <<'UNIT'
      ${unit}
      UNIT
      cp ${configJson} config.json

      echo "== generated config"
      cat config.json | jq .

      jq -e '.host == "0.0.0.0"' config.json > /dev/null
      jq -e '.port == 9090' config.json > /dev/null
      jq -e '.tuning.contextSize == 16384' config.json > /dev/null
      jq -e '.tuning.kvCacheType == "q8_0"' config.json > /dev/null
      jq -e '.rag.topK == 8 and .rag.injectInto == "user"' config.json > /dev/null
      jq -e '.diskCache.enable == false' config.json > /dev/null
      jq -e '.settings["--temp"] == 0' config.json > /dev/null
      jq -e '.extraFlags == ["--props"]' config.json > /dev/null
      jq -e '.apiKeyFile == "/run/secrets/key"' config.json > /dev/null
      # A path to a .gguf must become `model`, not `modelDir`.
      jq -e '.model == "/var/lib/models/example.gguf"' config.json > /dev/null
      jq -e '.modelDir == null' config.json > /dev/null

      echo "== unit hardening"
      for required in \
        'ProtectSystem=strict' \
        'ProtectHome=true' \
        'NoNewPrivileges=true' \
        'RestrictNamespaces=true' \
        'LockPersonality=true' \
        'SystemCallArchitectures=native' \
        'ProcSubset=pid' \
        'UMask=0077' \
        'User=nix-llama' \
        'Group=nix-llama' \
        'StateDirectory=nix-llama' \
        'GGML_VK_VISIBLE_DEVICES=0'; do
        grep -qF "$required" unit.txt \
          || { echo "the unit lost: $required" >&2; exit 1; }
      done

      # Vulkan needs the render nodes, which means PrivateDevices cannot be on
      # and the device allowlist has to be present instead.
      grep -qF 'PrivateDevices=false' unit.txt \
        || { echo "a GPU backend needs PrivateDevices=false" >&2; exit 1; }
      grep -qF 'DeviceAllow=/dev/dri rw' unit.txt \
        || { echo "a GPU backend needs /dev/dri allowed" >&2; exit 1; }
      grep -qF 'SupplementaryGroups=render' unit.txt \
        || { echo "a GPU backend needs the render group" >&2; exit 1; }

      # This configuration pins loadMode to "mmap", so the mlock capability and
      # rlimit must *not* be granted. Capabilities follow need, not habit --
      # the VM test covers the default path where mlock is used.
      if grep -qF 'LimitMEMLOCK' unit.txt; then
        echo "LimitMEMLOCK was granted although loadMode is mmap" >&2
        exit 1
      fi
      granted=$(grep '^CapabilityBoundingSet=' unit.txt | cut -d= -f2- || true)
      test -z "$granted" \
        || { echo "unexpected capabilities granted: $granted" >&2; exit 1; }

      echo "module evaluation checks passed"
      touch $out
    '';
}
