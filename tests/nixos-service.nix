# NixOS integration test for the nix-llama service.
#
# Boots two machines: one running the service as the module configures it, and
# one client on the same network. What it proves is that the module, not just
# the CLI, produces a working server -- the unit starts under its hardening,
# the service account can reach its state directory, API-key auth is enforced
# through the retrieval proxy, and the proxy injects documentation.
# `runNixOSTest` is called on an overlay-applied nixpkgs, so the nodes already
# see `pkgs.nix-llama-*`. That is why the bare module is imported here rather
# than `nixosModules.default`: the latter sets `nixpkgs.overlays`, which the
# test framework pins read-only.
{ self }:

_:

{
  name = "nix-llama-service";

  meta.maintainers = [ ];

  nodes = {
    server =
      { pkgs, ... }:
      {
        imports = [ self.nixosModules.nix-llama ];

        # 1 MB of nonsense weights with a valid GGUF header: enough to exercise
        # every code path without putting a 4.7 GB download in CI.
        services.nix-llama = {
          enable = true;
          model = pkgs.nix-llama-model-test;
          host = "0.0.0.0";
          port = 8080;
          openFirewall = true;
          apiKeyFile = "/etc/nix-llama/api-key";
          acceleration = "none";
          tuning = {
            contextSize = 512;
            threads = 2;
            webui = false;
          };
        };

        environment.etc."nix-llama/api-key" = {
          text = "test-key-not-a-secret\n";
          mode = "0444";
        };

        environment.systemPackages = [
          pkgs.curl
          pkgs.jq
        ];

        virtualisation = {
          memorySize = 2048;
          cores = 2;
          diskSize = 4096;
        };
      };

    client =
      { pkgs, ... }:
      {
        environment.systemPackages = [
          pkgs.curl
          pkgs.jq
        ];
      };
  };

  testScript = ''
    start_all()

    with subtest("the unit starts and stays up"):
        server.wait_for_unit("nix-llama.service")
        # The startup explanation and any failure both land here, so dumping it
        # unconditionally makes a failing run self-diagnosing.
        server.succeed("journalctl -u nix-llama.service --no-pager | tail -n 60 >&2")
        server.succeed("systemctl show nix-llama.service -p Result -p ExecMainStatus >&2")
        server.wait_for_open_port(8080)
        server.succeed("systemctl is-active nix-llama.service")

    with subtest("the probe and planner agree with the module"):
        server.succeed("nix-llama probe >&2")
        plan = server.succeed("nix-llama plan --format json")
        assert '"--ctx-size"' in plan, plan
        assert '"512"' in plan, "the module's contextSize did not reach llama-server"

    with subtest("it runs unprivileged under its own account"):
        # Asked of systemd rather than parsed out of ps: the service's main
        # process is nix-llama, llama-server is its child, and process names
        # are not a stable interface.
        main_pid = server.succeed(
            "systemctl show -p MainPID --value nix-llama.service"
        ).strip()
        assert main_pid.isdigit() and main_pid != "0", f"no main pid: {main_pid!r}"
        owner = server.succeed(f"stat -c %U /proc/{main_pid}").strip()
        assert owner == "nix-llama", f"the service runs as {owner!r}"

        # And the model server it spawned is owned by the same account.
        server.succeed("pgrep -u nix-llama -f llama-server >/dev/null")

        # Its state directory exists and is not world-readable.
        server.succeed("test -d /var/lib/nix-llama")
        mode = server.succeed("stat -c %a /var/lib/nix-llama").strip()
        assert mode == "750", f"state directory mode is {mode}"

    with subtest("the API key is enforced through the proxy"):
        status = server.succeed(
            "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/v1/models"
        ).strip()
        assert status == "401", f"an unauthenticated request got {status}, not 401"

        status = server.succeed(
            "curl -s -o /dev/null -w '%{http_code}' "
            "-H 'Authorization: Bearer test-key-not-a-secret' "
            "http://localhost:8080/v1/models"
        ).strip()
        assert status == "200", f"an authenticated request got {status}, not 200"

    with subtest("the configuration is discoverable from a shell"):
        # An operator debugging the service types `nix-llama plan`, not
        # `NIX_LLAMA_CONFIG=... nix-llama plan`.
        server.succeed("test -f /etc/nix-llama/config.json")
        server.succeed("nix-llama doctor >&2")

    with subtest("an authenticated client on the network can reach it"):
        client.wait_for_unit("multi-user.target")
        client.succeed("curl -sf http://server:8080/health | jq -e '.status == \"ok\"'")
        client.succeed(
            "curl -sf -H 'Authorization: Bearer test-key-not-a-secret' "
            "http://server:8080/v1/models | jq -e '.models | length >= 1'"
        )

    with subtest("retrieval is live"):
        client.succeed(
            "curl -sf 'http://server:8080/v1/rag/stats' | jq -e '.chunks > 0'"
        )
        client.succeed(
            "curl -sf 'http://server:8080/v1/rag/search?q=nixos%20service%20enable' "
            "| jq -e '.results | length >= 1'"
        )

    with subtest("a chat completion is answered and augmented"):
        client.succeed(
            "curl -sf -X POST http://server:8080/v1/chat/completions "
            "-H 'Content-Type: application/json' "
            "-H 'Authorization: Bearer test-key-not-a-secret' "
            "-d '{\"messages\":[{\"role\":\"user\",\"content\":"
            "\"how do I enable the openssh service in NixOS?\"}],\"max_tokens\":4}' "
            "| jq -e '.choices[0].message.content != null'"
        )
        server.succeed(
            "journalctl -u nix-llama.service | grep -q 'augmented request with'"
        )

    with subtest("it survives a restart"):
        server.succeed("systemctl restart nix-llama.service")
        server.wait_for_unit("nix-llama.service")
        server.wait_until_succeeds("curl -sf http://localhost:8080/health", timeout=120)

    with subtest("the service stops cleanly"):
        server.succeed("systemctl stop nix-llama.service")
        server.wait_until_fails("curl -sf --max-time 2 http://localhost:8080/health")
  '';
}
