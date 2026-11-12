# Option declarations shared by the NixOS and nix-darwin modules.
#
# Both platforms expose exactly the same surface; only the service manager
# differs. Everything here is turned into the JSON configuration file that
# `nix-llama serve` reads, so the module and the CLI cannot disagree about what
# a setting means.
{
  lib,
  pkgs,
  ...
}:

let
  inherit (lib)
    literalExpression
    mkEnableOption
    mkOption
    types
    ;

  tuningType = types.submodule {
    options = {
      contextSize = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = ''
          Context window in tokens. Null derives it from the model's trained
          context and the memory the probe finds, which is almost always the
          right answer -- set this only to deliberately trade context for
          headroom, or to push past what the probe considers safe.
        '';
      };
      maxContextSize = mkOption {
        type = types.ints.positive;
        default = 32768;
        description = ''
          Ceiling on the derived context size. The default matches the trained
          context of the shipped model; raising it past a model's training
          length relies on RoPE scaling and degrades quality.
        '';
      };
      minContextSize = mkOption {
        type = types.ints.positive;
        default = 4096;
        description = "Floor on the derived context size.";
      };
      gpuLayers = mkOption {
        type = types.nullOr types.int;
        default = null;
        description = ''
          Layers to offload to the accelerator. Null leaves it to llama.cpp's
          own `--fit`, which measures free device memory at startup -- better
          information than a one-shot probe has.
        '';
      };
      threads = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = ''
          Generation threads. Null uses the probe's count of usable physical
          cores, excluding hyper-threads, efficiency cores and anything a
          cgroup quota has taken away.
        '';
      };
      threadsBatch = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = "Threads for prompt processing. Null matches `threads`.";
      };
      batchSize = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = "Logical batch size (`--batch-size`).";
      };
      ubatchSize = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = "Physical batch size (`--ubatch-size`).";
      };
      parallel = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = ''
          Number of server slots. Null lets llama.cpp choose, which also
          enables its unified KV cache so a single long request can use the
          whole context.
        '';
      };
      kvCacheType = mkOption {
        type = types.enum [
          "auto"
          "f16"
          "q8_0"
          "q5_1"
          "q5_0"
          "q4_1"
          "q4_0"
        ];
        default = "auto";
        description = ''
          KV cache quantization. "auto" uses f16 when the target context fits
          and drops to q8_0 when quantizing is what makes the context reachable.
        '';
      };
      promptCacheMiB = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = ''
          Host RAM for the prompt cache (`--cache-ram`). Null derives it from
          available memory.
        '';
      };
      speculative = mkOption {
        type = types.str;
        default = "auto";
        example = "ngram-mod";
        description = ''
          Speculative decoding strategy passed to `--spec-type`, or "off".
          "auto" enables prompt-ngram drafting on an accelerator -- it costs no
          extra memory and pays off on code editing, where the model echoes
          long verbatim spans -- and leaves it off on CPU, where verifying
          drafts competes with generation for the same cores.
        '';
      };
      contextShift = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Keep answering when a session outgrows the context window by dropping
          the oldest tokens, instead of failing the request. The trade is real:
          the model silently loses the start of the conversation. Turn it off if
          you would rather see the error.
        '';
      };
      loadMode = mkOption {
        type = types.nullOr (
          types.enum [
            "auto"
            "none"
            "mmap"
            "mlock"
            "mmap+mlock"
            "dio"
          ]
        );
        default = null;
        description = ''
          How to load the weights (`--load-mode`). Null picks `mmap+mlock` when
          the model fits in half of memory -- pinning removes page-fault stalls
          mid-generation -- and plain `mmap` otherwise.
        '';
      };
      flashAttention = mkOption {
        type = types.enum [
          "auto"
          "on"
          "off"
        ];
        default = "auto";
        description = "Flash attention (`--flash-attn`).";
      };
      corsOrigins = mkOption {
        type = types.nullOr types.str;
        default = "localhost";
        example = "https://chat.example.com";
        description = ''
          Allowed CORS origins. Defaults to localhost rather than llama.cpp's
          `*`: a wildcard with credentials reflected back lets any page in the
          user's browser talk to the model.
        '';
      };
      metrics = mkOption {
        type = types.bool;
        default = true;
        description = "Expose the Prometheus `/metrics` endpoint.";
      };
      webui = mkOption {
        type = types.bool;
        default = true;
        description = "Serve llama.cpp's built-in web UI.";
      };
    };
  };

  diskCacheType = types.submodule {
    options = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Let llama-server spill slot KV caches to local storage, so a long
          conversation can be restored instead of re-evaluated. Only used when
          the probe finds flash storage with enough free space: doing this on a
          spinning disk is slower than recomputing.
        '';
      };
      path = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Where to keep spilled caches. Null uses `<stateDir>/slots`.";
      };
      minFreeGiB = mkOption {
        type = types.ints.positive;
        default = 8;
        description = "Refuse to use disk caching below this much free space.";
      };
    };
  };

  ragType = types.submodule {
    options = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Front llama-server with the retrieval proxy. The proxy owns the public
          port, injects documentation excerpts into chat requests whose answer
          plausibly lives in the corpus, and relays everything else untouched.
          Disable it to have llama-server bind the public port directly.
        '';
      };
      index = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          Retrieval index. Null uses the one shipped inside the model package.
        '';
      };
      topK = mkOption {
        type = types.ints.positive;
        default = 4;
        description = "Excerpts to retrieve per request.";
      };
      maxChars = mkOption {
        type = types.ints.positive;
        default = 6000;
        description = "Character budget for injected context.";
      };
      minScore = mkOption {
        type = types.float;
        default = 1.0;
        description = ''
          BM25 score below which a match is treated as noise and dropped.
          Lower it to retrieve more eagerly, raise it if irrelevant excerpts
          are crowding out the user's own context.
        '';
      };
      injectInto = mkOption {
        type = types.enum [
          "system"
          "user"
        ];
        default = "system";
        description = "Whether excerpts are appended to the system or the user message.";
      };
      upstreamPort = mkOption {
        type = types.nullOr types.port;
        default = null;
        description = ''
          Loopback port for llama-server when the proxy is in front. Null picks
          `port + 1`, falling back to any free port.
        '';
      };
    };
  };
in
{
  options.services.nix-llama = {
    enable = mkEnableOption "the nix-llama model server";

    package = mkOption {
      # Nullable: the real default depends on `acceleration`, which each
      # platform module resolves differently, so it cannot live here.
      type = types.nullOr types.package;
      default = null;
      defaultText = literalExpression "the nix-llama build matching services.nix-llama.acceleration";
      description = ''
        The nix-llama package to run. The default follows `acceleration`, so
        set that rather than this unless you have built your own variant.
      '';
    };

    acceleration = mkOption {
      type = types.enum [
        "auto"
        "none"
        "cuda"
        "rocm"
        "vulkan"
        "metal"
      ];
      default = "auto";
      description = ''
        Which llama.cpp backend to build and run against.

        "auto" infers it from the rest of the configuration: `cudaSupport` or
        an NVIDIA driver gives CUDA, `rocmSupport` gives ROCm, a graphics-
        enabled host gives Vulkan, Darwin gives Metal, and anything else runs
        on the CPU. Inference happens at evaluation time, so it reads your
        configuration rather than your hardware -- the runtime probe then
        checks the binary really can see a device and warns loudly if it
        cannot.
      '';
    };

    model = mkOption {
      type = types.either types.path types.package;
      default = pkgs.nix-llama-model;
      defaultText = literalExpression "pkgs.nix-llama-model";
      description = ''
        The model to serve: a model directory (as produced by `nix build
        .#model`, containing weights, retrieval index and system prompt), or a
        path straight to a `.gguf` file.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = ''
        Address to listen on. The default is loopback-only; binding a wider
        address without also setting `apiKeyFile` publishes an unauthenticated
        model server, and the module will warn about it.
      '';
    };

    port = mkOption {
      type = types.port;
      default = 8080;
      description = "Port to listen on.";
    };

    apiKeyFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = "/run/secrets/nix-llama-api-key";
      description = ''
        File containing API keys, one per line. Read by the service at startup,
        so it can come from a secret manager and never enters the Nix store.
      '';
    };

    systemPrompt = mkOption {
      type = types.nullOr (types.either types.path types.str);
      default = null;
      description = ''
        System prompt injected when a client sends none. Null uses the one
        shipped with the model package.
      '';
    };

    stateDir = mkOption {
      type = types.path;
      default = "/var/lib/nix-llama";
      description = "Writable directory for caches and spilled KV slots.";
    };

    user = mkOption {
      type = types.str;
      default = "nix-llama";
      description = "User to run the service as.";
    };

    group = mkOption {
      type = types.str;
      default = "nix-llama";
      description = "Group to run the service as.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = ''
        Open `port` in the firewall. Refuses to do so without `apiKeyFile`:
        an open port on an unauthenticated model server is a remote code
        execution surface for anyone who can reach it.
      '';
    };

    tuning = mkOption {
      type = tuningType;
      default = { };
      description = "Performance knobs. Every one defaults to what the probe infers.";
    };

    diskCache = mkOption {
      type = diskCacheType;
      default = { };
      description = "Disk-backed KV cache settings.";
    };

    rag = mkOption {
      type = ragType;
      default = { };
      description = "Documentation retrieval settings.";
    };

    tls = mkOption {
      type = types.submodule {
        options = {
          certFile = mkOption {
            type = types.nullOr types.path;
            default = null;
            description = "PEM certificate for serving HTTPS directly.";
          };
          keyFile = mkOption {
            type = types.nullOr types.path;
            default = null;
            description = "PEM private key for serving HTTPS directly.";
          };
        };
      };
      default = { };
      description = ''
        Serve TLS from llama-server itself. Usually better handled by a reverse
        proxy, but useful when there isn't one.
      '';
    };

    settings = mkOption {
      type = types.attrsOf (
        types.nullOr (
          types.oneOf [
            types.bool
            types.int
            types.str
          ]
        )
      );
      default = { };
      example = literalExpression ''
        {
          "--temp" = 0.2;
          "--top-p" = 0.9;
          "--reasoning-budget" = 0;
        }
      '';
      description = ''
        Raw llama-server flags, as a flag-to-value mapping. These override
        anything the probe derives for the same flag. Use `true` for flags that
        take no value.
      '';
    };

    extraFlags = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [
        "--log-timestamps"
        "--props"
      ];
      description = ''
        Raw arguments appended last, after everything else. The final word on
        the command line, and never inspected -- the escape hatch for anything
        the module does not model.
      '';
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      example = literalExpression ''{ GGML_VK_VISIBLE_DEVICES = "0"; }'';
      description = "Extra environment variables for the service.";
    };
  };
}
