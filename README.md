# nix-llama

A turnkey, self-hosted LLM server. It probes the machine it is running on, reads the
model's GGUF header, and starts `llama-server` with settings sized for that specific
combination of accelerator, memory, CPU topology and storage — then fronts it with a
documentation retrieval layer so a small model can cite current upstream docs instead
of guessing.

The goal is a model server that is fast, that does not choke on long contexts, and
that runs on hardware you already own.

```console
$ nix run github:heywoodlh/nix-llama
INFO nix-llama: backend cuda with a 23548 MiB device memory budget; model occupies 4466 MiB
INFO nix-llama: context 32768 tokens (model trained for 131072)
INFO nix-llama: 12 threads (of 24 logical / 12 physical cores)
INFO nix-llama: slot KV cache may spill to /var/lib/nix-llama/slots/ (flash storage, 412 GiB free)
INFO nix-llama: speculative decoding: ngram-mod
INFO nix-llama.rag: RAG proxy listening on 127.0.0.1:8080 -> 127.0.0.1:8081 (39587 chunks indexed)
```

That one command fetches the default model bundle (4.7 GB the first time), probes the
machine, and serves an OpenAI-compatible endpoint on `127.0.0.1:8080`.

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [NixOS](#nixos)
- [nix-darwin](#nix-darwin)
- [Home Manager](#home-manager)
- [Docker](#docker)
- [Kubernetes](#kubernetes)
- [Using it from a coding harness](#using-it-from-a-coding-harness)
  - [OpenCode](#opencode)
  - [Pi](#pi)
- [How the defaults are chosen](#how-the-defaults-are-chosen)
- [Documentation retrieval](#documentation-retrieval)
- [The model](#the-model)
- [Configuration reference](#configuration-reference)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## What you get

| | |
|---|---|
| `nix-llama` | Probes the host, plans a `llama-server` invocation, runs it, and fronts it with the retrieval proxy. |
| `nix-llama-rag` | Builds and queries the documentation index; also an MCP server and an augmenting proxy. |
| `nix-llama-probe` | Prints what nix-llama detected, as text or JSON. |
| NixOS / nix-darwin modules | The same option surface on both, with systemd hardening and a service account. |
| Home Manager module | The same options again, as a systemd user unit or a launchd agent, for machines whose system configuration you do not control. |
| Container images | Built from this flake, so the `llama-server` binary is the same store path the service runs. |
| `#model` | A model *bundle*: quantized weights, retrieval index, system prompt, model card, attribution. |

Hardware acceleration is detected for NVIDIA (CUDA), AMD (ROCm and Vulkan), Intel
(Vulkan) and Apple silicon (Metal), including inside containers — and cross-checked
against the backends the `llama.cpp` binary was actually built with, so a GPU that
cannot be used produces a warning instead of a silent 30× slowdown.

Supported systems: `x86_64-linux`, `aarch64-linux`, `aarch64-darwin`. Intel Macs are
not a flake target because nixpkgs 26.11 dropped `x86_64-darwin`; the container image
still runs there.

## Quick start

```bash
# Serve the default model. Fetches it on first run (~4.7 GB).
nix run github:heywoodlh/nix-llama

# Ask it something.
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"How do I enable sshd in NixOS?"}]}' \
  | jq -r '.choices[0].message.content'
```

To keep the model as a separate artifact — which is what the service modules and the
container do — build it on its own and point at it:

```bash
nix build github:heywoodlh/nix-llama#model
nix run github:heywoodlh/nix-llama -- --model-dir ./result

# Or serve a GGUF you already have; nothing here requires the bundled model.
nix run github:heywoodlh/nix-llama -- --model ~/models/some-model.gguf
```

Before starting anything, it is worth seeing what nix-llama thinks of your machine:

```bash
# What did it detect?
nix run github:heywoodlh/nix-llama#probe

# What would it run, and why?
nix run github:heywoodlh/nix-llama -- plan

# Is everything in place?
nix run github:heywoodlh/nix-llama -- doctor
```

`plan` is the one to reach for when performance is not what you expect: it prints the
exact command line and a line of reasoning for every choice it made.

## NixOS

```nix
{
  inputs.nix-llama.url = "github:heywoodlh/nix-llama";

  outputs = { nixpkgs, nix-llama, ... }: {
    nixosConfigurations.workstation = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        nix-llama.nixosModules.default
        {
          services.nix-llama.enable = true;
        }
      ];
    };
  };
}
```

That is the whole minimal configuration. It serves the default model on
`127.0.0.1:8080`, under its own system account, with systemd hardening, and picks a
`llama.cpp` backend from the rest of your configuration.

A more opinionated example:

```nix
services.nix-llama = {
  enable = true;

  # "auto" infers the backend from your configuration (cudaSupport, an NVIDIA
  # driver, rocmSupport, hardware.graphics). Set it explicitly when that guess
  # would be wrong -- it reads your configuration, not your PCI bus.
  acceleration = "cuda";

  # Reachable from the LAN. The module *asserts* that this is accompanied by
  # authentication; see the Security section.
  host = "0.0.0.0";
  apiKeyFile = config.age.secrets.nix-llama-key.path;
  openFirewall = true;

  # Everything below is optional. Each of these defaults to what the probe
  # infers, which is usually the right answer.
  tuning = {
    contextSize = 65536;      # push past the 32k default ceiling
    kvCacheType = "q8_0";     # halve the KV cache to afford it
    parallel = 2;             # two concurrent conversations
  };

  # Raw llama-server flags, for anything the module does not model.
  settings."--temp" = 0.2;
  extraFlags = [ "--props" ];
};
```

If your configuration already applies the overlay (or pins its own nixpkgs), import
`nix-llama.nixosModules.nix-llama` instead of `default` — the only difference is that
`default` also sets `nixpkgs.overlays`.

## nix-darwin

```nix
{
  inputs.nix-llama.url = "github:heywoodlh/nix-llama";

  outputs = { nix-darwin, nix-llama, ... }: {
    darwinConfigurations.laptop = nix-darwin.lib.darwinSystem {
      modules = [
        nix-llama.darwinModules.default
        {
          services.nix-llama.enable = true;
          # nix-darwin cannot allocate a uid automatically; pick a free one in
          # the system range if 541 is taken.
          services.nix-llama.uid = 541;
          services.nix-llama.gid = 541;
        }
      ];
    };
  };
}
```

The options are identical to the NixOS module. Differences on macOS: the service runs
under launchd, logs to `/var/log/nix-llama.log`, and `acceleration = "auto"` resolves
to Metal, where the memory budget is 75% of unified memory rather than a separate VRAM
pool.

## Home Manager

Runs the server as you rather than as a service account: a systemd user unit on
Linux, a launchd agent on macOS. Useful on a work Mac, a non-NixOS distribution, or
anywhere the system configuration is not yours to change.

```nix
{
  inputs.nix-llama.url = "github:heywoodlh/nix-llama";

  outputs = { home-manager, nixpkgs, nix-llama, ... }: {
    homeConfigurations.alice = home-manager.lib.homeManagerConfiguration {
      pkgs = nixpkgs.legacyPackages.aarch64-darwin;
      modules = [
        nix-llama.homeManagerModules.default
        { services.nix-llama.enable = true; }
      ];
    };
  };
}
```

The options are the same as the system modules, with these differences:

- `stateDir` defaults to `$XDG_STATE_HOME/nix-llama` (`~/.local/state/nix-llama`).
- `user`, `group` and `uid`/`gid` do not apply. `openFirewall` is rejected: the
  firewall belongs to the system configuration.
- `acceleration = "auto"` resolves to Metal on macOS. On Linux it reads the NixOS
  configuration when Home Manager runs as a NixOS module (`osConfig`); standalone
  there is nothing to read, so it picks the CPU and warns — set it explicitly.
- The Linux unit keeps the seccomp half of the system unit's hardening (syscall
  filter, address families, `NoNewPrivileges`); the filesystem sandbox needs
  privileges a user manager may not have. No mlock limit is raised, because a user
  unit cannot exceed its manager's hard limit; if pinning fails, llama.cpp warns and
  runs unpinned, and `tuning.loadMode = "mmap"` silences it.
- Logs go to the journal (`journalctl --user -u nix-llama`) on Linux and to
  `~/Library/Logs/nix-llama.log` on macOS. The rendered configuration is installed at
  `~/.config/nix-llama/config.json`, where `nix-llama plan` and `doctor` find it.

`homeManagerModules.default` supplies its own packages without touching
`nixpkgs.overlays`, so it works with `home-manager.useGlobalPkgs`.
`homeManagerModules.nix-llama` expects the overlay to be applied already.

## Docker

The image carries no weights — a 5 GB model does not belong in a container layer by
default — so mount a model bundle:

```bash
nix build github:heywoodlh/nix-llama#model -o ./model

docker run -d --name nix-llama \
  -p 127.0.0.1:8080:8080 \
  -v "$(readlink -f ./model)":/model:ro \
  -v nix-llama-state:/var/lib/nix-llama \
  --memory 10g \
  ghcr.io/heywoodlh/nix-llama:latest
```

Because the bundle is a Nix store path full of symlinks, also mount `/nix/store`
read-only (`-v /nix/store:/nix/store:ro`) — or use the image that has the model baked
in and skip the volume entirely:

```bash
nix build github:heywoodlh/nix-llama#container-model
./result | docker load
docker run -d -p 127.0.0.1:8080:8080 nix-llama-model:0.1.0
```

`--memory` is not merely advisory: nix-llama reads the container's cgroup limit and
sizes the context window and thread count to fit it.

### Compose

```yaml
# docker-compose.yml
services:
  nix-llama:
    image: ghcr.io/heywoodlh/nix-llama:latest
    restart: unless-stopped
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./model:/model:ro
      - /nix/store:/nix/store:ro
      - nix-llama-state:/var/lib/nix-llama
    environment:
      NIX_LLAMA_HOST: "0.0.0.0"
      NIX_LLAMA_MODEL_DIR: "/model"
    deploy:
      resources:
        limits:
          memory: 10G
    healthcheck:
      test: ["CMD", "/bin/nix-llama", "doctor"]
      interval: 30s
      start_period: 180s
    read_only: true
    tmpfs: [/tmp]
    security_opt: [no-new-privileges:true]

volumes:
  nix-llama-state:
```

```bash
nix build github:heywoodlh/nix-llama#model -o ./model
docker compose up -d
```

The fuller version in [`examples/docker-compose.yml`](examples/docker-compose.yml) adds
comments for the tuning knobs and a commented-out GPU passthrough block.

There are two ways to build the image, and they produce the same binaries:

```bash
nix build .#container && ./result | docker load   # no Docker needed to build
docker build -f docker/Dockerfile -t nix-llama .  # Nix runs inside stage 1
```

The Dockerfile's first stage builds `.#container-closure` with Nix and the second
stage is `scratch` plus that closure, so the `llama-server` inside the image is the
*same store path* the NixOS service runs. CI builds both and fails if they disagree.

## Kubernetes

A minimal, CPU-only Deployment and Service:
[`examples/kubernetes.yaml`](examples/kubernetes.yaml).

```bash
nix build github:heywoodlh/nix-llama#container-model
./result | docker load
# push to your registry, update the image reference, then:
kubectl apply -f examples/kubernetes.yaml
```

It uses the model-baked image so there is no volume to provision. `resources.limits`
is the main tuning knob — the pod's cgroup is what nix-llama reads to size itself.
Give the 7B model at least 8Gi, and keep the generous `startupProbe`: loading
multi-gigabyte weights on CPU is slow, and a tight probe produces a restart loop that
never converges.

Accelerators are deliberately out of scope here. GPUs in Kubernetes mean device
plugins, node labels and a runtime class, all of which depend on your cluster rather
than on nix-llama.

## Using it from a coding harness

nix-llama speaks the OpenAI chat-completions API, so anything that can point at a
custom base URL works. Two harnesses are covered here end to end.

Both sections assume the server is reachable at `http://127.0.0.1:8080/v1`. If it is
on another host, use `http://<host>:8080/v1` and set an API key (see
[Security](#security)).

### OpenCode

Add nix-llama as a provider in `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "nix-llama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "nix-llama",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "nix-llama": {
          "name": "nix-llama",
          "context_length": 32768,
          "max_tokens": 8192
        }
      }
    }
  },
  "model": "nix-llama/nix-llama"
}
```

Set `context_length` to the context nix-llama actually chose — `nix-llama plan` prints
it — rather than guessing high. OpenCode trims history against this number, and a
value larger than the server's context turns into rejected requests at exactly the
moment a long session gets interesting.

If the server has an API key, pass it through the provider options:

```json
"options": {
  "baseURL": "http://llm.example.internal:8080/v1",
  "apiKey": "{env:NIX_LLAMA_API_KEY}"
}
```

You can also give OpenCode the documentation search tool directly, so the model can
look things up on its own:

```json
{
  "mcp": {
    "nix-llama-docs": {
      "type": "local",
      "command": ["nix-llama-rag", "mcp"],
      "enabled": true,
      "environment": {
        "NIX_LLAMA_RAG_INDEX": "/var/lib/nix-llama/rag/index.json.gz"
      }
    }
  }
}
```

`nix-llama-rag mcp` exposes `search_docs` and `list_doc_sources`. Point
`NIX_LLAMA_RAG_INDEX` at the `rag/index.json.gz` inside your model bundle. This is
complementary to the proxy: the proxy injects documentation automatically, the MCP
tool lets the model ask for it when *it* notices a gap.

### Pi

Add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "nix-llama": {
      "api": "openai-completions",
      "apiKey": "local",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "compat": {
        "supportsDeveloperRole": false
      },
      "models": [
        {
          "id": "nix-llama",
          "name": "nix-llama",
          "contextWindow": 32768,
          "maxTokens": 8192,
          "input": ["text"],
          "reasoning": false,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

Then make it the default in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "nix-llama",
  "defaultModel": "nix-llama",
  "defaultTools": ["read", "write", "edit", "bash", "grep", "find", "ls"]
}
```

Notes that matter in practice:

- `"supportsDeveloperRole": false` is correct here. llama.cpp's chat templates expect
  `system`, not OpenAI's newer `developer` role.
- `apiKey` must be non-empty even when the server does not check it; `"local"` is the
  conventional placeholder. With `apiKeyFile` configured on the server, put the real
  key here instead.
- `contextWindow` should match what `nix-llama plan` reports, for the same reason as
  OpenCode.
- `reasoning` stays `false`: the shipped model is not a reasoning model, and claiming
  otherwise makes Pi wait for thinking blocks that never arrive.

For the documentation tool, Pi reaches MCP servers through `pi-mcp-adapter` (already
in `packages` in a default Pi install). Register `nix-llama-rag mcp` as a local MCP
server through that adapter and the model gets the same `search_docs` tool described
above.

## How the defaults are chosen

`nix-llama probe` gathers facts; `nix-llama plan` turns them into flags. The split is
deliberate — the probe does the I/O, the planner is a pure function, and you can
inspect either one.

What the probe looks at:

| Fact | How | Why it matters |
|---|---|---|
| Accelerators | `nvidia-smi`, `/dev/nvidia*`, `/dev/kfd`, `/sys/class/drm/*/device/vendor`, PCI vendor IDs, Darwin platform | Picks the backend |
| Usable backends | `llama-server --list-devices` | Hardware you can see is not hardware the *binary* can use |
| Memory | `/proc/meminfo`, `sysctl hw.memsize`, **cgroup v1/v2 limits** | Context size, KV cache type, prompt cache |
| CPU topology | `/proc/cpuinfo`, `thread_siblings_list`, `hw.perflevel0`, affinity mask, cgroup CPU quota | Thread count — physical cores only, no hyper-threads or efficiency cores |
| Storage | `/sys/block/*/queue/rotational` for the device backing the state dir | Whether to spill KV cache to disk |
| Container | `/.dockerenv`, `/run/.containerenv`, cgroup paths, `$container` | Device detection changes inside a container |
| Model geometry | the GGUF header: layers, KV heads, head dim, trained context | Exact KV-cache arithmetic |

Decisions worth knowing about:

- **Context size** is computed from the model's trained context, the KV bytes per
  token implied by its geometry, and the memory budget — then capped at 32768 by
  default. If `f16` KV does not fit, the cache is quantized to `q8_0` (about half the
  size) rather than shrinking the context, because for coding work a long context is
  worth more than a perfectly precise cache.
- **GPU offload is left unset.** `llama.cpp --fit` measures free device memory at
  startup, which is better information than a one-shot probe has. Set
  `tuning.gpuLayers` to override.
- **Threads** are physical cores, minus whatever a cgroup quota has taken away. Hyper-
  threads and Apple efficiency cores both cost more than they add.
- **Prompt cache reuse** (`--cache-reuse 256`, `--cache-ram`) is what makes an agent
  harness feel fast: every turn resends the whole conversation, and reuse lets
  llama.cpp skip re-evaluating the unchanged prefix.
- **Disk-backed KV cache** (`--slot-save-path`) is enabled only on flash storage with
  enough free space. Doing this on a spinning disk is slower than recomputing, and
  "unknown" counts as not-flash.
- **Speculative decoding** uses `--spec-type ngram-mod` on an accelerator: it drafts
  from the prompt itself, costs no extra memory, and code editing — where the model
  echoes long verbatim spans — is exactly the workload it helps. It stays off on CPU,
  where verifying drafts competes with generation for the same cores.
- **Context shift** is on, so a session that outgrows its window keeps answering
  instead of failing. The trade is real: the oldest tokens silently fall off. Set
  `tuning.contextShift = false` if you would rather see the error.

Everything is overridable, in this order of precedence:

```
extraFlags  >  settings  >  tuning  >  what the probe inferred
```

`extraFlags` is raw argv appended last and never inspected. `settings` is a
flag-to-value map that locks individual flags. `tuning` is the semantic layer.

## Documentation retrieval

A small model cannot know what Hyprland changed last week. nix-llama ships a BM25
index over pinned, permissively licensed documentation and wires it in three ways:

1. **Automatically.** The retrieval proxy owns the public port, forwards everything to
   `llama-server` on loopback, and injects relevant excerpts into chat requests whose
   answer plausibly lives in the corpus. Chit-chat is left alone, retrieval never
   claims more than a quarter of the context window, and if anything goes wrong the
   original request is forwarded unchanged — a broken index must never take the model
   offline.
2. **On the model's initiative,** through `nix-llama-rag mcp` — see the harness
   sections above.
3. **Directly:**

```bash
nix-llama-rag --index ./result/rag/index.json.gz search "hyprland bindm"
nix-llama-rag --index ./result/rag/index.json.gz stats

curl 'http://127.0.0.1:8080/v1/rag/search?q=services.openssh.enable' | jq .
curl 'http://127.0.0.1:8080/v1/rag/stats' | jq .
```

Retrieval is lexical (BM25), not embedding-based. That is a deliberate trade: no
second model to load, no GPU memory spent on embeddings, a fully deterministic and
therefore Nix-cacheable index, and strong behaviour on the identifier-heavy queries
this corpus is made of (`services.openssh.enable`, `kubectl rollout`, `bindm`).

The index covers about 39,500 passages from 17 sources: the Nixpkgs and NixOS
manuals, **every NixOS module option declaration** (16,000 of them, extracted straight
from `nixos/modules`), nix-darwin and home-manager options, the Hyprland wiki, the Niri
wiki, the Nix manual and nix.dev, Kubernetes concepts/tasks/kubectl, GitHub Actions and
GitLab CI references, Codeberg/Forgejo documentation, Homebrew, and DXVK, vkd3d-proton
and Lutris for the Proton stack.

All of them are declared with their licences and pinned revisions in
[`nix/model/sources.json`](nix/model/sources.json) and listed in the `ATTRIBUTION.md`
of every built model. Refresh them with:

```bash
nix run github:heywoodlh/nix-llama#update-sources
```

## The model

`nix build github:heywoodlh/nix-llama#model` produces a directory, not just a file:

```
result/
├── model.gguf          weights llama-server loads (symlink into the store)
├── manifest.json       what this model is and what it was built from
├── system-prompt.md    the specialisation prompt the server injects
├── rag/index.json.gz   the documentation retrieval index
├── ATTRIBUTION.md      every source, its licence and its revision
└── MODEL_CARD.md       provenance, intended use, known limitations
```

Bundling them means the weights and the documentation they are meant to be used with
cannot drift apart.

| Target | What it is |
|---|---|
| `#model` | Qwen2.5-Coder 7B Instruct, Q4_K_M (4.68 GB) plus the nix-llama specialisation pack |
| `#model-small` | The 1.5B variant (1.1 GB), for 4 GB machines and Raspberry Pis |
| `#model-trained` | The full fine-tuning pipeline, run locally |
| `#corpus` | Just the documentation corpus, index and training dataset |

The build asserts that the weights stay under 5 GB, so "fits on an 8 GiB accelerator
alongside its KV cache" remains true rather than aspirational.

### Specialisation and provenance

The base model is Apache-2.0 licensed Qwen2.5-Coder. On top of that, nix-llama targets
Nix and the Nix ecosystem (nixpkgs, flakes, NixOS, nix-darwin, home-manager), Wayland
compositors (Hyprland, Niri), macOS tooling, CI/CD on GitHub, GitLab and
Codeberg/Forgejo, Kubernetes, and Linux gaming through Proton, DXVK and vkd3d-proton.

Until a nix-llama fine-tune is published, `#model` serves the base weights plus the
retrieval index and system prompt, and the generated model card says so explicitly.
`#model-trained` runs the fine-tune yourself.

### Licensing policy

Only material whose licence permits what we do with it is used, and this is enforced
by the build rather than by a note in a README. Sources fall into two tiers:

- **training + retrieval** — permissive licences (MIT, BSD, Apache-2.0, Zlib, ISC,
  CC-BY, CC0). Fine-tuning on these and redistributing excerpts are both unambiguously
  permitted.
- **retrieval only** — share-alike and weak-copyleft licences (CC-BY-SA, GPL, LGPL,
  MPL). These are indexed so the model can quote them *with attribution*, but never
  used as fine-tuning material: the share-alike obligation on a derived work is not
  something that can be honoured inside model weights.

A source with any other licence fails the build. Explicitly excluded, and why:
proprietary vendor documentation including Apple's; GFDL wikis such as the Arch Wiki;
anything without a published licence (which is why tangled.org is absent — revisit if
upstream adds one); Valve's Proton repository, whose per-directory licensing is mixed
and partly unstated, which is why DXVK (Zlib) and vkd3d-proton (LGPL-2.1) cover that
ground instead; and the output of other language models.

### Building the fine-tune

```bash
# Pin the base weights once (about 15 GB for the 7B).
nix run github:heywoodlh/nix-llama#update-base-model -- Qwen/Qwen2.5-Coder-7B-Instruct

# Then run the pipeline: corpus -> dataset -> LoRA -> merge -> GGUF -> quantize.
nix build github:heywoodlh/nix-llama#model-trained
```

Each stage is a separate derivation, so the expensive ones cache independently. The
LoRA stage is the one that wants a GPU, and Nix's sandbox does not expose accelerators
— see [AGENTS.md](AGENTS.md#the-training-pipeline) for how to run it with the sandbox
relaxed, and for the hyper-parameters. The 1.5B variant
(`#model-small-trained`) is small enough to be a reasonable first run.

## Configuration reference

Every option exists on the NixOS, nix-darwin and Home Manager modules under
`services.nix-llama` (see [Home Manager](#home-manager) for its differences). The CLI
accepts the same settings as a JSON file (`--config`) or environment variables
(`NIX_LLAMA_*`).

| Option | Default | Notes |
|---|---|---|
| `enable` | `false` | |
| `package` | follows `acceleration` | Set only if you built your own variant |
| `acceleration` | `"auto"` | `auto`, `none`, `cuda`, `rocm`, `vulkan`, `metal` |
| `model` | `pkgs.nix-llama-model` | A model bundle directory, or a path to a `.gguf` |
| `host` / `port` | `127.0.0.1` / `8080` | |
| `apiKeyFile` | `null` | Read at runtime, so it never enters the Nix store |
| `openFirewall` | `false` | Asserts that `apiKeyFile` is set |
| `systemPrompt` | from the model bundle | A path or an inline string |
| `stateDir` | `/var/lib/nix-llama` | Prompt caches and spilled KV slots |
| `user` / `group` | `nix-llama` | |
| `tuning.*` | inferred | See below |
| `diskCache.{enable,path,minFreeGiB}` | `true`, `<stateDir>/slots`, `8` | Flash storage only |
| `rag.{enable,index,topK,maxChars,minScore,injectInto}` | `true`, bundled, `4`, `6000`, `1.0`, `system` | |
| `tls.{certFile,keyFile}` | `null` | Usually better handled by a reverse proxy |
| `settings` | `{}` | `{ "--temp" = 0.2; }` — overrides inferred values |
| `extraFlags` | `[]` | Raw argv, appended last |
| `environment` | `{}` | Extra environment variables |

`tuning` keys, all defaulting to `null`/`"auto"` (meaning "let the probe decide"):
`contextSize`, `maxContextSize` (32768), `minContextSize` (4096), `gpuLayers`,
`threads`, `threadsBatch`, `batchSize`, `ubatchSize`, `parallel`, `kvCacheType`,
`promptCacheMiB`, `speculative`, `contextShift` (`true`), `loadMode`,
`flashAttention`, `corsOrigins` (`"localhost"`), `metrics` (`true`), `webui` (`true`).

Environment variables mirror the main settings: `NIX_LLAMA_MODEL`,
`NIX_LLAMA_MODEL_DIR`, `NIX_LLAMA_HOST`, `NIX_LLAMA_PORT`, `NIX_LLAMA_ACCELERATION`,
`NIX_LLAMA_CONTEXT_SIZE`, `NIX_LLAMA_THREADS`, `NIX_LLAMA_GPU_LAYERS`,
`NIX_LLAMA_KV_CACHE_TYPE`, `NIX_LLAMA_SPECULATIVE`, `NIX_LLAMA_DISK_CACHE`,
`NIX_LLAMA_RAG`, `NIX_LLAMA_RAG_INDEX`, `NIX_LLAMA_STATE_DIR`, `NIX_LLAMA_CONFIG`.

## Security

An LLM server is a service that executes untrusted input for a living. The defaults
reflect that, and CI fails any pull request that weakens them:

- **Loopback by default.** Binding wider without `apiKeyFile` produces a module
  warning, and `openFirewall` without it is a hard assertion failure. An open port in
  front of an unauthenticated `llama-server` lets anyone who can route to it spend
  your GPU and read whatever the model has been given.
- **CORS is restricted to localhost**, not llama.cpp's `*`. A wildcard with
  credentials reflected back is reachable from any page in your browser.
- **llama.cpp's built-in agent tools stay off.** `--tools`, `--agent` and the MCP proxy
  include `exec_shell_command`; enabling them by default would make every nix-llama
  deployment a remote code execution surface. A CI check asserts nix-llama never sets
  them.
- **Unprivileged everywhere.** A dedicated system account under systemd with
  `ProtectSystem=strict`, `ProtectHome`, `PrivateDevices` (relaxed only for GPU
  backends, with an explicit device allowlist), a `@system-service` syscall filter,
  and a capability set that contains only what the chosen load mode actually needs.
  The container runs as uid 1000 with a read-only root filesystem.
- **Secrets stay out of the store.** `apiKeyFile` is a path read at runtime; there is
  deliberately no inline `apiKey` option, because option values are world-readable in
  `/nix/store`.
- **One flake input.** Every dependency is a nixpkgs package pinned by `flake.lock`,
  and CI fails if the input set grows without the check being updated.

To report a security issue, open a GitHub security advisory rather than a public issue.

## Troubleshooting

**It is running on the CPU even though I have a GPU.** `nix-llama probe` will say so:
`detected cuda hardware but the llama-server build exposes no matching device`. The
backend is a build-time choice — set `services.nix-llama.acceleration`, or use the
matching package (`#nix-llama-cuda`, `#nix-llama-rocm`, `#nix-llama-vulkan`). On
NixOS, `acceleration = "auto"` reads your configuration, not your hardware, so a
headless box with a GPU but no `hardware.graphics.enable` will guess CPU.

**Requests fail with "exceeds the available context size".** The client is sending
more than the server's context. `nix-llama plan` prints the context it chose; set the
same number in your harness's `context_length`/`contextWindow`. To raise it, either
`tuning.contextSize` with `tuning.kvCacheType = "q8_0"`, or accept less GPU offload.

**It is slower than expected.** Check `nix-llama plan` output for the thread count
(should be your physical core count), whether `--slot-save-path` appears (disk KV
cache), and whether `--spec-type` appears (speculative decoding, GPU only). On CPU,
make sure the model is `mmap+mlock` rather than plain `mmap` — that requires enough
free memory for the weights to be pinned.

**The first request after a restart is slow, the rest are fast.** Expected: prompt
caches live in RAM (`--cache-ram`) and on disk (`--slot-save-path`). Persist
`stateDir` across restarts so the disk cache survives.

**Retrieval never fires.** `curl localhost:8080/v1/rag/stats`. Zero chunks means no
index was found; `nix-llama doctor` reports where it looked. Retrieval also goes inert
when the context window leaves no room for it, which is logged at startup.

**On a machine running the service**, `nix-llama probe`, `plan` and `doctor` typed into
a shell describe *that* service: the modules install the rendered configuration at
`/etc/nix-llama/config.json` and the CLI picks it up. `journalctl -u nix-llama` carries
the same explanation lines from startup, plus one line per augmented request.

## Development

```bash
nix develop            # python, pytest, ruff, nixfmt, statix, deadnix, llama-server
pytest                 # 184 unit tests, no network, under 10 seconds
nix flake check        # everything CI gates on
nix fmt                # format all Nix
```

[AGENTS.md](AGENTS.md) documents the repository layout, the architecture, how each
subsystem works, and how to contribute — written so an LLM (or a human in a hurry) can
navigate it without reading every file.

## Licence

MIT for the code; see [LICENSE](LICENSE). Model artifacts are Apache-2.0 for the
weights, with the retrieval index carrying the share-alike obligations of some of its
sources — the `ATTRIBUTION.md` and `MODEL_CARD.md` in each built model have the
specifics.

## AI/LLM Disclosure

This was written using AI with the hope that the author could pivot to a more
sustainable, self-hosted LLM deployment.
