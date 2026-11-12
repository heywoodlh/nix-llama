# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versioning applies to the whole repository as one unit: the CLI, the service
modules, the container image and the model bundle all ship under the same
version, because they are expected to be used together. A breaking change to
any of them -- a renamed module option, an incompatible retrieval index format,
a model whose prompt format changed -- is a major bump.

## [Unreleased]

### Added

- **Home Manager module** (`homeManagerModules.default`). Runs nix-llama as a systemd
  user unit on Linux or a launchd agent on macOS, with the same options as the system
  modules. It supplies its own packages, so it works under `useGlobalPkgs`.

## [0.1.0] - 2026-11-11

The first release.

### Added

- **One-command serving.** `nix run github:heywoodlh/nix-llama` fetches the default
  model bundle, probes the machine and serves an OpenAI-compatible endpoint. Building
  the software alone (`nix build`) deliberately does not pull the weights.
- **`nix-llama` server.** Probes the host (accelerators, memory, CPU topology,
  cgroup limits, storage class, container runtime), reads the model's GGUF
  header, and derives a llama-server invocation sized for the combination.
  `nix-llama probe`, `plan`, `doctor`, `inspect` and `serve` subcommands.
- **Hardware detection** for NVIDIA (CUDA), AMD (ROCm and Vulkan), Intel
  (Vulkan) and Apple silicon (Metal), including inside containers, and
  cross-checked against the backends the llama.cpp binary was actually built
  with so an unusable GPU produces a warning rather than silent CPU fallback.
- **Disk-backed KV cache.** Slot caches spill to local flash when the probe
  finds it, so long conversations resume instead of being re-evaluated.
- **Documentation retrieval (RAG).** A BM25 index over ~39,500 passages from 17
  pinned documentation sources, exposed three ways: an OpenAI-compatible proxy that
  augments chat requests automatically, an MCP server so a harness can let the model
  look things up itself, and a `nix-llama-rag search` CLI. Every NixOS module option
  is indexed individually (16,000 of them, extracted from `nixos/modules`), so
  `services.openssh.enable` is findable by name rather than buried in a window of
  surrounding Nix. Injected context is capped at a share of the context window, so
  retrieval can never turn a request into a context-overflow error.
- **Licence policy enforced at build time.** Permissive sources are used for
  both fine-tuning and retrieval; share-alike sources are indexed for retrieval
  only and never used as training material. An unvetted licence fails the build.
- **Model packages.** `#model` (Qwen2.5-Coder 7B Instruct, Q4_K_M, 4.68 GB) and
  `#model-small` (1.5B), each bundled with the retrieval index, a specialised
  system prompt, a generated model card and an attribution file.
- **Fine-tuning pipeline.** `#model-trained` runs the full pipeline as Nix
  derivations: corpus, dataset, LoRA fine-tune, merge, GGUF conversion,
  quantization. Base weights are pinned per-file for reproducibility.
- **NixOS and nix-darwin modules** with the same option surface, systemd
  hardening, a service account, and assertions that refuse an unauthenticated
  open port.
- **Container images** built from the same flake, so the `llama-server` binary
  in the image is the same store path the service runs. A `Dockerfile` kept in
  parity by CI, plus `#container` and `#container-model`.
- **CI/CD**: `nix flake check` on Linux and macOS, a NixOS VM integration test,
  security gates (secret scanning, image scanning, policy checks), GHCR
  publishing and Hugging Face model publishing.

[Unreleased]: https://github.com/heywoodlh/nix-llama/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/heywoodlh/nix-llama/releases/tag/v0.1.0
