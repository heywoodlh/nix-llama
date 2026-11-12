# AGENTS.md

Repository guide for coding agents and for humans who want the map before the
territory. If you are changing code here, read the section for the subsystem you are
touching plus [Invariants](#invariants) and [Before you open a PR](#before-you-open-a-pr).

## What this repository is

nix-llama is a self-tuning `llama-server` deployment plus the model it serves. Three
things live here:

1. **A Python application** (`src/nix_llama/`) that probes the host, derives a
   `llama-server` command line from those facts plus the model's GGUF header, runs it,
   and fronts it with a documentation-retrieval proxy.
2. **Nix packaging** (`flake.nix`, `nix/`, `modules/`): the application, accelerated
   `llama.cpp` variants, the model bundle and its training pipeline, container images,
   and NixOS/nix-darwin service modules.
3. **A documentation corpus** (`nix/model/sources.json` + `sources.lock.json`) that
   becomes a retrieval index and fine-tuning dataset, under a licence policy the build
   enforces.

The original design brief is [`DESIGN.md`](DESIGN.md). It is kept as written — the
intent, not a changelog.

## Layout

```
flake.nix                    packages, apps, checks, devShells, modules
VERSION                      single source of truth for the version
CHANGELOG.md                 Keep a Changelog format; a release needs a section here

src/nix_llama/
  probe.py                   host facts: accelerators, memory, CPU, storage, container
  gguf.py                    dependency-free GGUF header reader
  defaults.py                the planner: facts + model + config -> argv  (pure)
  config.py                  config loading, merging, validation
  cli.py                     `nix-llama`: serve / probe / plan / doctor / inspect
  dataset.py                 corpus -> fine-tuning records
  rag/
    index.py                 chunking and BM25 index construction
    retrieve.py              query execution and prompt formatting
    corpus.py                source manifests and the licence allowlist
    proxy.py                 the augmenting OpenAI-compatible reverse proxy
    mcp.py                   stdio MCP server exposing search_docs
    cli.py                   `nix-llama-rag`: index / corpus / search / serve / mcp

nix/
  package.nix                the Python application derivation
  overlay.nix                everything this flake adds to a nixpkgs instance
  container.nix              OCI images and the closure the Dockerfile copies
  checks.nix                 every `nix flake check` check
  update-sources.py          re-pins the documentation corpus
  update-base-model.py       pins base model weights for the training pipeline
  model/
    default.nix              assembles the model bundle
    weights.nix              the weights registry (URLs, hashes, sizes, licences)
    sources.json             documentation sources, hand-maintained
    sources.lock.json        their revisions and hashes, generated
    base.lock.json           base model weights, per file, generated
    corpus.nix               builds the retrieval index and training dataset
    test-corpus.nix          the same, over three fixture files, for tests
    train.nix                the fine-tuning pipeline
    model-card.nix           generates MODEL_CARD.md
    system-prompt.md         the specialisation prompt shipped with the model

modules/
  options.nix                option declarations, shared by both platforms
  settings.nix               renders the JSON config and picks the package
  nixos.nix                  systemd service and hardening
  darwin.nix                 launchd daemon and service account
  home-manager.nix           systemd user unit / launchd agent, no service account

training/
  train_lora.py              LoRA fine-tune (needs torch; runs in nix/model/train.nix)
  merge_lora.py              folds the adapter into the base weights

tests/
  conftest.py                synthetic /proc and /sys trees, synthetic GGUF files
  test_*.py                  184 unit tests, no network, under 10 seconds
  nixos-service.nix          the NixOS VM integration test
  fixtures/docs/             three markdown files used as a miniature corpus

docker/Dockerfile            the second way to build the image; CI checks parity
examples/                    docker-compose.yml, kubernetes.yaml
.github/workflows/           ci, security, release, model
```

## Architecture

### The serving path

```
client
  │
  ▼  :8080  (public)
nix-llama-rag proxy ──────── BM25 index (rag/index.json.gz)
  │                            injects excerpts into chat requests
  ▼  :8081  (loopback)
llama-server  ← argv from nix_llama.defaults.plan()
  │                            ← facts from nix_llama.probe.probe()
  ▼                            ← geometry from nix_llama.gguf.read_gguf()
model.gguf
```

`nix-llama serve` does all of this in one process tree: it plans, execs
`llama-server` as a child on a loopback port, and runs the proxy in a thread on the
public port. With `rag.enable = false`, the proxy is skipped and `llama-server` binds
the public port directly.

One asymmetry worth knowing: `apps.default` (what `nix run` with no arguments uses)
wraps the program with `NIX_LLAMA_MODEL_DIR` pointing at the default model bundle, so
`nix run github:heywoodlh/nix-llama` serves something immediately. `packages.default`
and `apps.nix-llama` deliberately do not carry the model, so building the software
never pulls 4.7 GB of weights.

One asymmetry worth knowing: `apps.default` (what `nix run` without arguments uses)
wraps the program with `NIX_LLAMA_MODEL_DIR` pointing at the default model bundle, so
`nix run github:heywoodlh/nix-llama` serves something immediately. `packages.default`
and `apps.nix-llama` deliberately do not carry the model, so building the software
never pulls 4.7 GB of weights.

### Why probe and plan are separate

`probe.py` does all the I/O and knows nothing about `llama-server` flags.
`defaults.plan()` is a pure function from `(config, Probe, GGUFInfo)` to an argv list
plus an explanation. That split is what makes the defaults testable: `tests/
test_defaults.py` constructs a 24-core machine with a 24 GB GPU without having one,
and `tests/test_probe.py` points the probe at a fixture directory.

Keep it that way. If you need a new fact, add it to `Probe`; if you need a new flag,
add it to `plan()`.

### Precedence

Both the CLI and the modules implement the same order, strongest first:

1. `extraFlags` — raw argv appended last, never inspected
2. `settings` — a `{"--flag": value}` map; each entry *locks* that flag
3. `tuning` — the semantic layer (`contextSize`, `threads`, …)
4. whatever the probe and the GGUF header imply

Implemented by `_Args` in `defaults.py`: user layers are applied first with
`lock = true`, so nothing computed afterwards can displace them.

### Configuration

One shape, three producers. `modules/settings.nix` renders the JSON the service reads;
`nix_llama.config.DEFAULTS` holds the built-in values; `NIX_LLAMA_*` environment
variables and CLI flags layer on top. **Null-valued module options are dropped rather
than serialised**, which keeps `config.py` the single place a default lives. If you
add an option, add it in both `modules/options.nix` and `config.py:DEFAULTS`, and let
the Nix side pass `null` for "unset".

### Retrieval

BM25, not embeddings — deliberately. No second model to load, no GPU memory for
embeddings, a deterministic (therefore Nix-cacheable) index, and good behaviour on the
identifier-heavy queries this corpus is full of. If you are tempted to add embeddings,
the cost is a second model in memory and a non-reproducible index; make that case
explicitly.

Four scoring decisions were measured against the real corpus rather than guessed, and
each one fixed a query class that was previously broken. Changing them is fine;
changing them without re-measuring is not:

- **Nix modules are indexed per option**, not as windows of source (`option_chunks` in
  `rag/corpus.py`). `services.openssh.enable` has to be findable by its own name.
- **Titles are weighted** (`TITLE_WEIGHT = 3`). Without it, a long introduction page
  that says "home-manager" twenty times beats the option someone actually asked for.
- **Dotted query tokens are weighted** (`PATH_WEIGHT = 2.5`). Only dots: hyphens were
  tried and made things worse, because "nix-darwin" and "fail-fast" are hyphenated
  *words*, so boosting them promoted whichever README repeats the project's own name.
- **Dotted prefixes are indexed**, so `programs.git` reaches every `programs.git.*`
  option, and **plurals are normalised** so "which options enable sshd" reaches a
  document that says "option". Verb stemming (-ing/-ed) was tried and removed: no rule
  short of a real stemmer maps "enabled" to "enable" without also splitting apart
  things that must match.

The index format is gzipped JSON with sorted keys and `mtime=0`, so building the same
corpus twice is byte-identical. `INDEX_VERSION` in `rag/index.py` must be bumped on
any format change; loading a mismatched version fails loudly rather than silently
misbehaving.

### The licence policy

This is the part most likely to be weakened by accident. `rag/corpus.py` holds
`LICENSE_ALLOWLIST` and `SHARE_ALIKE`; `nix/model/sources.json` declares each source's
licence and a `usage` list. The rules:

- Permissive licence → `["training", "retrieval"]` is allowed.
- Share-alike or weak copyleft → `["retrieval"]` only. Model weights cannot carry a
  share-alike obligation, so such material is indexed and quoted with attribution but
  never trained on.
- Any other licence → the build fails.

Two things enforce it: `build_corpus()` raises `LicenseError` at build time, and the
`corpus-licenses` flake check cross-references `sources.json` against the allowlist.
Adding a source means adding it to `sources.json`, running `nix run .#update-sources`,
and justifying the licence in the PR.

### The training pipeline

`nix/model/train.nix` is four chained derivations so each expensive stage caches
independently:

```
base      fetch the Apache-2.0 base model, one fixed-output derivation per file
adapter   LoRA fine-tune on the permissive corpus          (training/train_lora.py)
merged    fold the adapter into the base weights           (training/merge_lora.py)
gguf      convert with llama.cpp's convert_hf_to_gguf.py, then llama-quantize
```

Running it:

```bash
# once per variant: pins the base weights in nix/model/base.lock.json
nix run .#update-base-model -- Qwen/Qwen2.5-Coder-1.5B-Instruct --as small

# then
nix build .#model-small-trained
```

The `adapter` stage wants a GPU, and Nix's sandbox does not expose accelerators. Two
options: build with `--option sandbox relaxed` and
`.override { impureGpu = true; }` (which sets `__noChroot`), or run it on a builder
configured for that. On CPU the pipeline is correct but slow — fine for the 1.5B
variant, measured in days for the 7B.

Hyper-parameters live in `defaults` inside `train.nix` and are overridable:
`.override { trainingConfig = { rank = 64; epochs = 2; }; }`.

**Known limitation, stated plainly:** the instruction pairs in `dataset.py` are
generated deterministically from structure — `mkOption` declarations, question-shaped
headings, documented code fences. That is reproducible, attributable, and free of
other models' output, but templated pairs teach recall and formatting rather than
reasoning. A curated instruction set would be better and is the obvious next
improvement; it must stay licence-clean.

### Container parity

The image is built two ways and they must agree. `nix build .#container` streams a
layered image; `docker build -f docker/Dockerfile` runs Nix inside stage 1 to build
`.#container-closure` and copies it onto `scratch`. Both take `llama-server` from this
flake's pinned nixpkgs, so the binary is the same store path the service runs.

The `container-parity` flake check diffs the Dockerfile's `ENV` lines against
`nix/container.nix`'s `Env`. Change one and you must change the other.

### Service hardening, and one trap in it

`modules/nixos.nix` applies most of systemd's sandbox. One restriction is
**deliberately absent**, and the NixOS VM test is what catches its return:

`SystemCallFilter` does **not** include `~@resources`. That group denies
`sched_setaffinity` and `setpriority`, which llama.cpp's threadpool calls while the
model loads — the service dies with SIGSYS a second after startup. It presents badly
too: a child killed by a signal used to be returned as a negative exit code, which
wrapped into systemd's reserved range and was reported as `225/NETWORK`, i.e. "failed
to set up networking". `cmd_serve` now converts signal deaths to `128 + signal` so the
next person reads the right diagnosis.

Capabilities follow need rather than habit: `CAP_IPC_LOCK` only when the load mode
uses mlock, `CAP_NET_BIND_SERVICE` only for a privileged port, and the bounding set is
kept equal to what is granted. `PrivateDevices` is relaxed only for GPU backends, with
an explicit `DeviceAllow` list.

## Invariants

Things that are enforced, and that a change must not quietly reverse. Most have a
check attached; the check names are in `nix/checks.nix`.

| Invariant | Enforced by |
|---|---|
| Default listen address is `127.0.0.1` | `security-policy` |
| Default CORS origin is `localhost`, never `*` | `security-policy` |
| nix-llama never sets llama.cpp's `--tools` / `--agent` / MCP-proxy flags | `security-policy` |
| `openFirewall` requires `apiKeyFile` | module assertion + `security-policy` |
| Secrets are file paths, never option values | `security-policy` |
| No `shell=True` anywhere | `security-policy` + ruff `S` rules |
| Model weights stay under 5 GB | `nix-llama inspect --max-bytes` in `model/default.nix` |
| Only allowlisted licences enter the corpus | `build_corpus()` + `corpus-licenses` |
| Share-alike sources never reach training data | `corpus-licenses` |
| The retrieval index is byte-reproducible | `test_index_is_byte_reproducible` |
| Injected context never exceeds its budget | `test_the_formatted_block_respects_the_total_budget` |
| Retrieval failure never fails a request | `test_a_malformed_chat_body_is_relayed_unchanged` |
| The Dockerfile and `container.nix` agree | `container-parity` |
| `VERSION`, the flake and `CHANGELOG.md` agree | `version-consistency` |
| There is exactly one flake input | `flake-input-review` in `security.yml` |
| `SystemCallFilter` excludes `~@resources` (llama.cpp needs it) | `nixos-module` VM test |
| The container does not run as root | `security.yml` + `container-parity` |

## Conventions

**Python.** Standard library only — nix-llama sits in front of a model server on an
8 GiB box, and the runtime budget is the stdlib. Type hints throughout,
`from __future__ import annotations`, ruff for lint and format (line length 100,
`E,F,W,I,N,UP,B,C4,SIM,RUF,S` selected). A `# noqa` needs a reason on the same line or
just above it.

**Nix.** `nixfmt-rfc-style` (`nix fmt`), `statix` and `deadnix --fail` all clean.
Derivations are commented with *why*, not what.

**Comments.** Explain decisions and trade-offs, not syntax. The useful comment in this
codebase is the one that says why hyper-threads are excluded from the thread count, or
why unknown storage counts as not-flash. A comment restating the code is noise.

**Tests.** Every behavioural change needs a test. Unit tests must not touch the
network and must not depend on the host's hardware — use the `fakeroot`/`nvidia_root`/
`amd_root` fixtures and `conftest.build_gguf`. Name tests after the behaviour
(`test_disk_cache_requires_flash_storage`), not the function.

**Commits.** Conventional Commits (`feat:`, `fix:`, `docs:`, `corpus:`, `ci:`).

**Versioning.** Semantic, repository-wide: the CLI, the modules, the image and the
model bundle all ship one version because they are used together. A renamed module
option, an incompatible index format, or a changed model prompt format is a major
bump.

## Common tasks

```bash
# development shell with every tool CI uses
nix develop

# unit tests (fast, hermetic)
pytest
pytest tests/test_defaults.py -k context    # one area

# the whole gate
nix flake check

# one check, with logs
nix build --print-build-logs .#checks.x86_64-linux.serve-smoke

# see what the planner would do on this machine
python -m nix_llama.cli probe
python -m nix_llama.cli plan --model-dir ./result

# refresh the documentation corpus (network)
nix run .#update-sources
nix build .#corpus && cat result/rag/stats.txt

# rebuild the fixture corpus used by the tests
nix build .#packages.x86_64-linux.model-test
```

### Adding a tuning knob

1. `modules/options.nix` — declare it, with a description that says what the default
   infers and when you would override it.
2. `modules/settings.nix` — pass it through (null means "unset").
3. `src/nix_llama/config.py` — add it to `DEFAULTS`, and to `_INT_KEYS`/`_BOOL_KEYS`
   if it is coerced from an environment variable.
4. `src/nix_llama/defaults.py` — use it in `plan()`, and append a line to
   `result.explain` saying what you chose and why.
5. `tests/test_defaults.py` — one test for the inferred value, one for the override.
6. `README.md` — the configuration table.

### Adding a documentation source

1. Verify the licence from the repository itself, not from a badge.
2. Add it to `nix/model/sources.json` with `usage` per the licence policy above.
3. `nix run .#update-sources` to pin it.
4. `nix build .#corpus` and read `result/rag/stats.txt` — a source contributing almost
   nothing is not worth its download.
5. Mention the licence and tier in the PR description.

### Debugging a slow or wrong deployment

`nix-llama plan` first, always. It prints the exact command line and one line of
reasoning per decision. `nix-llama probe` shows the facts underneath. `nix-llama
doctor` checks that the binary, the model, the state directory and the index are all
where they should be. In a running service, `journalctl -u nix-llama` carries the same
explanation lines from startup, plus one line per augmented request.

## Before you open a PR

```bash
nix fmt
pytest
nix flake check
```

CI runs the same checks plus the NixOS VM test, both container builds, the corpus
build, CodeQL, secret scanning and image scanning. The security workflow is a merge
gate, not advisory.

If your change touches anything in [Invariants](#invariants), say so explicitly in the
PR description and explain why the new behaviour is still correct.

## CI/CD and credentials

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push, PR | lint and policy checks, unit tests on Linux and macOS, serve/MCP smoke tests, NixOS VM test, corpus build, both container builds and a parity check |
| `security.yml` | push, PR, weekly | deployment policy, secret scanning (tree and history), ruff security rules, CodeQL, trivy image scan, dependency review, flake input review |
| `release.yml` | `v*` tag | verifies the tag matches `VERSION`, full `nix flake check`, multi-arch images to GHCR with semver tags, GitHub release from the changelog section |
| `model.yml` | `v*` tag, manual, weekly | builds the model, verifies the 5 GB budget, publishes to Hugging Face; weekly it re-pins the corpus and opens a PR |

**Credentials to configure in the repository** (only needed for publishing):

- Secret `HF_TOKEN` — a Hugging Face access token with **write** permission to the
  target model repository. Create it at
  <https://huggingface.co/settings/tokens> with the "Write access to contents/settings
  of selected repos" scope, scoped to the repo in `HF_REPO`.
- Variable `HF_REPO` — the target repo id, e.g. `heywoodlh/nix-llama`.

GHCR needs no secret: `release.yml` uses the built-in `GITHUB_TOKEN` with
`packages: write`. If you want faster CI, add `CACHIX_AUTH_TOKEN` and a cache name and
push the flake outputs — nothing in the workflows depends on it today.
