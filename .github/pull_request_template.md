## What this changes

<!-- One or two sentences. What is different afterwards? -->

## Why

<!-- The problem, not the patch. -->

## Checks

- [ ] `pytest` passes
- [ ] `nix fmt` is clean
- [ ] `nix flake check` passes
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`

## Invariants

AGENTS.md lists the behaviours CI enforces — the loopback default, the CORS policy,
llama.cpp's agent tools staying off, secrets as file paths, the 5 GB model budget, the
documentation licence tiers, Dockerfile/`container.nix` parity, the single flake input.

- [ ] This change does not cross any of them, **or** it does and the reason is below.

<!-- If it does: which one, and why is the new behaviour still correct? -->

## Documentation corpus

<!-- Only if nix/model/sources.json changed. -->

- [ ] Licence verified from the repository itself, not from a badge
- [ ] `usage` tier matches the licence (share-alike and copyleft are retrieval-only)
- [ ] `nix run .#update-sources` run, and `nix build .#corpus` shows the source earns
      its download
