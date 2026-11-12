# Contributing

Thanks for looking. [AGENTS.md](AGENTS.md) is the real guide — it documents the
repository layout, how each subsystem works, the invariants CI enforces, and
step-by-step recipes for the common kinds of change. This file is the short version.

## Getting set up

```bash
git clone https://github.com/heywoodlh/nix-llama
cd nix-llama
nix develop     # python, pytest, ruff, nixfmt, statix, deadnix, llama-server, …
```

## The loop

```bash
pytest                  # 184 unit tests, hermetic, under 10 seconds
nix fmt                 # format every .nix file
nix flake check         # everything CI gates on
```

Unit tests must not touch the network and must not depend on your hardware: the
fixtures in `tests/conftest.py` provide synthetic `/proc` and `/sys` trees and
synthetic GGUF files for exactly that reason.

## What a good change looks like

- **One concern per pull request.** A tuning knob and a corpus addition are two PRs.
- **A test for the behaviour**, named after the behaviour.
- **A `CHANGELOG.md` entry** under `## [Unreleased]`.
- **Conventional Commit messages**: `feat:`, `fix:`, `docs:`, `corpus:`, `ci:`.
- **Comments that explain decisions**, not syntax. The valuable comment here is the
  one that says why unknown storage counts as not-flash, not the one that says a loop
  iterates.

## Things that will fail CI

The security workflow is a merge gate. It fails a PR that weakens a deployment's
posture: binding beyond loopback by default, a CORS wildcard, enabling llama.cpp's
built-in shell tools, an inline `apiKey` option, `shell=True`, an un-vetted
documentation licence, a model over 5 GB, a Dockerfile that drifts from
`nix/container.nix`, or a second flake input.

Every one of these is listed with its enforcing check in
[AGENTS.md](AGENTS.md#invariants). If your change genuinely needs to cross one of
those lines, say so in the PR description and make the case — the checks exist to make
the decision deliberate, not to make it impossible.

## Adding a documentation source

The corpus is licence-gated, and the gate is the point. Verify the licence from the
repository itself, add the source to `nix/model/sources.json` with the right `usage`
tier (permissive licences may be used for training and retrieval; share-alike and
copyleft licences are retrieval-only), run `nix run .#update-sources`, and state the
licence and tier in the PR. The full policy is in
[AGENTS.md](AGENTS.md#the-licence-policy).

## Reporting bugs

Please include the output of:

```bash
nix-llama probe
nix-llama plan --format json
nix-llama doctor
```

Those three cover nearly everything anyone would ask you for: what was detected, what
was run and why, and whether the pieces are where they should be.

## Security issues

Open a GitHub security advisory rather than a public issue.
