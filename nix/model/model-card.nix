# Generate MODEL_CARD.md for the packaged model.
#
# The card is generated rather than hand-written so it cannot drift from the
# thing it describes: every source, revision and licence in it comes from the
# same data the corpus was built from.
{
  lib,
  writeText,
  version,
  variant,
  spec,
  manifest,
  corpus,
  provenance,
}:

let
  sourceRow =
    s:
    "| ${s.name} | ${s.license} | ${lib.concatStringsSep " + " s.usage} | `${builtins.substring 0 12 s.rev}` |";

  provenanceText =
    {
      upstream-base = ''
        These are the **upstream base weights**, unmodified. No nix-llama fine-tune has
        been published for this variant yet, so the specialisation you get comes from
        the system prompt and the retrieval index rather than from altered weights.
        Build `.#model-trained` to run the fine-tune yourself.
      '';
      nix-llama-finetune = ''
        These weights are a **published nix-llama fine-tune** of the base model, trained
        on the permissively licensed corpus described below and re-quantized to
        ${spec.quantization}.
      '';
      trained-locally = ''
        These weights were **fine-tuned by this build** from the base model, on the
        permissively licensed corpus described below, then converted to GGUF and
        quantized to ${spec.quantization}.
      '';
    }
    .${provenance};
in
writeText "nix-llama-model-card.md" ''
  # nix-llama ${variant} (v${version})

  ${spec.description}, packaged for self-hosted serving through llama.cpp.

  | | |
  |---|---|
  | Parameters | ${spec.params} |
  | Quantization | ${spec.quantization} |
  | Base model | ${spec.hfRepo} |
  | Weights licence | ${manifest.license} |
  | Provenance | ${provenance} |

  ## What these weights are

  ${provenanceText}

  ## Intended use

  Coding and systems administration on the user's own hardware: Nix and the Nix
  ecosystem (nixpkgs, flakes, NixOS, nix-darwin, home-manager), Wayland compositors
  (Hyprland, Niri), macOS tooling, CI/CD on GitHub, GitLab and Codeberg/Forgejo,
  Kubernetes, and Linux gaming through Proton, DXVK and vkd3d-proton.

  It is sized to run on a single 8 GiB accelerator, or on CPU with 8 GiB of RAM. It
  is not a general-purpose assistant and it is not a replacement for reading the
  documentation it cites.

  ## Training and retrieval data

  Every source below is pinned to a revision and carries a licence that permits what
  nix-llama does with it. Sources are used in one of two ways:

  - **training + retrieval** -- permissive licences (MIT, BSD, Apache-2.0, Zlib,
    CC-BY). Fine-tuning on these and redistributing excerpts are both permitted.
  - **retrieval** only -- share-alike or weak-copyleft licences (CC-BY-SA, GPL,
    LGPL). Indexed so the model can quote them *with attribution*, never used as
    fine-tuning material.

  | Source | Licence | Used for | Revision |
  |---|---|---|---|
  ${lib.concatMapStringsSep "\n" sourceRow corpus.summary}

  No copyrighted material was used outside these terms. Specifically excluded:
  proprietary vendor documentation (including Apple's), GFDL wikis, any source
  without a published licence, and the output of other language models.

  See `ATTRIBUTION.md` in this directory for the full notice, including the
  share-alike obligations that travel with the retrieval index.

  ## Known limitations

  - **Quantization.** ${spec.quantization} trades some accuracy for fitting in 8 GiB.
    Expect occasional wrong-but-plausible option names; this is exactly what the
    retrieval index exists to catch.
  - **Documentation freshness.** The index is pinned at build time. Rebuild the model
    (or run `nix run .#update-sources`) to pick up upstream changes. Hyprland and Niri
    move fastest.
  - **Small-model reasoning.** A ${spec.params} model will lose a long multi-step
    debugging thread that a frontier model would hold. Ask narrower questions.
  - **Templated instruction data.** The fine-tuning pairs derived from documentation
    are generated deterministically from structure (option declarations, headings,
    examples), not curated by hand. They teach recall and formatting, not reasoning.

  ## Reproducing this build

  ```
  nix build github:heywoodlh/nix-llama#model${lib.optionalString (variant != "default") "-${variant}"}
  ```
''
