# The nix-llama model package.
#
# `nix build .#model` produces a *model directory*, not just a weights file:
#
#   model.gguf          the quantized weights llama-server loads
#   manifest.json       what this model is, what it was built from
#   system-prompt.md    the specialisation prompt the server injects
#   rag/index.json.gz   the documentation retrieval index
#   ATTRIBUTION.md      per-source licences and revisions
#   MODEL_CARD.md       provenance, intended use and known limitations
#
# That bundle is the unit the service consumes, so the weights and the
# documentation they are meant to be used with can never drift apart.
{
  lib,
  stdenvNoCC,
  fetchurl,
  nix-llama,
  nix-llama-corpus,
  version,
  # "default" (7B) or "small" (1.5B); "test" is the CI fixture.
  variant ? "default",
  # Build the fine-tune from scratch instead of fetching published weights.
  trained ? false,
  # Which documentation corpus to bundle. Overridden by the test variant, which
  # uses a fixture corpus so the tests do not fetch every pinned repository.
  corpus ? nix-llama-corpus,
  callPackage,
  writeText,
}:

let
  weights = import ./weights.nix;
  spec =
    weights.variants.${variant}
      or (throw "nix-llama: unknown model variant ${variant}; choose one of ${lib.concatStringsSep ", " (lib.attrNames weights.variants)}");

  release = weights.release.${variant} or null;

  trainedModel = callPackage ./train.nix { inherit version variant corpus; };

  # Three ways to get weights, in descending preference:
  #   1. built here from the training pipeline (`trained = true`)
  #   2. a published nix-llama fine-tune
  #   3. the upstream Apache-2.0 base model
  source =
    if trained then
      {
        kind = "trained-locally";
        gguf = "${trainedModel}/${trainedModel.ggufName}";
        inherit (spec) license;
        base = spec.hfBaseRepo;
      }
    else if release != null then
      {
        kind = "nix-llama-finetune";
        gguf = fetchurl { inherit (release) url hash; };
        license = "apache-2.0";
        base = spec.hfRepo;
      }
    else
      {
        kind = "upstream-base";
        gguf = fetchurl {
          inherit (spec) url hash;
          name = spec.file;
        };
        inherit (spec) license;
        base = spec.hfRepo;
      };

  # The test fixture has a 2048-token context, which the real system prompt
  # would consume most of. Shipping a short one keeps the smoke tests
  # exercising retrieval rather than hitting the context limit.
  systemPrompt = if variant == "test" then ./system-prompt-test.md else ./system-prompt.md;

  # The 5 GB ceiling is a hard product requirement -- the model has to fit
  # comfortably on an 8 GiB accelerator alongside its KV cache -- so it is
  # asserted at build time rather than trusted.
  maxBytes = 5000000000;

  manifest = {
    name = "nix-llama-${variant}";
    inherit version variant;
    weights = "model.gguf";
    provenance = source.kind;
    baseModel = source.base;
    inherit (source) license;
    inherit (spec) quantization params description;
    systemPrompt = "system-prompt.md";
    rag = {
      index = "rag/index.json.gz";
      sources = "rag/sources.json";
    };
    sources = corpus.summary;
  };

  manifestFile = writeText "nix-llama-model-manifest.json" (builtins.toJSON manifest);

  modelCard = callPackage ./model-card.nix {
    inherit
      version
      variant
      spec
      manifest
      ;
    inherit corpus;
    provenance = source.kind;
  };
in
stdenvNoCC.mkDerivation {
  pname = "nix-llama-model${lib.optionalString (variant != "default") "-${variant}"}";
  inherit version;

  dontUnpack = true;
  nativeBuildInputs = [ nix-llama ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/rag

    # A symlink, not a copy: the weights are already a store path, and
    # duplicating 4.7 GB per rebuild is not a cost worth paying.
    ln -s ${source.gguf} $out/model.gguf

    cp ${systemPrompt} $out/system-prompt.md
    cp ${corpus}/rag/index.json.gz $out/rag/index.json.gz
    cp ${corpus}/rag/sources.json $out/rag/sources.json
    cp ${corpus}/ATTRIBUTION.md $out/ATTRIBUTION.md
    cp ${manifestFile} $out/manifest.json
    cp ${modelCard} $out/MODEL_CARD.md

    echo "verifying weights..."
    nix-llama inspect $out/model.gguf --max-bytes ${toString maxBytes}

    runHook postInstall
  '';

  passthru = {
    inherit spec variant manifest;
    weights = source.gguf;
    provenance = source.kind;
    inherit corpus;
  };

  meta = {
    description = "nix-llama model: ${spec.description} with the nix-llama specialisation pack";
    longDescription = ''
      A ${spec.params} coding and systems-administration model quantized to
      ${spec.quantization}, bundled with a documentation retrieval index and a
      system prompt specialised for Nix, NixOS, nix-darwin, home-manager,
      Hyprland, Niri, macOS tooling, CI/CD, Kubernetes and Linux gaming.
    '';
    homepage = "https://github.com/heywoodlh/nix-llama";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
}
