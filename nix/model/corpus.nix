# Build the documentation corpus: a BM25 retrieval index, the attribution file
# the share-alike licences require, and a training corpus for the fine-tune.
#
# Two manifests go in, because the licence policy differs by use: everything
# indexable reaches the retrieval index, but only permissively licensed sources
# reach the training corpus. `nix_llama.rag.corpus` enforces the allowlist, so
# an unvetted licence fails this derivation instead of quietly shipping.
{
  lib,
  stdenvNoCC,
  nix-llama,
  nix-llama-sources,
  version,
}:

stdenvNoCC.mkDerivation {
  pname = "nix-llama-corpus";
  inherit version;

  dontUnpack = true;
  nativeBuildInputs = [ nix-llama ];

  buildPhase = ''
    runHook preBuild

    mkdir -p $out/rag $out/training

    cp ${nix-llama-sources.retrievalManifest} $out/rag/sources.json
    cp ${nix-llama-sources.trainingManifest} $out/training/sources.json

    echo "building retrieval index..."
    nix-llama-rag index \
      --manifest $out/rag/sources.json \
      --output $out/rag/index.json.gz \
      --attribution $out/ATTRIBUTION.md \
      --verbose

    echo "building training corpus..."
    nix-llama-rag corpus \
      --manifest $out/training/sources.json \
      --output $out/training/corpus.jsonl \
      --stats $out/training/stats.json

    nix-llama-rag --index $out/rag/index.json.gz stats > $out/rag/stats.txt

    runHook postBuild
  '';

  dontInstall = true;

  # The index is a deterministic function of the pinned sources, so it is safe
  # to mark it content-addressed-friendly: no timestamps, sorted traversal.
  preferLocalBuild = false;

  passthru = {
    inherit (nix-llama-sources) summary;
  };

  meta = {
    description = "nix-llama documentation corpus and retrieval index";
    longDescription = ''
      A BM25 index over permissively licensed documentation for Nix, NixOS,
      nix-darwin, home-manager, Hyprland, Niri, Kubernetes, CI/CD systems,
      Homebrew/macOS tooling and the Proton graphics stack. Every source is
      pinned to a revision and carries its licence; share-alike sources are
      indexed for retrieval but excluded from training material.
    '';
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
  };
}
