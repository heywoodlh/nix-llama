# A tiny corpus for the test fixtures.
#
# The real corpus fetches 14 documentation repositories; the smoke tests
# need a working index, not a complete one. This keeps `nix flake check` to
# minutes rather than gigabytes while still exercising the same code path --
# the same chunker, the same BM25 index, the same licence gate.
{
  lib,
  stdenvNoCC,
  nix-llama,
  writeText,
  version,
}:

let
  manifest = writeText "nix-llama-test-sources.json" (
    builtins.toJSON {
      sources = [
        {
          name = "fixtures";
          license = "MIT";
          # Interpolated, not `toString`: this copies the fixture directory
          # into the store as its own path, so the builder can actually read it.
          root = "${../../tests/fixtures/docs}";
          homepage = "https://github.com/heywoodlh/nix-llama";
          rev = "fixtures";
          include = [ "**/*.md" ];
          exclude = [ ];
          url_template = "";
          description = "nix-llama test fixtures";
        }
      ];
    }
  );
in
stdenvNoCC.mkDerivation {
  pname = "nix-llama-corpus-test";
  inherit version;

  dontUnpack = true;
  nativeBuildInputs = [ nix-llama ];

  buildPhase = ''
    runHook preBuild
    mkdir -p $out/rag $out/training
    cp ${manifest} $out/rag/sources.json
    cp ${manifest} $out/training/sources.json
    nix-llama-rag index \
      --manifest $out/rag/sources.json \
      --output $out/rag/index.json.gz \
      --attribution $out/ATTRIBUTION.md \
      --verbose
    nix-llama-rag corpus \
      --manifest $out/training/sources.json \
      --output $out/training/corpus.jsonl \
      --stats $out/training/stats.json
    nix-llama-rag --index $out/rag/index.json.gz stats > $out/rag/stats.txt
    runHook postBuild
  '';

  dontInstall = true;

  passthru.summary = [
    {
      name = "fixtures";
      license = "MIT";
      rev = "fixtures";
      usage = [
        "training"
        "retrieval"
      ];
      description = "nix-llama test fixtures";
    }
  ];

  meta = {
    description = "Minimal documentation corpus used by the nix-llama tests";
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
  };
}
