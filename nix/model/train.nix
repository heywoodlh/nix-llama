# The fine-tuning pipeline.
#
# Four derivations, so each expensive stage is cached independently:
#
#   base      fetch the Apache-2.0 base model (safetensors, tokenizer, config)
#   adapter   LoRA fine-tune on the permissively licensed corpus
#   merged    fold the adapter into the base weights
#   gguf      convert to GGUF and quantize
#
# The `adapter` stage is the one that wants a GPU. Nix's sandbox does not expose
# accelerators, so a real run needs either `--option sandbox relaxed` locally or
# a CI runner with `sandbox = relaxed` configured; see AGENTS.md. On CPU the
# pipeline is correct but slow -- fine for the 1.5B variant and the tests, not
# for a 7B production run.
{
  lib,
  stdenvNoCC,
  fetchurl,
  linkFarm,
  python3,
  llama-cpp,
  writeText,
  version,
  variant,
  corpus,
  # Hyper-parameters. Overridable: `.#model-trained.override { trainingConfig = ...; }`
  trainingConfig ? { },
  # Allow the training stage out of the sandbox so it can see /dev/nvidia*.
  # Requires the daemon to permit it.
  impureGpu ? false,
}:

let
  baseLock = (builtins.fromJSON (builtins.readFile ./base.lock.json)).bases or { };
  weights = import ./weights.nix;
  spec = weights.variants.${variant};

  entry =
    baseLock.${variant} or (throw ''
      nix-llama: no base model is pinned for the "${variant}" variant, so the
      fine-tuning pipeline cannot run reproducibly.

      Pin it once with:

          nix run .#update-base-model -- ${
            spec.hfBaseRepo or "Qwen/Qwen2.5-Coder-7B-Instruct"
          } --as ${variant}

      That downloads the base weights and records their hashes in
      nix/model/base.lock.json. `nix build .#model` does not need this --
      only the training pipeline does.
    '');

  defaults = {
    rank = 32;
    alpha = 64;
    dropout = 5.0e-2;
    learning-rate = 1.0e-4;
    epochs = 1;
    max-steps = -1;
    batch-size = 1;
    grad-accum = 16;
    seq-len = 2048;
    warmup-ratio = 3.0e-2;
    seed = 20260101;
  };

  configFile = writeText "nix-llama-training-config.json" (
    builtins.toJSON (defaults // trainingConfig)
  );

  # One link farm holding every pinned file, which is what transformers expects
  # a local model directory to look like.
  base = linkFarm "nix-llama-base-${variant}" (
    lib.mapAttrsToList (name: hash: {
      inherit name;
      path = fetchurl {
        url = "https://huggingface.co/${entry.repo}/resolve/${entry.revision}/${name}";
        inherit hash;
        name = "${lib.replaceStrings [ "/" ] [ "-" ] entry.repo}-${name}";
      };
    }) entry.files
  );

  pythonEnv = python3.withPackages (
    ps: with ps; [
      torch
      transformers
      peft
      accelerate
      safetensors
      datasets
      sentencepiece
      numpy
      # Needed by llama.cpp's convert_hf_to_gguf.py.
      gguf
      protobuf
    ]
  );

  adapter = stdenvNoCC.mkDerivation {
    pname = "nix-llama-adapter-${variant}";
    inherit version;
    dontUnpack = true;
    nativeBuildInputs = [ pythonEnv ];

    buildPhase = ''
      runHook preBuild
      export HOME=$TMPDIR
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      export TOKENIZERS_PARALLELISM=false
      mkdir -p $out
      python3 ${../../training/train_lora.py} \
        --base ${base} \
        --corpus ${corpus}/training/corpus.jsonl \
        --config ${configFile} \
        --output $out
      runHook postBuild
    '';

    dontInstall = true;

    # Fine-tuning is the single most expensive thing this repository does.
    requiredSystemFeatures = [ "big-parallel" ];
    __noChroot = impureGpu;

    meta.description = "LoRA adapter for nix-llama ${variant}";
  };

  merged = stdenvNoCC.mkDerivation {
    pname = "nix-llama-merged-${variant}";
    inherit version;
    dontUnpack = true;
    nativeBuildInputs = [ pythonEnv ];

    buildPhase = ''
      runHook preBuild
      export HOME=$TMPDIR
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      mkdir -p $out
      python3 ${../../training/merge_lora.py} \
        --base ${base} \
        --adapter ${adapter} \
        --output $out
      cp ${adapter}/training-summary.json $out/
      runHook postBuild
    '';

    dontInstall = true;
    requiredSystemFeatures = [ "big-parallel" ];
    meta.description = "nix-llama ${variant} with its adapter merged in";
  };

  ggufName = "nix-llama-${variant}-${version}-${lib.toLower spec.quantization}.gguf";
in
stdenvNoCC.mkDerivation {
  pname = "nix-llama-gguf-${variant}";
  inherit version;
  dontUnpack = true;

  nativeBuildInputs = [
    pythonEnv
    llama-cpp
  ];

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    mkdir -p $out

    echo "converting to GGUF (f16)..."
    python3 ${llama-cpp.src}/convert_hf_to_gguf.py \
      ${merged} \
      --outfile f16.gguf \
      --outtype f16

    echo "quantizing to ${spec.quantization}..."
    llama-quantize f16.gguf "$out/${ggufName}" ${spec.quantization} $NIX_BUILD_CORES

    cp ${merged}/training-summary.json $out/
    ls -lh "$out/${ggufName}"
    runHook postBuild
  '';

  dontInstall = true;
  requiredSystemFeatures = [ "big-parallel" ];

  passthru = {
    inherit
      base
      adapter
      merged
      ggufName
      configFile
      ;
  };

  meta = {
    description = "nix-llama ${variant} fine-tune, converted and quantized to ${spec.quantization}";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
}
