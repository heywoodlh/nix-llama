# The weights registry.
#
# nix-llama's served model is a quantized GGUF. Which one depends on the
# variant, and on whether a fine-tune has been published yet:
#
#   * `base`    -- the upstream Apache-2.0 base model. Used directly by
#                  `.#model` until a nix-llama fine-tune exists, and used as the
#                  starting point by `.#model-trained`.
#   * `release` -- a published nix-llama fine-tune. CI fills this in after the
#                  training job pushes weights to Hugging Face; while it is
#                  null, `.#model` serves the base weights plus the nix-llama
#                  specialisation pack (retrieval index + system prompt).
#
# Only permissively licensed weights belong here. Qwen2.5-Coder 7B and 1.5B are
# Apache-2.0; the 3B is under the Qwen Research licence and is therefore not an
# option.
{
  variants = {
    default = {
      description = "Qwen2.5-Coder 7B Instruct, Q4_K_M";
      params = "7B";
      quantization = "Q4_K_M";
      license = "apache-2.0";
      hfRepo = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF";
      hfBaseRepo = "Qwen/Qwen2.5-Coder-7B-Instruct";
      file = "qwen2.5-coder-7b-instruct-q4_k_m.gguf";
      url = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
      hash = "sha256-UJKH94y01M9rOENzRzO5FLLBWOQ+Iqf0v16WOACJTTw=";
      # 4.68 GB: under the 5 GB budget, and leaves room for a 32k KV cache on
      # an 8 GiB accelerator once the cache is quantised.
      bytes = 4683073536;
    };

    small = {
      description = "Qwen2.5-Coder 1.5B Instruct, Q4_K_M";
      params = "1.5B";
      quantization = "Q4_K_M";
      license = "apache-2.0";
      hfRepo = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF";
      hfBaseRepo = "Qwen/Qwen2.5-Coder-1.5B-Instruct";
      file = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
      url = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
      hash = "sha256-zDJK8HDC7L/TJKMIhNL5Uaf/dWq6hcuBGm7ENpM7sEY=";
      bytes = 1117320768;
    };

    # 1 MB of nonsense weights with a working GGUF header and tokenizer. Used
    # by the integration tests so CI can exercise the whole serving path -- the
    # probe, the planner, llama-server, the retrieval proxy -- without pulling
    # gigabytes.
    test = {
      description = "tinyllamas stories260K (test fixture only)";
      params = "260K";
      quantization = "F32";
      license = "mit";
      hfRepo = "ggml-org/models";
      hfBaseRepo = null;
      file = "stories260K.gguf";
      url = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf";
      hash = "sha256-Jwy6G9UQn0LQM1D2BAYCRWBGTbFzwOOH2R8EJtO9JW0=";
      bytes = 1185376;
    };
  };

  # Set by CI once `.#model-trained` output is published. Shape:
  #   release.default = { url = "..."; hash = "sha256-..."; bytes = 0; version = "0.2.0"; };
  release = { };
}
