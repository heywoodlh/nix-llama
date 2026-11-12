#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model.

llama.cpp can load an adapter at runtime with `--lora`, but merging first means
the published artifact is a single self-contained GGUF that quantizes cleanly
and needs no extra flags to serve.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger("merge-lora")

# Tokenizer and config files the GGUF converter needs alongside the weights.
SIDECAR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading base model %s", args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    log.info("applying adapter %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)

    try:
        AutoTokenizer.from_pretrained(args.adapter).save_pretrained(str(out))
    except OSError:
        AutoTokenizer.from_pretrained(args.base).save_pretrained(str(out))

    base = Path(args.base)
    for name in SIDECAR_FILES:
        source = base / name
        if source.is_file() and not (out / name).exists():
            shutil.copy(source, out / name)

    log.info("merged model written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
