#!/usr/bin/env python3
"""LoRA fine-tune a Qwen2.5-Coder base model on the nix-llama corpus.

Input is the JSONL produced by `nix-llama-rag corpus`: a mix of ``text``
records (documentation prose, for continued pretraining) and ``chat`` records
(instruction/response pairs derived from option declarations, question-shaped
headings and documented examples).

Only the assistant's tokens are trained on for chat records -- loss over the
prompt teaches the model to generate questions, which is not the job.

This is the one stage of the pipeline that wants a GPU. It runs on CPU, and the
tests exercise that path with a tiny model, but a real run on the 7B base model
on CPU is measured in days rather than hours.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("train-lora")

# Modules targeted by the adapter: all attention projections plus the MLP, which
# is the configuration that behaves best for domain adaptation on Qwen2 models.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="path to the base model directory")
    parser.add_argument("--corpus", required=True, help="corpus.jsonl from `nix-llama-rag corpus`")
    parser.add_argument("--output", required=True, help="where to write the LoRA adapter")
    parser.add_argument("--config", help="JSON file of hyper-parameters overriding the defaults")

    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 for a full epoch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--max-records", type=int, default=0, help="0 = use everything")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="auto picks bf16 on capable GPUs, fp32 on CPU",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    args = parser.parse_args(argv)

    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for key, value in overrides.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                raise SystemExit(f"unknown hyper-parameter {key!r} in {args.config}")
            setattr(args, attr, value)
    return args


def load_records(path: str, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    if not records:
        raise SystemExit(f"{path} contains no records")
    return records


def build_examples(records: list[dict[str, Any]], tokenizer: Any, seq_len: int) -> list[dict]:
    """Tokenise records into ``input_ids``/``labels`` pairs.

    Prompt tokens get a label of -100 so the loss only covers the response (or,
    for continued-pretraining text, the whole sequence).
    """
    examples: list[dict[str, list[int]]] = []
    skipped = 0
    for record in records:
        if record.get("kind") == "text":
            ids = tokenizer(
                record["text"], truncation=True, max_length=seq_len, add_special_tokens=True
            )["input_ids"]
            if len(ids) < 32:
                skipped += 1
                continue
            examples.append({"input_ids": ids, "labels": list(ids)})
            continue

        messages = record.get("messages") or []
        if len(messages) < 2:
            skipped += 1
            continue
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        answer = messages[-1]["content"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos = tokenizer.eos_token_id
        if eos is not None:
            answer_ids = [*answer_ids, eos]
        ids = (prompt_ids + answer_ids)[:seq_len]
        if len(ids) <= len(prompt_ids):
            # The prompt alone filled the window; there is nothing to learn.
            skipped += 1
            continue
        labels = [-100] * min(len(prompt_ids), len(ids))
        labels += ids[len(labels) :]
        examples.append({"input_ids": ids, "labels": labels})

    if skipped:
        log.info("skipped %d unusable record(s)", skipped)
    if not examples:
        raise SystemExit("no trainable examples survived tokenisation")
    return examples


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def pick_dtype(requested: str) -> Any:
    import torch

    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    # Reproducibility: the pipeline is a Nix derivation, so a rerun must give
    # the same adapter.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dtype = pick_dtype(args.precision)
    log.info("base=%s dtype=%s cuda=%s", args.base, dtype, torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(args.corpus, args.max_records)
    log.info("loaded %d corpus records", len(records))
    examples = build_examples(records, tokenizer, args.seq_len)
    # Seeded for reproducibility, not secrecy: the same corpus and the same
    # seed have to produce the same adapter.
    random.Random(args.seed).shuffle(examples)  # noqa: S311
    log.info("prepared %d training examples", len(examples))

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("training %d of %d parameters (%.3f%%)", trainable, total, 100 * trainable / total)

    steps_per_epoch = max(1, math.ceil(len(examples) / (args.batch_size * args.grad_accum)))
    log.info("~%d optimiser steps per epoch", steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output) / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    result = trainer.train()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "training-summary.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "examples": len(examples),
                "records": len(records),
                "steps": result.global_step,
                "train_loss": result.training_loss,
                "hyperparameters": {
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "seq_len": args.seq_len,
                    "seed": args.seed,
                    "target_modules": DEFAULT_TARGET_MODULES,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("wrote adapter to %s (final loss %.4f)", out, result.training_loss or float("nan"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
