"""Turn the licensed corpus into fine-tuning records.

Two kinds of record come out of here:

``text``
    Continued-pretraining passages: documentation prose, verbatim. This is what
    teaches the base model the vocabulary and idioms of Nix, Hyprland, Niri,
    Kubernetes and the rest.

``chat``
    Instruction/response pairs derived *deterministically* from structured
    content -- ``mkOption`` declarations, question-shaped headings, and
    documented configuration snippets. No model is used to synthesise them, so
    the dataset is reproducible, attributable, and free of another model's
    output and of its licence terms.

The honest limitation: templated pairs teach recall and format, not reasoning.
They are a complement to the base model's instruction tuning, not a substitute
for a curated instruction set. See AGENTS.md for how to extend this.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .nixopts import options_from_nix
from .rag.corpus import Source
from .rag.index import clean_markup

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(?P<lang>[\w.+-]*)\s*$(?P<body>.*?)^```\s*$", re.DOTALL | re.MULTILINE)

# Headings that read as a question or a task, and so make a usable prompt.
_QUESTION_HEADING_RE = re.compile(
    r"^(how\b|why\b|what\b|when\b|where\b|can\b|should\b|configur|installing|setting up|"
    r"enabl|using|troubleshoot|migrat|upgrad|writing|creating|adding|debugg)",
    re.IGNORECASE,
)

MIN_ANSWER_CHARS = 120
MAX_ANSWER_CHARS = 4000
MIN_TEXT_CHARS = 200


def _option_records(source: Source, relative: str, text: str) -> Iterator[dict[str, Any]]:
    """Turn a Nix module's documented options into question/answer pairs."""
    namespace = {
        "home-manager": "home-manager",
        "nix-darwin": "nix-darwin",
        "nixos-manual": "NixOS",
        "nixpkgs-manual": "NixOS",
    }.get(source.name, source.name)

    for option in options_from_nix(text, prefix_hint=namespace):
        name = option["name"]
        # An option whose path could not be reconstructed teaches nothing:
        # "what does `enable` do?" has no answer worth learning.
        if "." not in name:
            continue
        answer = [option["description"]]
        if option.get("type"):
            answer.append(f"Type: `{option['type']}`.")
        if option.get("default"):
            answer.append(f"Default: `{option['default']}`.")
        if option.get("example"):
            answer.append(f"Example: `{option['example']}`.")
        yield {
            "kind": "chat",
            "messages": [
                {"role": "user", "content": f"What does the {namespace} option `{name}` do?"},
                {"role": "assistant", "content": " ".join(answer)},
            ],
            "source": source.name,
            "license": source.license,
            "url": source.file_url(relative),
            "origin": "option",
        }


def _sections(text: str) -> Iterator[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            yield match.group(2).strip(), body


def _doc_records(source: Source, relative: str, text: str, title: str) -> Iterator[dict[str, Any]]:
    topic = source.description or source.name
    for heading, body in _sections(text):
        if not (MIN_ANSWER_CHARS <= len(body) <= MAX_ANSWER_CHARS):
            continue
        if _QUESTION_HEADING_RE.match(heading):
            question = (
                heading
                if heading.endswith("?")
                else f"{heading} -- explain, in the context of {title}."
            )
            yield {
                "kind": "chat",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": body},
                ],
                "source": source.name,
                "license": source.license,
                "url": source.file_url(relative),
                "origin": "heading",
            }
            continue
        fence = _FENCE_RE.search(body)
        if fence and len(fence.group("body").strip()) > 60:
            lang = fence.group("lang") or ""
            yield {
                "kind": "chat",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Show a {topic} configuration example for {heading}.",
                    },
                    {
                        "role": "assistant",
                        "content": f"```{lang}\n{fence.group('body').strip()}\n```",
                    },
                ],
                "source": source.name,
                "license": source.license,
                "url": source.file_url(relative),
                "origin": "snippet",
            }


def records_for_source(source: Source) -> Iterator[dict[str, Any]]:
    """Yield every training record derived from one licensed source."""
    root = Path(source.root)
    for path in source.files():
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = clean_markup(raw)
        if not text:
            continue
        relative = str(path.relative_to(root)) if root.is_dir() else path.name
        title = f"{source.name}/{relative}"

        if path.suffix == ".nix":
            yield from _option_records(source, relative, text)
            continue

        if len(text) >= MIN_TEXT_CHARS:
            yield {
                "kind": "text",
                "text": text[:16000],
                "source": source.name,
                "license": source.license,
                "url": source.file_url(relative),
                "origin": "document",
            }
        yield from _doc_records(source, relative, text, title)


def build_dataset(sources: list[Source]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the full training dataset, enforcing the licence allowlist."""
    records: list[dict[str, Any]] = []
    for source in sources:
        source.validate()
        records.extend(records_for_source(source))

    # Deduplicate on content: documentation repeats itself across pages, and
    # duplicated examples are the fastest way to overfit a small model.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = (
            record["text"][:4000]
            if record["kind"] == "text"
            else record["messages"][1]["content"][:4000]
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    stats = {
        "records": len(unique),
        "duplicates_dropped": len(records) - len(unique),
        "by_kind": dict(Counter(r["kind"] for r in unique)),
        "by_origin": dict(Counter(r["origin"] for r in unique)),
        "by_source": dict(Counter(r["source"] for r in unique)),
        "by_license": dict(Counter(r["license"] for r in unique)),
        "text_characters": sum(len(r.get("text", "")) for r in unique),
    }
    return unique, stats


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return path
