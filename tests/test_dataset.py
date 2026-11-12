"""Deriving fine-tuning records from the licensed corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nix_llama.dataset import build_dataset, write_jsonl
from nix_llama.nixopts import options_from_nix
from nix_llama.rag.corpus import LicenseError, Source

MODULE_NIX = """{ config, lib, ... }:
{
  options.services.widget = {
    enable = lib.mkEnableOption "the widget daemon";

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
      example = 9090;
      description = "Port the widget daemon listens on for incoming requests.";
    };

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        The widget package to use. Override this to run a patched build of the
        daemon instead of the one from nixpkgs.
      '';
    };

    undocumented = lib.mkOption {
      type = lib.types.str;
      default = "";
    };
  };
}
"""

GUIDE_MD = """---
title: Widget guide
---

# Widget guide

## How do I enable the widget daemon?

Set `services.widget.enable = true;` in your configuration and rebuild. The daemon
listens on port 8080 by default, which you can change with `services.widget.port`.
This paragraph exists to clear the minimum answer length that the dataset builder
enforces, because very short sections make poor training targets.

## Reverse proxy

```nix
services.nginx.virtualHosts."widget.example.com" = {
  locations."/".proxyPass = "http://127.0.0.1:8080";
  enableACME = true;
  forceSSL = true;
};
```

## Tiny

Too short.
"""


@pytest.fixture
def source(tmp_path: Path) -> Source:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.nix").write_text(MODULE_NIX, encoding="utf-8")
    (root / "guide.md").write_text(GUIDE_MD, encoding="utf-8")
    return Source(
        name="widget",
        license="MIT",
        root=str(root),
        include=["**/*.md", "**/*.nix"],
        description="the widget daemon",
        url_template="https://example.invalid/{path}",
    )


def test_options_are_extracted_with_their_full_path() -> None:
    options = {o["name"]: o for o in options_from_nix(MODULE_NIX)}
    # The leaf name alone would be useless; the path is what makes it usable.
    assert "services.widget.port" in options
    port = options["services.widget.port"]
    assert port["description"].startswith("Port the widget daemon")
    assert "port" in port["type"]
    assert port["default"] == "8080"
    assert port["example"] == "9090"


def test_mk_enable_option_is_understood() -> None:
    options = {o["name"]: o for o in options_from_nix(MODULE_NIX)}
    enable = options["services.widget.enable"]
    assert enable["description"] == "Whether to enable the widget daemon."
    assert enable["type"] == "boolean"
    # mkEnableOption takes a bare string, not an attribute set, so the scanner
    # must not adopt the *next* option's block as its body.
    assert "Port the widget daemon" not in enable["description"]


def test_multiline_nix_strings_are_collapsed() -> None:
    options = {o["name"]: o for o in options_from_nix(MODULE_NIX)}
    package = options["services.widget.package"]
    assert "\n" not in package["description"]
    assert "patched build" in package["description"]


def test_undocumented_options_are_skipped() -> None:
    assert "services.widget.undocumented" not in {o["name"] for o in options_from_nix(MODULE_NIX)}


def test_dataset_contains_text_option_and_snippet_records(source: Source) -> None:
    records, stats = build_dataset([source])
    origins = {r["origin"] for r in records}
    assert "document" in origins  # continued-pretraining prose
    assert "option" in origins  # from mkOption declarations
    assert "snippet" in origins  # from a documented code fence
    assert "heading" in origins  # from a question-shaped heading
    assert stats["records"] == len(records)
    assert stats["by_license"] == {"MIT": len(records)}


def test_chat_records_are_well_formed(source: Source) -> None:
    records, _ = build_dataset([source])
    for record in records:
        if record["kind"] != "chat":
            continue
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["user", "assistant"]
        assert record["messages"][0]["content"].strip()
        assert record["messages"][1]["content"].strip()
        assert record["url"].startswith("https://example.invalid/")


def test_front_matter_is_stripped(source: Source) -> None:
    records, _ = build_dataset([source])
    text = next(r for r in records if r["origin"] == "document")
    assert "title: Widget guide" not in text["text"]


def test_too_short_sections_are_dropped(source: Source) -> None:
    records, _ = build_dataset([source])
    assert not any(r.get("messages", [{}, {}])[1].get("content") == "Too short." for r in records)


def test_duplicates_are_dropped(tmp_path: Path) -> None:
    root = tmp_path / "dup"
    root.mkdir()
    for name in ("a.md", "b.md"):
        (root / name).write_text(GUIDE_MD, encoding="utf-8")
    _records, stats = build_dataset(
        [Source(name="dup", license="MIT", root=str(root), include=["**/*.md"])]
    )
    assert stats["duplicates_dropped"] > 0


def test_the_licence_gate_applies_to_training_data_too(tmp_path: Path) -> None:
    root = tmp_path / "gpl"
    root.mkdir()
    (root / "a.md").write_text(GUIDE_MD, encoding="utf-8")
    with pytest.raises(LicenseError):
        build_dataset([Source(name="gfdl", license="GFDL-1.3", root=str(root))])


def test_jsonl_output_is_one_object_per_line(source: Source, tmp_path: Path) -> None:
    records, _ = build_dataset([source])
    path = write_jsonl(records, tmp_path / "out" / "corpus.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records)
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_jsonl_output_is_deterministic(source: Source, tmp_path: Path) -> None:
    records, _ = build_dataset([source])
    first = write_jsonl(records, tmp_path / "a.jsonl").read_bytes()
    records2, _ = build_dataset([source])
    second = write_jsonl(records2, tmp_path / "b.jsonl").read_bytes()
    assert first == second
