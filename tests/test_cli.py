"""Command-line behaviour: model resolution, argument dispatch, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import build_gguf
from nix_llama.cli import (
    EXIT_NO_MODEL,
    ModelNotFoundError,
    build_parser,
    cmd_inspect,
    main,
    resolve_model,
)


def test_an_explicit_gguf_path_is_used(gguf_file: Path) -> None:
    assert resolve_model({"model": str(gguf_file)}) == gguf_file


def test_a_missing_model_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="does not exist"):
        resolve_model({"model": str(tmp_path / "absent.gguf")})


def test_no_model_configured_explains_how_to_configure_one() -> None:
    with pytest.raises(ModelNotFoundError, match="no model configured"):
        resolve_model({})


def test_a_model_directory_is_searched(tmp_path: Path) -> None:
    build_gguf(tmp_path / "model.gguf")
    assert resolve_model({"modelDir": str(tmp_path)}).name == "model.gguf"


def test_a_directory_passed_as_model_is_treated_as_a_model_directory(tmp_path: Path) -> None:
    build_gguf(tmp_path / "model.gguf")
    assert resolve_model({"model": str(tmp_path)}).name == "model.gguf"


def test_the_manifest_names_the_weights(tmp_path: Path) -> None:
    build_gguf(tmp_path / "weights.gguf")
    build_gguf(tmp_path / "decoy.gguf", padding=10_000)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"weights": "weights.gguf"}), encoding="utf-8"
    )
    # The manifest wins over "largest file", which is how a model package
    # declares which GGUF is the model rather than a projector or draft model.
    assert resolve_model({"modelDir": str(tmp_path)}).name == "weights.gguf"


def test_an_unreadable_manifest_falls_back_to_the_largest_gguf(tmp_path: Path) -> None:
    build_gguf(tmp_path / "small.gguf")
    build_gguf(tmp_path / "big.gguf", padding=100_000)
    (tmp_path / "manifest.json").write_text("{broken", encoding="utf-8")
    assert resolve_model({"modelDir": str(tmp_path)}).name == "big.gguf"


def test_an_empty_model_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match=r"no \.gguf file found"):
        resolve_model({"modelDir": str(tmp_path)})


def test_an_absent_subcommand_means_serve() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.port == 9000
    assert args.func.__name__ == "cmd_serve"


def test_config_before_a_subcommand_is_not_clobbered(tmp_path: Path) -> None:
    config = tmp_path / "c.json"
    config.write_text("{}", encoding="utf-8")
    args = build_parser().parse_args(["-c", str(config), "probe"])
    assert args.config == str(config)


def test_config_after_a_subcommand_also_works(tmp_path: Path) -> None:
    config = tmp_path / "c.json"
    config.write_text("{}", encoding="utf-8")
    args = build_parser().parse_args(["probe", "-c", str(config)])
    assert args.config == str(config)


def test_serving_without_a_model_exits_with_a_dedicated_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NIX_LLAMA_MODEL", raising=False)
    monkeypatch.delenv("NIX_LLAMA_MODEL_DIR", raising=False)
    monkeypatch.delenv("NIX_LLAMA_CONFIG", raising=False)
    assert main(["plan"]) == EXIT_NO_MODEL


def test_probe_emits_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu"]["available"] >= 1


def test_plan_json_includes_the_argv_and_the_reasoning(
    gguf_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["plan", "--model", str(gguf_file), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"][0].endswith("llama-server")
    assert payload["explain"]
    assert payload["probe"]["cpu"]["available"] >= 1


def test_inspect_prints_metadata(gguf_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(["inspect", str(gguf_file), "--format", "json"])
    assert cmd_inspect(args) == 0
    assert json.loads(capsys.readouterr().out)["architecture"] == "qwen2"


def test_inspect_enforces_a_size_budget(gguf_file: Path) -> None:
    args = build_parser().parse_args(["inspect", str(gguf_file), "--max-bytes", "10"])
    # The model package relies on this to keep the 5 GB promise true.
    assert cmd_inspect(args) == 1

    args = build_parser().parse_args(["inspect", str(gguf_file), "--max-bytes", "100000000"])
    assert cmd_inspect(args) == 0


def test_inspect_rejects_a_non_gguf_file(tmp_path: Path) -> None:
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"nope")
    args = build_parser().parse_args(["inspect", str(bad)])
    assert cmd_inspect(args) == 1


def test_dry_run_prints_the_command_without_starting_anything(
    gguf_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["serve", "--model", str(gguf_file), "--dry-run"]) == 0
    assert "llama-server" in capsys.readouterr().out
