"""Configuration loading, merging and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nix_llama.config import (
    DEFAULTS,
    SYSTEM_CONFIG,
    ConfigError,
    deep_merge,
    discover,
    from_env,
    load,
    validate,
)


def test_defaults_are_conservative() -> None:
    assert DEFAULTS["host"] == "127.0.0.1"
    assert DEFAULTS["tuning"]["corsOrigins"] == "localhost"
    assert DEFAULTS["rag"]["enable"] is True
    assert DEFAULTS["apiKeyFile"] is None


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_file_then_env_then_cli(tmp_path: Path) -> None:
    config_file = tmp_path / "nix-llama.json"
    config_file.write_text(json.dumps({"port": 9000, "tuning": {"threads": 4}}), encoding="utf-8")

    config = load(
        config_file=config_file,
        env={"NIX_LLAMA_PORT": "9100"},
        overrides={"port": 9200},
    )
    # Each layer beats the one before it.
    assert config["port"] == 9200
    # Untouched keys survive from the file.
    assert config["tuning"]["threads"] == 4
    # And untouched sections keep their defaults.
    assert config["tuning"]["corsOrigins"] == "localhost"


def test_env_is_read_from_the_config_pointer(tmp_path: Path) -> None:
    config_file = tmp_path / "c.json"
    config_file.write_text(json.dumps({"alias": "from-file"}), encoding="utf-8")
    config = load(env={"NIX_LLAMA_CONFIG": str(config_file)})
    assert config["alias"] == "from-file"


def test_env_coercion() -> None:
    parsed = from_env(
        {
            "NIX_LLAMA_PORT": "8081",
            "NIX_LLAMA_RAG": "false",
            "NIX_LLAMA_RAG_TOP_K": "9",
            "NIX_LLAMA_HOST": "0.0.0.0",
        }
    )
    assert parsed == {"port": 8081, "rag": {"enable": False, "topK": 9}, "host": "0.0.0.0"}


def test_env_booleans_accept_the_usual_spellings() -> None:
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert from_env({"NIX_LLAMA_DISK_CACHE": truthy})["diskCache"]["enable"] is True
    for falsy in ("0", "false", "no", "off"):
        assert from_env({"NIX_LLAMA_DISK_CACHE": falsy})["diskCache"]["enable"] is False


def test_empty_env_values_are_ignored() -> None:
    assert from_env({"NIX_LLAMA_HOST": ""}) == {}


def test_bad_integer_env_is_reported_clearly() -> None:
    with pytest.raises(ConfigError, match="expected an integer"):
        from_env({"NIX_LLAMA_PORT": "not-a-port"})


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown configuration keys: nonsense"):
        validate({**DEFAULTS, "nonsense": 1})


def test_unknown_section_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown keys in tuning: speculatve"):
        validate({**DEFAULTS, "tuning": {"speculatve": "ngram-mod"}})


def test_port_range_is_validated() -> None:
    with pytest.raises(ConfigError, match="port must be between"):
        validate({**DEFAULTS, "port": 70000})


def test_extra_flags_must_be_a_list() -> None:
    with pytest.raises(ConfigError, match="extraFlags must be a list"):
        validate({**DEFAULTS, "extraFlags": "--props"})


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load(config_file=tmp_path / "absent.json")


def test_malformed_config_file_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load(config_file=bad)


def test_non_object_config_file_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a JSON object"):
        load(config_file=bad)


def test_none_overrides_are_dropped() -> None:
    # The CLI passes None for every flag the user did not give.
    config = load(overrides={"host": None, "port": 9999})
    assert config["host"] == DEFAULTS["host"]
    assert config["port"] == 9999


def test_discovery_prefers_the_environment_pointer(tmp_path: Path) -> None:
    named = tmp_path / "named.json"
    named.write_text("{}", encoding="utf-8")
    assert discover({"NIX_LLAMA_CONFIG": str(named)}) == named


def test_discovery_falls_back_to_the_user_config(tmp_path: Path) -> None:
    user = tmp_path / "nix-llama" / "config.json"
    user.parent.mkdir(parents=True)
    user.write_text("{}", encoding="utf-8")
    assert discover({"XDG_CONFIG_HOME": str(tmp_path)}) == user


def test_discovery_falls_back_to_dot_config(tmp_path: Path) -> None:
    user = tmp_path / ".config" / "nix-llama" / "config.json"
    user.parent.mkdir(parents=True)
    user.write_text("{}", encoding="utf-8")
    assert discover({"HOME": str(tmp_path)}) == user


def test_discovery_returns_none_when_there_is_nothing(tmp_path: Path) -> None:
    # Only /etc/nix-llama/config.json could still match, and it does not exist
    # on a machine that is not running the service.
    found = discover({"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "absent")})
    assert found is None or found == Path(SYSTEM_CONFIG)


def test_the_service_config_path_is_the_documented_one() -> None:
    # The NixOS and nix-darwin modules install their rendered config here, so
    # `nix-llama plan` in a shell describes the running service. Changing this
    # string without changing the modules breaks that.
    assert SYSTEM_CONFIG == "/etc/nix-llama/config.json"
