"""Configuration loading and merging.

A nix-llama configuration is a plain nested dict. The Nix modules render one as
JSON; humans can also hand-write one. Precedence, lowest to highest:

1. built-in defaults (:data:`DEFAULTS`)
2. the config file (``--config`` / ``NIX_LLAMA_CONFIG``)
3. ``NIX_LLAMA_*`` environment variables
4. command-line flags

Within the server plan itself a second precedence order applies, documented in
:mod:`nix_llama.defaults`: ``extraFlags`` beats ``settings`` beats ``tuning``
beats anything the probe inferred.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "model": None,
    "modelDir": None,
    "host": "127.0.0.1",
    "port": 8080,
    "alias": "nix-llama",
    "acceleration": "auto",
    "apiKeyFile": None,
    "stateDir": None,
    "llamaServer": "llama-server",
    "systemPrompt": None,
    "tuning": {
        "contextSize": None,
        "maxContextSize": 32768,
        "minContextSize": 4096,
        "gpuLayers": None,
        "threads": None,
        "threadsBatch": None,
        "batchSize": None,
        "ubatchSize": None,
        "parallel": None,
        "kvCacheType": "auto",
        "promptCacheMiB": None,
        "speculative": "auto",
        "contextShift": True,
        "loadMode": None,
        "flashAttention": "auto",
        "corsOrigins": "localhost",
        "metrics": True,
        "webui": True,
    },
    "diskCache": {
        "enable": True,
        "path": None,
        "minFreeGiB": 8,
    },
    "rag": {
        "enable": True,
        "index": None,
        "sources": [],
        "topK": 4,
        "maxChars": 6000,
        "minScore": 1.0,
        "upstreamPort": None,
        "injectInto": "system",
    },
    "tls": {
        "certFile": None,
        "keyFile": None,
    },
    "settings": {},
    "extraFlags": [],
}

_ENV_MAP = {
    "NIX_LLAMA_MODEL": ("model",),
    "NIX_LLAMA_MODEL_DIR": ("modelDir",),
    "NIX_LLAMA_HOST": ("host",),
    "NIX_LLAMA_PORT": ("port",),
    "NIX_LLAMA_ALIAS": ("alias",),
    "NIX_LLAMA_ACCELERATION": ("acceleration",),
    "NIX_LLAMA_API_KEY_FILE": ("apiKeyFile",),
    "NIX_LLAMA_STATE_DIR": ("stateDir",),
    "NIX_LLAMA_LLAMA_SERVER": ("llamaServer",),
    "NIX_LLAMA_CONTEXT_SIZE": ("tuning", "contextSize"),
    "NIX_LLAMA_MAX_CONTEXT_SIZE": ("tuning", "maxContextSize"),
    "NIX_LLAMA_GPU_LAYERS": ("tuning", "gpuLayers"),
    "NIX_LLAMA_THREADS": ("tuning", "threads"),
    "NIX_LLAMA_PARALLEL": ("tuning", "parallel"),
    "NIX_LLAMA_KV_CACHE_TYPE": ("tuning", "kvCacheType"),
    "NIX_LLAMA_SPECULATIVE": ("tuning", "speculative"),
    "NIX_LLAMA_DISK_CACHE": ("diskCache", "enable"),
    "NIX_LLAMA_RAG": ("rag", "enable"),
    "NIX_LLAMA_RAG_INDEX": ("rag", "index"),
    "NIX_LLAMA_RAG_TOP_K": ("rag", "topK"),
}

_INT_KEYS = {
    "port",
    "contextSize",
    "maxContextSize",
    "minContextSize",
    "gpuLayers",
    "threads",
    "threadsBatch",
    "batchSize",
    "ubatchSize",
    "parallel",
    "promptCacheMiB",
    "topK",
    "maxChars",
    "upstreamPort",
    "minFreeGiB",
}
_BOOL_KEYS = {"enable", "contextShift", "metrics", "webui"}


# Where to look for a configuration file when none was named. The service
# modules install their rendered config at the system path, so `nix-llama plan`
# and `nix-llama doctor` typed into a shell on a machine running the service
# describe *that* service rather than reporting no model configured.
SYSTEM_CONFIG = "/etc/nix-llama/config.json"
USER_CONFIG = "nix-llama/config.json"


class ConfigError(Exception):
    """Raised on a malformed configuration."""


def discover(env: dict[str, str] | None = None) -> Path | None:
    """Find a configuration file to use when the caller named none."""
    env = dict(os.environ if env is None else env)

    explicit = env.get("NIX_LLAMA_CONFIG")
    if explicit:
        return Path(explicit)

    xdg = env.get("XDG_CONFIG_HOME")
    candidates = []
    if xdg:
        candidates.append(Path(xdg) / USER_CONFIG)
    home = env.get("HOME")
    if home:
        candidates.append(Path(home) / ".config" / USER_CONFIG)
    candidates.append(Path(SYSTEM_CONFIG))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(key: str, raw: str) -> Any:
    if key in _BOOL_KEYS:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if key in _INT_KEYS:
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"expected an integer for {key}, got {raw!r}") from None
    return raw


def from_env(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    out: dict[str, Any] = {}
    for name, path in _ENV_MAP.items():
        if name not in env or env[name] == "":
            continue
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(path[-1], env[name])
    return out


def load_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def validate(config: dict[str, Any]) -> None:
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    for section in ("tuning", "diskCache", "rag", "tls"):
        extra = set(config.get(section, {})) - set(DEFAULTS[section])
        if extra:
            raise ConfigError(f"unknown keys in {section}: {', '.join(sorted(extra))}")
    if not isinstance(config.get("extraFlags", []), list):
        raise ConfigError("extraFlags must be a list of strings")
    if not isinstance(config.get("settings", {}), dict):
        raise ConfigError("settings must be an object mapping flags to values")
    port = config.get("port")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError(f"port must be between 1 and 65535, got {port!r}")


def load(
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the effective configuration."""
    config = copy.deepcopy(DEFAULTS)
    chosen = config_file or discover(env)
    if chosen:
        config = deep_merge(config, load_file(chosen))
    config = deep_merge(config, from_env(env))
    if overrides:
        config = deep_merge(config, {k: v for k, v in overrides.items() if v is not None})
    validate(config)
    return config
