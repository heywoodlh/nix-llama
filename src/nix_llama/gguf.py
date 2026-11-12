"""Minimal, dependency-free reader for GGUF metadata.

Only the header is parsed: we need a handful of hyper-parameters (layer count,
training context, KV-head geometry) to size a llama-server invocation, and
pulling in the full ``gguf`` Python package for that would drag a large closure
into every nix-llama install.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# GGUF value types.
(
    T_UINT8,
    T_INT8,
    T_UINT16,
    T_INT16,
    T_UINT32,
    T_INT32,
    T_FLOAT32,
    T_BOOL,
    T_STRING,
    T_ARRAY,
    T_UINT64,
    T_INT64,
    T_FLOAT64,
) = range(13)

_SCALAR = {
    T_UINT8: ("<B", 1),
    T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2),
    T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4),
    T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4),
    T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8),
    T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}

# Maximum number of array elements we materialise. Token lists in a vocabulary
# run to hundreds of thousands of entries and we never need them, so arrays are
# skipped past rather than decoded once they exceed this bound.
_MAX_ARRAY = 64


class GGUFError(Exception):
    """Raised when a file is not valid GGUF or uses an unknown encoding."""


@dataclass
class GGUFInfo:
    """The subset of GGUF metadata nix-llama reasons about."""

    path: Path
    architecture: str = "unknown"
    name: str = ""
    version: int = 0
    tensor_count: int = 0
    file_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def _arch_key(self, suffix: str) -> Any:
        return self.metadata.get(f"{self.architecture}.{suffix}")

    @property
    def n_ctx_train(self) -> int | None:
        value = self._arch_key("context_length")
        return int(value) if value is not None else None

    @property
    def n_layer(self) -> int | None:
        value = self._arch_key("block_count")
        return int(value) if value is not None else None

    @property
    def n_embd(self) -> int | None:
        value = self._arch_key("embedding_length")
        return int(value) if value is not None else None

    @property
    def n_head(self) -> int | None:
        value = self._arch_key("attention.head_count")
        return int(value) if value is not None else None

    @property
    def n_head_kv(self) -> int | None:
        value = self._arch_key("attention.head_count_kv")
        if value is None:
            return self.n_head
        if isinstance(value, (list, tuple)):
            # Per-layer head counts (hybrid models); use the largest.
            return int(max(value)) if value else None
        return int(value)

    @property
    def head_dim(self) -> int | None:
        explicit = self._arch_key("attention.key_length")
        if explicit is not None:
            return int(explicit)
        n_embd, n_head = self.n_embd, self.n_head
        if n_embd and n_head:
            return n_embd // n_head
        return None

    def kv_cache_bytes_per_token(self, bits_per_element: int = 16) -> int | None:
        """Bytes of KV cache consumed by a single token across all layers."""
        n_layer, n_head_kv, head_dim = self.n_layer, self.n_head_kv, self.head_dim
        if not (n_layer and n_head_kv and head_dim):
            return None
        # K and V, hence the factor of two.
        return 2 * n_layer * n_head_kv * head_dim * bits_per_element // 8

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "architecture": self.architecture,
            "gguf_version": self.version,
            "tensor_count": self.tensor_count,
            "file_size": self.file_size,
            "n_ctx_train": self.n_ctx_train,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_head_kv": self.n_head_kv,
            "head_dim": self.head_dim,
            "kv_bytes_per_token_f16": self.kv_cache_bytes_per_token(16),
        }


def _read(fh: BinaryIO, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file while reading GGUF header")
    return data


def _read_scalar(fh: BinaryIO, vtype: int) -> Any:
    try:
        fmt, size = _SCALAR[vtype]
    except KeyError:
        raise GGUFError(f"unsupported GGUF value type {vtype}") from None
    return struct.unpack(fmt, _read(fh, size))[0]


def _read_string(fh: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read(fh, 8))
    if length > (1 << 24):
        raise GGUFError("implausibly long string in GGUF header")
    return _read(fh, length).decode("utf-8", errors="replace")


def _read_value(fh: BinaryIO, vtype: int) -> Any:
    if vtype == T_STRING:
        return _read_string(fh)
    if vtype == T_ARRAY:
        (elem_type,) = struct.unpack("<I", _read(fh, 4))
        (count,) = struct.unpack("<Q", _read(fh, 8))
        if count > _MAX_ARRAY:
            _skip_array(fh, elem_type, count)
            return {"__skipped_array__": {"type": elem_type, "len": count}}
        return [_read_value(fh, elem_type) for _ in range(count)]
    return _read_scalar(fh, vtype)


def _skip_array(fh: BinaryIO, elem_type: int, count: int) -> None:
    if elem_type in _SCALAR:
        fh.seek(_SCALAR[elem_type][1] * count, 1)
        return
    if elem_type == T_STRING:
        for _ in range(count):
            (length,) = struct.unpack("<Q", _read(fh, 8))
            fh.seek(length, 1)
        return
    raise GGUFError(f"cannot skip GGUF array of type {elem_type}")


def read_gguf(path: str | Path) -> GGUFInfo:
    """Parse the metadata block of a GGUF file."""
    path = Path(path)
    info = GGUFInfo(path=path, file_size=path.stat().st_size)
    with path.open("rb") as fh:
        if _read(fh, 4) != GGUF_MAGIC:
            raise GGUFError(f"{path} is not a GGUF file (bad magic)")
        (info.version,) = struct.unpack("<I", _read(fh, 4))
        if info.version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {info.version}")
        (info.tensor_count,) = struct.unpack("<Q", _read(fh, 8))
        (kv_count,) = struct.unpack("<Q", _read(fh, 8))
        if kv_count > 4096:
            raise GGUFError("implausible metadata entry count")
        for _ in range(kv_count):
            key = _read_string(fh)
            (vtype,) = struct.unpack("<I", _read(fh, 4))
            info.metadata[key] = _read_value(fh, vtype)
    info.architecture = str(info.metadata.get("general.architecture", "unknown"))
    info.name = str(info.metadata.get("general.name", path.stem))
    return info
