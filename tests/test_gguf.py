"""The GGUF header reader."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from conftest import build_gguf
from nix_llama.gguf import GGUFError, read_gguf


def test_reads_architecture_and_geometry(gguf_file: Path) -> None:
    info = read_gguf(gguf_file)
    assert info.architecture == "qwen2"
    assert info.n_layer == 28
    assert info.n_ctx_train == 32768
    assert info.n_head_kv == 2
    assert info.head_dim == 128  # 1536 / 12


def test_kv_cache_sizing_matches_hand_calculation(gguf_file: Path) -> None:
    info = read_gguf(gguf_file)
    # 2 (K and V) * 28 layers * 2 kv heads * 128 dims * 2 bytes
    assert info.kv_cache_bytes_per_token(16) == 2 * 28 * 2 * 128 * 2
    # Quantising to q8_0 is budgeted at 9 bits per element.
    assert info.kv_cache_bytes_per_token(9) == 2 * 28 * 2 * 128 * 9 // 8


def test_head_dim_falls_back_to_embedding_over_heads(tmp_path: Path) -> None:
    info = read_gguf(build_gguf(tmp_path / "m.gguf", n_embd=4096, n_head=32))
    assert info.head_dim == 128


def test_missing_kv_head_count_falls_back_to_head_count(tmp_path: Path) -> None:
    path = build_gguf(tmp_path / "m.gguf", n_head=8, n_head_kv=8)
    assert read_gguf(path).n_head_kv == 8


def test_rejects_non_gguf(tmp_path: Path) -> None:
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GGUFError, match="not a GGUF file"):
        read_gguf(bad)


def test_rejects_unsupported_version(tmp_path: Path) -> None:
    bad = tmp_path / "v9.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<I", 9) + struct.pack("<Q", 0) + struct.pack("<Q", 0))
    with pytest.raises(GGUFError, match="unsupported GGUF version"):
        read_gguf(bad)


def test_truncated_header_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "short.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0))
    with pytest.raises(GGUFError):
        read_gguf(bad)


def test_file_size_is_reported(tmp_path: Path) -> None:
    path = build_gguf(tmp_path / "m.gguf", padding=4096)
    info = read_gguf(path)
    assert info.file_size == path.stat().st_size
    assert info.as_dict()["file_size"] == info.file_size
