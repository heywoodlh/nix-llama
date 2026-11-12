"""Shared fixtures: synthetic /proc and /sys trees, and a synthetic GGUF file."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

CPUINFO_RYZEN = """processor\t: 0
model name\t: AMD Ryzen 9 5900X 12-Core Processor
flags\t\t: fpu vme avx avx2 f16c fma sse4_2
physical id\t: 0
core id\t\t: 0

processor\t: 1
model name\t: AMD Ryzen 9 5900X 12-Core Processor
flags\t\t: fpu vme avx avx2 f16c fma sse4_2
physical id\t: 0
core id\t\t: 0

processor\t: 2
model name\t: AMD Ryzen 9 5900X 12-Core Processor
flags\t\t: fpu vme avx avx2 f16c fma sse4_2
physical id\t: 0
core id\t\t: 1

processor\t: 3
model name\t: AMD Ryzen 9 5900X 12-Core Processor
flags\t\t: fpu vme avx avx2 f16c fma sse4_2
physical id\t: 0
core id\t\t: 1
"""

MEMINFO_32G = """MemTotal:       32817356 kB
MemFree:         2000000 kB
MemAvailable:   24000000 kB
"""


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def fakeroot(tmp_path: Path) -> Path:
    """A minimal Linux-looking filesystem root."""
    root = tmp_path / "root"
    write(root / "proc/cpuinfo", CPUINFO_RYZEN)
    write(root / "proc/meminfo", MEMINFO_32G)
    write(root / "proc/1/cgroup", "0::/init.scope\n")
    write(root / "proc/1/sched", "systemd (1, #threads: 1)\n")
    write(root / "sys/fs/cgroup/cpu.max", "max 100000\n")
    write(root / "sys/fs/cgroup/memory.max", "max\n")
    return root


@pytest.fixture
def nvidia_root(fakeroot: Path) -> Path:
    """A root that looks like a container with an NVIDIA device passed through."""
    write(fakeroot / ".dockerenv", "")
    write(fakeroot / "proc/driver/nvidia/version", "NVRM version: 550.54.14\n")
    (fakeroot / "dev").mkdir(parents=True, exist_ok=True)
    (fakeroot / "dev/nvidia0").write_text("", encoding="utf-8")
    write(fakeroot / "sys/fs/cgroup/memory.max", "8589934592\n")
    write(fakeroot / "sys/fs/cgroup/cpu.max", "400000 100000\n")
    return fakeroot


@pytest.fixture
def amd_root(fakeroot: Path) -> Path:
    """A root with a discrete AMD GPU and the ROCm compute node present."""
    write(fakeroot / "sys/class/drm/card0/device/vendor", "0x1002\n")
    write(
        fakeroot / "sys/class/drm/card0/device/mem_info_vram_total",
        f"{16 * 1024**3}\n",
    )
    (fakeroot / "dev").mkdir(parents=True, exist_ok=True)
    (fakeroot / "dev/kfd").write_text("", encoding="utf-8")
    return fakeroot


def build_gguf(
    path: Path,
    *,
    architecture: str = "qwen2",
    n_layer: int = 28,
    n_embd: int = 1536,
    n_head: int = 12,
    n_head_kv: int = 2,
    n_ctx_train: int = 32768,
    name: str = "test model",
    padding: int = 0,
) -> Path:
    """Write a GGUF file with a real header and no tensor data."""
    T_UINT32, T_STRING = 4, 8

    def gguf_string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    entries: list[tuple[str, int, bytes]] = [
        ("general.architecture", T_STRING, gguf_string(architecture)),
        ("general.name", T_STRING, gguf_string(name)),
        (f"{architecture}.block_count", T_UINT32, struct.pack("<I", n_layer)),
        (f"{architecture}.embedding_length", T_UINT32, struct.pack("<I", n_embd)),
        (f"{architecture}.attention.head_count", T_UINT32, struct.pack("<I", n_head)),
        (f"{architecture}.attention.head_count_kv", T_UINT32, struct.pack("<I", n_head_kv)),
        (f"{architecture}.context_length", T_UINT32, struct.pack("<I", n_ctx_train)),
    ]

    body = b"".join(
        gguf_string(key) + struct.pack("<I", vtype) + value for key, vtype, value in entries
    )
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + b"\0" * padding)
    return path


@pytest.fixture
def gguf_file(tmp_path: Path) -> Path:
    return build_gguf(tmp_path / "model.gguf")
