"""Hardware probing, against synthetic /proc and /sys trees."""

from __future__ import annotations

from pathlib import Path

import pytest

from nix_llama import probe as probe_mod
from nix_llama.probe import (
    Accelerator,
    Probe,
    _Fs,
    choose_backend,
    detect_amd,
    detect_container,
    detect_cpu_linux,
    detect_intel,
    detect_memory_linux,
    detect_nvidia,
    detect_storage,
    list_llama_devices,
    probe,
    vram_for,
)


@pytest.fixture(autouse=True)
def _no_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests off the host's real nvidia-smi/sysctl."""
    monkeypatch.setattr(probe_mod, "_run", lambda *a, **k: None)


def test_container_detection_prefers_explicit_markers(fakeroot: Path) -> None:
    assert detect_container(_Fs(str(fakeroot))) == (False, None)
    (fakeroot / ".dockerenv").write_text("", encoding="utf-8")
    assert detect_container(_Fs(str(fakeroot))) == (True, "docker")


def test_container_detection_reads_cgroups(fakeroot: Path) -> None:
    (fakeroot / "proc/1/cgroup").write_text(
        "0::/kubepods/besteffort/pod123/abc\n", encoding="utf-8"
    )
    assert detect_container(_Fs(str(fakeroot))) == (True, "kubernetes")


def test_container_detection_uses_the_env_marker(
    fakeroot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("container", "systemd-nspawn")
    assert detect_container(_Fs(str(fakeroot))) == (True, "systemd-nspawn")


def test_cpu_topology_counts_physical_cores(fakeroot: Path) -> None:
    cpu = detect_cpu_linux(_Fs(str(fakeroot)))
    assert cpu.physical == 2  # two distinct (package, core) pairs
    assert "avx2" in cpu.isa
    assert cpu.model.startswith("AMD Ryzen")
    # Never more threads than physical cores: hyper-threads do not help here.
    assert cpu.available <= cpu.physical


def test_cgroup_cpu_quota_caps_the_thread_count(nvidia_root: Path) -> None:
    # cpu.max of "400000 100000" is a 4-CPU quota, but the fixture only has
    # 2 physical cores, so the physical cap still wins.
    cpu = detect_cpu_linux(_Fs(str(nvidia_root)))
    assert cpu.available == 2

    (nvidia_root / "sys/fs/cgroup/cpu.max").write_text("100000 100000\n", encoding="utf-8")
    assert detect_cpu_linux(_Fs(str(nvidia_root))).available == 1


def test_memory_reads_meminfo(fakeroot: Path) -> None:
    mem = detect_memory_linux(_Fs(str(fakeroot)))
    assert mem.total_bytes == 32817356 * 1024
    assert mem.available_bytes == 24000000 * 1024
    assert mem.limit_bytes is None
    assert mem.budget_bytes == mem.total_bytes


def test_cgroup_memory_limit_wins_over_total(nvidia_root: Path) -> None:
    mem = detect_memory_linux(_Fs(str(nvidia_root)))
    assert mem.limit_bytes == 8 * 1024**3
    # A container must plan against its limit, not the host's RAM.
    assert mem.budget_bytes == 8 * 1024**3


def test_nvidia_detected_from_device_nodes_in_a_container(nvidia_root: Path) -> None:
    found = detect_nvidia(_Fs(str(nvidia_root)))
    assert [a.backend for a in found] == ["cuda"]
    assert found[0].source == "devnode"


def test_amd_detected_with_vram_and_rocm(amd_root: Path) -> None:
    found = detect_amd(_Fs(str(amd_root)))
    assert len(found) == 1
    assert found[0].backend == "rocm"  # /dev/kfd present
    assert found[0].memory_bytes == 16 * 1024**3


def test_amd_without_kfd_falls_back_to_vulkan(amd_root: Path) -> None:
    (amd_root / "dev/kfd").unlink()
    assert detect_amd(_Fs(str(amd_root)))[0].backend == "vulkan"


def test_intel_igpu_maps_to_vulkan(fakeroot: Path) -> None:
    (fakeroot / "sys/class/drm/card0/device").mkdir(parents=True)
    (fakeroot / "sys/class/drm/card0/device/vendor").write_text("0x8086\n", encoding="utf-8")
    found = detect_intel(_Fs(str(fakeroot)))
    assert [a.backend for a in found] == ["vulkan"]


def test_storage_reports_flash_from_rotational_flag(tmp_path: Path, fakeroot: Path) -> None:
    (fakeroot / "sys/block/nvme0n1/queue").mkdir(parents=True)
    (fakeroot / "sys/block/nvme0n1/queue/rotational").write_text("0\n", encoding="utf-8")
    info = detect_storage(_Fs(str(fakeroot)), str(tmp_path))
    assert info.is_ssd is True
    assert info.free_bytes > 0


def test_storage_treats_a_spinning_disk_as_not_flash(tmp_path: Path, fakeroot: Path) -> None:
    (fakeroot / "sys/block/sda/queue").mkdir(parents=True)
    (fakeroot / "sys/block/sda/queue/rotational").write_text("1\n", encoding="utf-8")
    assert detect_storage(_Fs(str(fakeroot)), str(tmp_path)).is_ssd is False


def test_storage_unknown_is_not_treated_as_flash(tmp_path: Path, fakeroot: Path) -> None:
    # No /sys/block entries at all: we must not guess flash and then spill a
    # KV cache onto something slow.
    info = detect_storage(_Fs(str(fakeroot)), str(tmp_path))
    assert info.rotational is None
    assert info.is_ssd is False


def test_storage_handles_a_path_that_does_not_exist_yet(tmp_path: Path, fakeroot: Path) -> None:
    info = detect_storage(_Fs(str(fakeroot)), str(tmp_path / "not" / "created" / "yet"))
    assert info.total_bytes > 0


def test_list_devices_parses_llama_server_output(monkeypatch: pytest.MonkeyPatch) -> None:
    output = (
        "Available devices:\n"
        "  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 23950 MiB free)\n"
        "  Vulkan0: Intel(R) Arc(tm) A770 (16384 MiB, 16000 MiB free)\n"
        "  BLAS: BLAS (0 MiB, 0 MiB free)\n"
    )
    monkeypatch.setattr(probe_mod, "_run", lambda *a, **k: output)
    devices = list_llama_devices("/usr/bin/llama-server")
    assert [d["backend"] for d in devices] == ["cuda0", "vulkan0", "blas"]
    assert devices[0]["memory_bytes"] == 24564 * 1024 * 1024


def test_list_devices_without_a_binary_is_empty() -> None:
    assert list_llama_devices(None) == []


def test_backend_choice_requires_the_binary_to_agree() -> None:
    facts = Probe(
        accelerators=[Accelerator(vendor="nvidia", backend="cuda", name="GPU")],
        devices=[{"backend": "blas", "name": "BLAS", "memory_bytes": None}],
    )
    # Hardware is present but this llama.cpp build cannot address it.
    assert choose_backend(facts) == "cpu"
    assert facts.warnings and "falling back to CPU" in facts.warnings[0]


def test_backend_choice_uses_a_reported_device() -> None:
    facts = Probe(
        accelerators=[Accelerator(vendor="nvidia", backend="cuda", name="GPU")],
        devices=[{"backend": "cuda", "name": "GPU", "memory_bytes": 8 * 1024**3}],
    )
    assert choose_backend(facts) == "cuda"
    assert facts.warnings == []
    assert vram_for(facts, "cuda") == 8 * 1024**3


def test_forced_backend_overrides_detection() -> None:
    facts = Probe(devices=[{"backend": "cuda", "name": "GPU", "memory_bytes": 1}])
    assert choose_backend(facts, "vulkan") == "vulkan"
    assert choose_backend(facts, "none") == "cpu"


def test_vram_sums_multiple_devices() -> None:
    facts = Probe(
        devices=[
            {"backend": "cuda", "name": "a", "memory_bytes": 8 * 1024**3},
            {"backend": "cuda", "name": "b", "memory_bytes": 8 * 1024**3},
        ]
    )
    assert vram_for(facts, "cuda") == 16 * 1024**3
    assert vram_for(facts, "cpu") is None


def test_probe_end_to_end_is_serialisable(fakeroot: Path, tmp_path: Path) -> None:
    facts = probe(root=str(fakeroot), state_dir=str(tmp_path))
    data = facts.as_dict()
    assert "root" not in data  # never leak the test root into output
    assert data["memory"]["budget_bytes"] > 0
    assert data["cpu"]["available"] >= 1
    assert "is_ssd" in data["storage"]
    assert facts.to_json().startswith("{")
