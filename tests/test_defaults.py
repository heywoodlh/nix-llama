"""The planner: facts in, llama-server argv out."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import build_gguf
from nix_llama.config import load
from nix_llama.defaults import GiB, MiB, memory_budget, plan, plan_context
from nix_llama.gguf import read_gguf
from nix_llama.probe import Accelerator, CPUInfo, MemoryInfo, Probe, StorageInfo


def facts(**kwargs) -> Probe:
    """A plausible 32 GiB CPU-only Linux box, overridable per test."""
    base = {
        "os": "linux",
        "arch": "x86_64",
        "cpu": CPUInfo(logical=24, physical=12, available=12, model="test"),
        "memory": MemoryInfo(total_bytes=32 * GiB, available_bytes=24 * GiB),
        "storage": StorageInfo(path="/var/lib/nix-llama", rotational=False, free_bytes=500 * GiB),
        "acceleration": "cpu",
    }
    base.update(kwargs)
    return Probe(**base)


def argv_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    return argv[index + 1] if index + 1 < len(argv) else ""


def test_cpu_plan_sets_threads_from_physical_cores(gguf_file: Path) -> None:
    result = plan(load(), facts(), read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--threads") == "12"
    assert argv_value(result.argv, "--threads-batch") == "12"
    # No accelerator, so no offload flags at all.
    assert "--gpu-layers" not in result.argv
    assert "--fit" not in result.argv


def test_context_is_capped_by_the_models_trained_length(tmp_path: Path) -> None:
    gguf = read_gguf(build_gguf(tmp_path / "m.gguf", n_ctx_train=8192))
    result = plan(load(), facts(), gguf, model_path="/m.gguf")
    assert argv_value(result.argv, "--ctx-size") == "8192"


def test_context_is_capped_by_the_configured_ceiling(gguf_file: Path) -> None:
    config = load(overrides={"tuning": {"maxContextSize": 8192}})
    result = plan(config, facts(), read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--ctx-size") == "8192"


def test_kv_cache_is_quantised_only_when_that_is_what_makes_the_context_fit() -> None:
    # Geometry borrowed from a 7B model: 56 KiB of KV per token at f16.
    class Geometry:
        n_ctx_train = 32768
        file_size = 4 * GiB

        @staticmethod
        def kv_cache_bytes_per_token(bits: int = 16) -> int:
            return 2 * 28 * 4 * 128 * bits // 8

    budget = 5 * GiB  # 1 GiB left for KV after the weights
    ctx, kv_type, notes = plan_context(budget, 4 * GiB, Geometry(), {"maxContextSize": 32768})
    assert kv_type == "q8_0", notes
    assert ctx <= 32768

    # With plenty of room, f16 is kept: quantising costs accuracy for nothing.
    ctx, kv_type, _ = plan_context(24 * GiB, 4 * GiB, Geometry(), {"maxContextSize": 32768})
    assert (ctx, kv_type) == (32768, "f16")


def test_an_explicit_context_wins_but_warns_when_it_does_not_fit(gguf_file: Path) -> None:
    config = load(overrides={"tuning": {"contextSize": 131072}})
    result = plan(config, facts(), read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--ctx-size") == "131072"
    assert any("exceeds the model's trained context" in line for line in result.explain)


def test_unknown_geometry_defers_context_sizing_to_llama_cpp() -> None:
    result = plan(load(), facts(), None, model_path="/m.gguf")
    assert "--ctx-size" not in result.argv
    assert any("deferring context sizing" in line for line in result.explain)


def test_gpu_plan_leaves_offload_to_fit(gguf_file: Path) -> None:
    gpu = facts(
        acceleration="cuda",
        vram_bytes=24 * GiB,
        accelerators=[Accelerator(vendor="nvidia", backend="cuda", name="GPU")],
    )
    result = plan(load(), gpu, read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--fit") == "on"
    assert "--gpu-layers" not in result.argv
    # Prompt processing is the GPU bottleneck, so the micro-batch grows.
    assert argv_value(result.argv, "--ubatch-size") == "1024"
    assert argv_value(result.argv, "--spec-type") == "ngram-mod"


def test_speculative_decoding_stays_off_on_cpu(gguf_file: Path) -> None:
    result = plan(load(), facts(), read_gguf(gguf_file), model_path="/m.gguf")
    assert "--spec-type" not in result.argv


def test_explicit_gpu_layers_are_honoured(gguf_file: Path) -> None:
    config = load(overrides={"tuning": {"gpuLayers": 20}})
    result = plan(config, facts(acceleration="cuda", vram_bytes=8 * GiB), read_gguf(gguf_file))
    assert argv_value(result.argv, "--gpu-layers") == "20"
    assert "--fit" not in result.argv


def test_small_memory_budget_shrinks_the_batch(gguf_file: Path) -> None:
    tight = facts(memory=MemoryInfo(total_bytes=4 * GiB, available_bytes=2 * GiB))
    result = plan(load(), tight, read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--batch-size") == "1024"
    assert argv_value(result.argv, "--ubatch-size") == "256"


def test_disk_cache_requires_flash_storage(gguf_file: Path) -> None:
    gguf = read_gguf(gguf_file)
    on_ssd = plan(load(), facts(), gguf, model_path="/m.gguf")
    assert "--slot-save-path" in on_ssd.argv

    spinning = facts(
        storage=StorageInfo(path="/var/lib/nix-llama", rotational=True, free_bytes=500 * GiB)
    )
    result = plan(load(), spinning, gguf, model_path="/m.gguf")
    assert "--slot-save-path" not in result.argv
    assert any("not flash" in line for line in result.explain)


def test_disk_cache_refuses_a_nearly_full_disk(gguf_file: Path) -> None:
    cramped = facts(
        storage=StorageInfo(path="/var/lib/nix-llama", rotational=False, free_bytes=2 * GiB)
    )
    result = plan(load(), cramped, read_gguf(gguf_file), model_path="/m.gguf")
    assert "--slot-save-path" not in result.argv
    assert any("only 2 GiB free" in line for line in result.warnings)


def test_mlock_only_when_the_model_fits_in_half_of_memory(tmp_path: Path) -> None:
    small = read_gguf(build_gguf(tmp_path / "small.gguf", padding=1024))
    assert (
        argv_value(plan(load(), facts(), small, model_path="/m.gguf").argv, "--load-mode")
        == "mmap+mlock"
    )

    tight = facts(memory=MemoryInfo(total_bytes=2048, available_bytes=1024))
    assert (
        argv_value(plan(load(), tight, small, model_path="/m.gguf").argv, "--load-mode") == "mmap"
    )


def test_settings_override_computed_values(gguf_file: Path) -> None:
    config = load(overrides={"settings": {"--ctx-size": 4096, "--temp": 0.1, "--props": True}})
    result = plan(config, facts(), read_gguf(gguf_file), model_path="/m.gguf")
    # The locked value wins, and is not duplicated.
    assert result.argv.count("--ctx-size") == 1
    assert argv_value(result.argv, "--ctx-size") == "4096"
    assert argv_value(result.argv, "--temp") == "0.1"
    # True renders as a bare flag, with no value after it.
    props = result.argv.index("--props")
    assert result.argv[props + 1].startswith("--")


def test_settings_accept_flags_without_dashes(gguf_file: Path) -> None:
    config = load(overrides={"settings": {"temp": 0.5}})
    result = plan(config, facts(), read_gguf(gguf_file))
    assert argv_value(result.argv, "--temp") == "0.5"


def test_extra_flags_are_appended_last(gguf_file: Path) -> None:
    config = load(overrides={"extraFlags": ["--props", "--log-timestamps"]})
    result = plan(config, facts(), read_gguf(gguf_file), model_path="/m.gguf")
    assert result.argv[-2:] == ["--props", "--log-timestamps"]


def test_security_defaults_are_conservative(gguf_file: Path) -> None:
    result = plan(load(), facts(), read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--host") == "127.0.0.1"
    assert argv_value(result.argv, "--cors-origins") == "localhost"
    # The llama.cpp built-in tool/agent flags must never be set by us.
    assert not any(flag in result.argv for flag in ("--tools", "--agent", "-ag"))


def test_api_key_and_tls_are_wired_through(gguf_file: Path) -> None:
    config = load(
        overrides={
            "apiKeyFile": "/run/secrets/key",
            "tls": {"certFile": "/tls/cert.pem", "keyFile": "/tls/key.pem"},
        }
    )
    result = plan(config, facts(), read_gguf(gguf_file))
    assert argv_value(result.argv, "--api-key-file") == "/run/secrets/key"
    assert argv_value(result.argv, "--ssl-cert-file") == "/tls/cert.pem"
    assert argv_value(result.argv, "--ssl-key-file") == "/tls/key.pem"


def test_upstream_port_moves_llama_server_to_loopback(gguf_file: Path) -> None:
    result = plan(
        load(overrides={"host": "0.0.0.0", "port": 8080}),
        facts(),
        read_gguf(gguf_file),
        upstream_host="127.0.0.1",
        upstream_port=18081,
    )
    # The proxy owns the public port; llama-server must not also be exposed.
    assert argv_value(result.argv, "--host") == "127.0.0.1"
    assert argv_value(result.argv, "--port") == "18081"


def test_unified_memory_budget_is_three_quarters_of_ram() -> None:
    apple = facts(
        os="darwin",
        acceleration="metal",
        unified_memory=True,
        memory=MemoryInfo(total_bytes=16 * GiB, available_bytes=12 * GiB),
    )
    budget, where = memory_budget(apple, "metal")
    assert where == "unified memory"
    assert budget == int(16 * GiB * 0.75) - 1 * GiB


def test_device_budget_reserves_headroom() -> None:
    gpu = facts(acceleration="cuda", vram_bytes=8 * GiB)
    budget, where = memory_budget(gpu, "cuda")
    assert where == "device memory"
    assert budget == 7 * GiB


def test_prompt_cache_is_sized_from_available_memory(gguf_file: Path) -> None:
    result = plan(load(), facts(), read_gguf(gguf_file), model_path="/m.gguf")
    cram = int(argv_value(result.argv, "--cache-ram"))
    assert 512 <= cram <= 8192
    assert cram == min(8192, (24 * GiB) // 4 // MiB)


def test_context_shift_can_be_disabled(gguf_file: Path) -> None:
    config = load(overrides={"tuning": {"contextShift": False}})
    result = plan(config, facts(), read_gguf(gguf_file))
    assert "--context-shift" not in result.argv


def test_plan_reports_probe_warnings(gguf_file: Path) -> None:
    noisy = facts()
    noisy.warnings.append("a GPU is going unused")
    result = plan(load(), noisy, read_gguf(gguf_file))
    assert "a GPU is going unused" in result.warnings


def test_command_is_shell_quoted(gguf_file: Path) -> None:
    result = plan(load(), facts(), read_gguf(gguf_file), model_path="/models/with space.gguf")
    assert "'/models/with space.gguf'" in result.command


@pytest.mark.parametrize("backend", ["cuda", "rocm", "vulkan", "metal"])
def test_every_backend_produces_a_runnable_plan(backend: str, gguf_file: Path) -> None:
    result = plan(
        load(),
        facts(acceleration=backend, vram_bytes=8 * GiB, unified_memory=backend == "metal"),
        read_gguf(gguf_file),
        model_path="/m.gguf",
    )
    assert result.argv[0].endswith("llama-server")
    assert "--model" in result.argv
    assert result.facts["backend"] == backend


def test_the_host_reserve_never_zeroes_a_small_budget() -> None:
    # A flat 1 GiB reserve leaves a 2 GiB container with nothing to plan
    # against, and then no model fits at all.
    tiny = facts(memory=MemoryInfo(total_bytes=2 * GiB, available_bytes=2 * GiB))
    budget, where = memory_budget(tiny, "cpu")
    assert where == "host memory"
    assert budget == 2 * GiB - int(2 * GiB * 0.2)

    # On a large machine the flat reserve is the smaller of the two and wins.
    big = facts(memory=MemoryInfo(total_bytes=32 * GiB, available_bytes=32 * GiB))
    assert memory_budget(big, "cpu")[0] == 31 * GiB


def test_a_tiny_machine_still_gets_a_runnable_plan(gguf_file: Path) -> None:
    tiny = facts(
        cpu=CPUInfo(logical=2, physical=2, available=2),
        memory=MemoryInfo(total_bytes=2 * GiB, available_bytes=1 * GiB),
        storage=StorageInfo(path="/var/lib/nix-llama", rotational=None, free_bytes=4 * GiB),
    )
    result = plan(load(), tiny, read_gguf(gguf_file), model_path="/m.gguf")
    assert argv_value(result.argv, "--threads") == "2"
    assert int(argv_value(result.argv, "--ctx-size")) >= 1024
    # Unknown storage class means no disk KV cache: spilling onto something
    # slow is worse than recomputing.
    assert "--slot-save-path" not in result.argv
