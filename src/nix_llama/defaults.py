"""Turn probe facts into a llama-server command line.

The planner is pure: it takes facts (:class:`nix_llama.probe.Probe`), model
metadata (:class:`nix_llama.gguf.GGUFInfo`) and a configuration dict, and
returns an argument vector plus a human-readable explanation of every choice.
No I/O happens here, which is what makes the defaults testable.

Precedence, strongest first:

1. ``extraFlags``   -- raw argv, appended last, never inspected
2. ``settings``     -- explicit ``{"--flag": value}`` overrides
3. ``tuning``       -- semantic knobs (contextSize, threads, ...)
4. inferred values  -- what the probe and the GGUF header imply
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from .gguf import GGUFInfo
from .probe import Probe

MiB = 1024 * 1024
GiB = 1024 * MiB

# Head-room left on an accelerator for activations, the compute graph and the
# backend's own allocations. llama.cpp's own --fit default is 1 GiB.
DEVICE_RESERVE_BYTES = 1 * GiB
# Host RAM we refuse to plan into, so the box stays usable. Capped at a share
# of what exists: a flat 1 GiB reserve leaves a 2 GiB container or a small
# single-board machine with a budget of zero, and then nothing fits at all.
HOST_RESERVE_BYTES = 1 * GiB
HOST_RESERVE_FRACTION = 0.2

# Bits per element of the supported KV cache types. q8_0 stores 32 values per
# 34-byte block, so 8.5 bits -- rounding up keeps the estimate conservative.
KV_BITS = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 9, "q5_1": 6, "q5_0": 6, "q4_1": 5, "q4_0": 5}
# Elements per block of the quantised KV types. llama.cpp refuses to create a
# context when the attention head dimension is not a multiple of this, so the
# planner must not offer such a type to a model with small heads.
KV_BLOCK = {"q8_0": 32, "q5_1": 32, "q5_0": 32, "q4_1": 32, "q4_0": 32}


@dataclass
class Plan:
    """A ready-to-exec llama-server invocation plus the reasoning behind it."""

    argv: list[str] = field(default_factory=list)
    explain: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


class _Args:
    """Ordered flag accumulator that honours the precedence rules."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._values: dict[str, Any] = {}
        self._locked: set[str] = set()

    def set(self, flag: str, value: Any = True, *, lock: bool = False) -> None:
        """Set ``flag``, unless a higher-precedence layer already locked it."""
        if flag in self._locked:
            return
        if flag not in self._values:
            self._order.append(flag)
        self._values[flag] = value
        if lock:
            self._locked.add(flag)

    def has(self, flag: str) -> bool:
        return flag in self._values

    def drop(self, flag: str) -> None:
        if flag in self._locked:
            return
        self._values.pop(flag, None)
        if flag in self._order:
            self._order.remove(flag)

    def to_argv(self) -> list[str]:
        argv: list[str] = []
        for flag in self._order:
            value = self._values[flag]
            if value is True:
                argv.append(flag)
            elif value is False or value is None:
                continue
            else:
                argv.extend([flag, str(value)])
        return argv


def _round_ctx(tokens: int, step: int = 1024) -> int:
    return max(step, (tokens // step) * step)


def memory_budget(facts: Probe, backend: str) -> tuple[int, str]:
    """Bytes available for model weights plus KV cache, and where they live."""
    if backend == "metal" or (backend != "cpu" and facts.unified_memory):
        total = facts.memory.budget_bytes
        # Metal can address ~75% of unified memory; leave the rest to the OS.
        return max(0, int(total * 0.75) - DEVICE_RESERVE_BYTES), "unified memory"
    if backend != "cpu" and facts.vram_bytes:
        return max(0, facts.vram_bytes - DEVICE_RESERVE_BYTES), "device memory"
    total = facts.memory.budget_bytes
    reserve = min(HOST_RESERVE_BYTES, int(total * HOST_RESERVE_FRACTION))
    return max(0, total - reserve), "host memory"


def plan_context(
    budget: int,
    model_bytes: int,
    gguf: GGUFInfo | None,
    tuning: dict[str, Any],
) -> tuple[int | None, str, list[str]]:
    """Choose a context size and KV cache type that fit the memory budget.

    Returns ``(ctx, kv_type, notes)``. ``ctx`` is ``None`` when the model's
    geometry is unknown, in which case llama.cpp's own ``--fit`` logic is left
    to decide -- a worse answer than ours, but never a wrong one.
    """
    notes: list[str] = []
    requested = tuning.get("contextSize")
    kv_setting = (tuning.get("kvCacheType") or "auto").lower()
    ceiling = int(tuning.get("maxContextSize") or 32768)
    floor = int(tuning.get("minContextSize") or 4096)

    if gguf is None or not gguf.kv_cache_bytes_per_token(16):
        if requested:
            notes.append(f"context {requested} taken from configuration")
            return int(requested), ("f16" if kv_setting == "auto" else kv_setting), notes
        notes.append("model geometry unknown; deferring context sizing to llama.cpp --fit")
        return None, ("auto" if kv_setting == "auto" else kv_setting), notes

    train_ctx = gguf.n_ctx_train or ceiling
    target = min(ceiling, train_ctx)
    if requested:
        target = int(requested)
        if train_ctx and target > train_ctx:
            notes.append(
                f"requested context {target} exceeds the model's trained context "
                f"{train_ctx}; quality past that point depends on RoPE scaling"
            )

    kv_room = max(0, budget - model_bytes)
    candidates = ["f16", "q8_0"] if kv_setting == "auto" else [kv_setting]
    if kv_setting == "auto":
        # Unknown geometry keeps the old behaviour; a known head dimension that
        # the block size does not divide would make llama-server exit at load.
        head_dim = getattr(gguf, "head_dim", None)
        candidates = [t for t in candidates if not head_dim or head_dim % KV_BLOCK.get(t, 1) == 0]
    chosen_type = candidates[0]
    ctx = 0
    for kv_type in candidates:
        per_token = gguf.kv_cache_bytes_per_token(KV_BITS.get(kv_type, 16)) or 1
        fits = int(kv_room // per_token)
        chosen_type = kv_type
        ctx = min(target, _round_ctx(fits))
        if ctx >= target:
            break

    if requested:
        # An explicit request wins even if it will not fit; warn instead of
        # silently serving a smaller context than the operator asked for.
        if ctx < int(requested):
            notes.append(
                f"configured context {requested} needs more memory than the "
                f"{budget // MiB} MiB budget allows (~{ctx} would fit); llama.cpp will "
                f"reduce GPU offload to compensate"
            )
        return int(requested), (chosen_type if kv_setting == "auto" else kv_setting), notes

    # The floor is a memory-pressure backstop, not a licence to exceed what the
    # model was trained for: a tiny model with a 2048-token training context
    # must not be started at 4096 just because the floor says so.
    effective_floor = min(floor, target)
    if ctx < effective_floor:
        notes.append(
            f"only ~{ctx} tokens of KV cache fit in the {budget // MiB} MiB budget; "
            f"holding context at the {effective_floor}-token floor and letting llama.cpp spill"
        )
        ctx = effective_floor
    if chosen_type != "f16":
        notes.append(f"KV cache quantised to {chosen_type} to reach a {ctx}-token context")
    notes.append(f"context {ctx} tokens (model trained for {train_ctx})")
    return ctx, chosen_type, notes


def plan(
    config: dict[str, Any],
    facts: Probe,
    gguf: GGUFInfo | None = None,
    *,
    model_path: str | None = None,
    upstream_host: str | None = None,
    upstream_port: int | None = None,
) -> Plan:
    """Compose the llama-server argument vector."""
    result = Plan()
    result.warnings.extend(facts.warnings)
    tuning = config.get("tuning", {})
    args = _Args()

    # ---- layer 2/1: user-supplied overrides are locked in first so that
    # nothing computed below can displace them.
    for flag, value in (config.get("settings") or {}).items():
        flag = flag if flag.startswith("-") else f"--{flag}"
        args.set(flag, value, lock=True)
    if config.get("settings"):
        result.explain.append(f"settings override: {' '.join(sorted(config['settings']))} (locked)")

    backend = facts.acceleration
    budget, budget_where = memory_budget(facts, backend)
    model = model_path or config.get("model")
    model_bytes = gguf.file_size if gguf else 0

    args.set("--model", model)
    args.set("--alias", config.get("alias") or "nix-llama")
    args.set("--host", upstream_host or config.get("host") or "127.0.0.1")
    args.set("--port", upstream_port or config.get("port") or 8080)

    result.explain.append(
        f"backend {backend} with a {budget // MiB} MiB {budget_where} budget"
        + (f"; model occupies {model_bytes // MiB} MiB" if model_bytes else "")
    )

    # ---- context and KV cache
    ctx, kv_type, notes = plan_context(budget, model_bytes, gguf, tuning)
    result.explain.extend(notes)
    if ctx:
        args.set("--ctx-size", ctx)
    if kv_type not in ("auto", "f16"):
        args.set("--cache-type-k", kv_type)
        args.set("--cache-type-v", kv_type)

    # ---- threading. Hyper-threads and efficiency cores both hurt more than
    # they help, so probe.cpu.available has already excluded them.
    threads = tuning.get("threads") or facts.cpu.available
    args.set("--threads", threads)
    args.set("--threads-batch", tuning.get("threadsBatch") or threads)
    result.explain.append(
        f"{threads} threads (of {facts.cpu.logical} logical / {facts.cpu.physical} physical cores)"
    )

    # ---- offload. Left unset on purpose when the user has no opinion:
    # llama.cpp's --fit computes the largest offload that fits, and it can see
    # live free VRAM that a one-shot probe cannot.
    gpu_layers = tuning.get("gpuLayers")
    if gpu_layers is not None:
        args.set("--gpu-layers", gpu_layers)
        result.explain.append(f"offloading {gpu_layers} layers (configured)")
    elif backend == "cpu":
        result.explain.append("no usable accelerator; running on CPU")
    else:
        args.set("--fit", "on")
        result.explain.append("GPU offload left to llama.cpp --fit (auto)")

    # ---- batching
    if tuning.get("batchSize"):
        args.set("--batch-size", tuning["batchSize"])
    elif budget and budget < 6 * GiB:
        args.set("--batch-size", 1024)
        result.explain.append("batch size reduced to 1024 for a small memory budget")
    if tuning.get("ubatchSize"):
        args.set("--ubatch-size", tuning["ubatchSize"])
    elif budget and budget < 6 * GiB:
        args.set("--ubatch-size", 256)
    elif backend != "cpu" and budget >= 16 * GiB:
        args.set("--ubatch-size", 1024)
        result.explain.append("ubatch raised to 1024: prompt processing is the bottleneck on GPU")

    if tuning.get("parallel"):
        args.set("--parallel", tuning["parallel"])

    # ---- prompt cache. Reuse is what makes an agent harness feel fast: each
    # turn resends the whole conversation, and cache-reuse lets llama.cpp skip
    # re-evaluating the unchanged prefix.
    args.set("--cache-reuse", 256)
    cram = tuning.get("promptCacheMiB")
    if cram is None:
        host_free = facts.memory.available_bytes or facts.memory.budget_bytes
        cram = max(512, min(8192, int(host_free * 0.25) // MiB))
    args.set("--cache-ram", cram)
    result.explain.append(f"{cram} MiB of host RAM for the prompt cache")

    # ---- disk-backed KV cache
    disk = config.get("diskCache") or {}
    slot_path = disk.get("path")
    if disk.get("enable", True) and (slot_path or facts.storage.is_ssd):
        path = slot_path or f"{facts.storage.path.rstrip('/')}/slots/"
        free_gib = facts.storage.free_bytes // GiB
        if free_gib >= int(disk.get("minFreeGiB") or 8):
            args.set("--slot-save-path", path)
            result.explain.append(
                f"slot KV cache may spill to {path} (flash storage, {free_gib} GiB free)"
            )
        else:
            result.warnings.append(
                f"disk KV cache disabled: only {free_gib} GiB free at {facts.storage.path}"
            )
    elif disk.get("enable", True):
        result.explain.append("disk KV cache disabled: backing store is not flash")

    # ---- loading strategy
    load_mode = tuning.get("loadMode")
    if load_mode is None and model_bytes:
        host = facts.memory.budget_bytes
        # Pinning the weights removes page-fault stalls mid-generation, but
        # only when there is room to spare for everything else.
        load_mode = "mmap+mlock" if host and model_bytes < host * 0.5 else "mmap"
    if load_mode:
        args.set("--load-mode", load_mode)
        result.explain.append(f"load mode {load_mode}")

    # ---- speculative decoding. ngram-mod drafts from the prompt itself, with
    # no draft model to load; code editing (large verbatim spans echoed back)
    # is the workload it helps most. On CPU the draft verification competes
    # with generation for the same cores, so it stays off there.
    spec = (tuning.get("speculative") or "auto").lower()
    if spec == "auto":
        spec = "ngram-mod" if backend != "cpu" else "off"
    if spec not in ("off", "none", ""):
        args.set("--spec-type", spec)
        result.explain.append(f"speculative decoding: {spec}")

    # ---- context shift keeps a long session answering instead of erroring out
    # with "context full"; the cost is that the oldest tokens silently fall off.
    if tuning.get("contextShift", True):
        args.set("--context-shift")
        result.explain.append("context shift on: long sessions degrade instead of failing")

    fa = (tuning.get("flashAttention") or "auto").lower()
    if fa != "auto":
        args.set("--flash-attn", fa)

    # ---- serving surface
    if tuning.get("metrics", True):
        args.set("--metrics")
    if not tuning.get("webui", True):
        args.set("--no-webui")
    cors = tuning.get("corsOrigins")
    if cors:
        # Default to localhost rather than llama.cpp's "*": a wildcard with
        # credentials echoed back lets any page in the browser reach the model.
        args.set("--cors-origins", cors)

    if config.get("apiKeyFile"):
        args.set("--api-key-file", config["apiKeyFile"])
        result.explain.append("API key authentication enabled")
    tls = config.get("tls") or {}
    if tls.get("certFile") and tls.get("keyFile"):
        args.set("--ssl-cert-file", tls["certFile"])
        args.set("--ssl-key-file", tls["keyFile"])
        result.explain.append("TLS enabled")

    argv = [config.get("llamaServer") or "llama-server", *args.to_argv()]
    # ---- layer 1: raw escape hatch, appended last so it wins outright.
    extra = config.get("extraFlags") or []
    if extra:
        argv.extend(str(x) for x in extra)
        result.explain.append(f"extraFlags appended: {' '.join(str(x) for x in extra)}")

    result.argv = argv
    result.facts = {
        "backend": backend,
        "budget_bytes": budget,
        "budget_where": budget_where,
        "context_size": ctx,
        "kv_cache_type": kv_type,
        "threads": threads,
        "model": model,
        "model_bytes": model_bytes,
    }
    return result
