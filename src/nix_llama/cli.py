"""``nix-llama`` -- probe the machine, then serve the model on it."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

from . import __version__
from .config import ConfigError, load
from .defaults import Plan, plan
from .gguf import GGUFError, GGUFInfo, read_gguf
from .probe import Probe, probe

log = logging.getLogger("nix-llama")

EXIT_CONFIG = 2
EXIT_NO_MODEL = 3
EXIT_UPSTREAM = 4


class ModelNotFoundError(Exception):
    pass


def resolve_model(config: dict[str, Any]) -> Path:
    """Locate the GGUF to serve.

    Order: an explicit ``model``, then the single GGUF in ``modelDir`` (which
    is how the ``model`` package presents itself), then the largest GGUF there
    if several exist -- a model directory normally holds one weights file plus
    optional projectors.
    """
    explicit = config.get("model")
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            config = {**config, "model": None, "modelDir": str(path)}
        elif path.is_file():
            return path
        else:
            raise ModelNotFoundError(f"configured model {path} does not exist")

    model_dir = config.get("modelDir")
    if not model_dir:
        raise ModelNotFoundError(
            "no model configured: set services.nix-llama.model, NIX_LLAMA_MODEL, "
            "or pass --model /path/to/model.gguf"
        )
    directory = Path(model_dir)
    manifest = directory / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            candidate = directory / data["weights"]
            if candidate.is_file():
                return candidate
        except (OSError, ValueError, KeyError):
            log.warning("ignoring unreadable manifest at %s", manifest)
    ggufs = sorted(p for p in directory.glob("*.gguf") if p.is_file())
    if not ggufs:
        raise ModelNotFoundError(f"no .gguf file found in {directory}")
    if len(ggufs) == 1:
        return ggufs[0]
    return max(ggufs, key=lambda p: p.stat().st_size)


def read_model_metadata(path: Path) -> GGUFInfo | None:
    try:
        return read_gguf(path)
    except (GGUFError, OSError) as exc:
        # Not fatal: without geometry we simply defer sizing to llama.cpp.
        log.warning("could not read GGUF metadata from %s: %s", path, exc)
        return None


def _state_dir(config: dict[str, Any]) -> Path:
    configured = config.get("stateDir")
    if configured:
        return Path(configured)
    runtime = os.environ.get("STATE_DIRECTORY")
    if runtime:
        return Path(runtime.split(":")[0])
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "nix-llama"
    return Path.home() / ".local" / "state" / "nix-llama"


def _free_port(host: str, preferred: int) -> int:
    """Pick a loopback port for llama-server when the proxy owns the real one."""
    for candidate in (preferred, preferred + 1, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, candidate))
                return sock.getsockname()[1]
        except OSError:
            continue
    raise OSError("no free loopback port for the llama-server backend")


def _system_prompt(config: dict[str, Any], model_dir: Path | None) -> str | None:
    """Read the system prompt from config, or from the model package."""
    configured = config.get("systemPrompt")
    if configured:
        path = Path(configured)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        return str(configured)
    if model_dir:
        candidate = model_dir / "system-prompt.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return None


def _resolve_index(config: dict[str, Any], model_dir: Path | None) -> Path | None:
    configured = (config.get("rag") or {}).get("index")
    if configured:
        return Path(configured)
    if model_dir:
        for candidate in (model_dir / "rag" / "index.json.gz", model_dir / "index.json.gz"):
            if candidate.is_file():
                return candidate
    return None


def build_plan(config: dict[str, Any], *, upstream_port: int | None = None) -> tuple[Plan, Probe]:
    model = resolve_model(config)
    state = _state_dir(config)
    llama_server = shutil.which(config.get("llamaServer") or "llama-server")
    facts = probe(
        state_dir=str(state),
        llama_server=llama_server,
        force_acceleration=config.get("acceleration"),
    )
    gguf = read_model_metadata(model)
    upstream_host = "127.0.0.1" if upstream_port else None
    result = plan(
        config,
        facts,
        gguf,
        model_path=str(model),
        upstream_host=upstream_host,
        upstream_port=upstream_port,
    )
    if llama_server:
        result.argv[0] = llama_server
    result.facts["state_dir"] = str(state)
    result.facts["model_metadata"] = gguf.as_dict() if gguf else None
    return result, facts


def _prepare_state(plan_result: Plan) -> None:
    """Create directories llama-server expects to already exist."""
    argv = plan_result.argv
    if "--slot-save-path" in argv:
        path = Path(argv[argv.index("--slot-save-path") + 1])
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("cannot create slot cache directory %s: %s", path, exc)
            idx = argv.index("--slot-save-path")
            del argv[idx : idx + 2]


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_probe(args: argparse.Namespace) -> int:
    config = _load_config(args)
    llama_server = shutil.which(config.get("llamaServer") or "llama-server")
    facts = probe(
        root=args.root,
        state_dir=config.get("stateDir") or str(_state_dir(config)),
        llama_server=llama_server,
        force_acceleration=config.get("acceleration"),
    )
    if args.format == "json":
        print(facts.to_json())
        return 0
    print(_render_probe(facts))
    return 0


def _render_probe(facts: Probe) -> str:
    gib = 1024**3
    lines = [
        f"os            {facts.os}/{facts.arch}"
        + (f" (container: {facts.container_runtime})" if facts.container else ""),
        f"cpu           {facts.cpu.model or 'unknown'}",
        f"              {facts.cpu.available} usable / {facts.cpu.physical} physical / "
        f"{facts.cpu.logical} logical cores"
        + (f"; isa: {', '.join(facts.cpu.isa)}" if facts.cpu.isa else ""),
        f"memory        {facts.memory.total_bytes / gib:.1f} GiB total, "
        f"{facts.memory.available_bytes / gib:.1f} GiB available"
        + (
            f", cgroup limit {facts.memory.limit_bytes / gib:.1f} GiB"
            if facts.memory.limit_bytes
            else ""
        ),
        f"acceleration  {facts.acceleration}"
        + (f" ({facts.vram_bytes / gib:.1f} GiB device memory)" if facts.vram_bytes else ""),
    ]
    for accel in facts.accelerators:
        size = f", {accel.memory_bytes / gib:.1f} GiB" if accel.memory_bytes else ""
        lines.append(f"              detected: {accel.vendor} {accel.name} [{accel.backend}{size}]")
    for device in facts.devices:
        lines.append(f"              llama.cpp device: {device['backend']}: {device['name']}")
    lines.append(
        f"storage       {facts.storage.path} "
        f"({'flash' if facts.storage.is_ssd else 'rotational/unknown'}, "
        f"{facts.storage.free_bytes / gib:.1f} GiB free)"
    )
    for warning in facts.warnings:
        lines.append(f"warning       {warning}")
    return "\n".join(lines)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print a GGUF file's metadata, optionally asserting a size budget.

    The size assertion is used by the model derivation: nix-llama's model is
    specified to stay under 5 GB so it fits on an 8 GiB accelerator, and a
    build-time check is the only way that stays true.
    """
    path = Path(args.gguf)
    try:
        info = read_gguf(path)
    except (GGUFError, OSError) as exc:
        log.error("%s", exc)
        return 1
    data = info.as_dict()
    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        width = max(len(k) for k in data)
        for key in sorted(data):
            print(f"{key.ljust(width)}  {data[key]}")
    if args.max_bytes and info.file_size > args.max_bytes:
        log.error(
            "%s is %.2f GB, over the %.2f GB budget",
            path.name,
            info.file_size / 1e9,
            args.max_bytes / 1e9,
        )
        return 1
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    config = _load_config(args)
    try:
        result, facts = build_plan(config)
    except ModelNotFoundError as exc:
        log.error("%s", exc)
        return EXIT_NO_MODEL
    if args.format == "json":
        print(
            json.dumps(
                {
                    "argv": result.argv,
                    "explain": result.explain,
                    "warnings": result.warnings,
                    "facts": result.facts,
                    "probe": facts.as_dict(),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0
    print(result.command)
    print()
    for line in result.explain:
        print(f"  - {line}")
    for line in result.warnings:
        print(f"  ! {line}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    config = _load_config(args)
    rag_config = config.get("rag") or {}
    rag_enabled = bool(rag_config.get("enable", True))

    upstream_port: int | None = None
    if rag_enabled:
        preferred = int(rag_config.get("upstreamPort") or (int(config["port"]) + 1))
        try:
            upstream_port = _free_port("127.0.0.1", preferred)
        except OSError as exc:
            log.error("%s", exc)
            return EXIT_UPSTREAM

    try:
        result, _facts = build_plan(config, upstream_port=upstream_port)
    except ModelNotFoundError as exc:
        log.error("%s", exc)
        return EXIT_NO_MODEL

    _prepare_state(result)
    for line in result.explain:
        log.info("%s", line)
    for line in result.warnings:
        log.warning("%s", line)

    if args.dry_run:
        print(result.command)
        return 0

    log.info("exec: %s", result.command)
    try:
        child = subprocess.Popen(result.argv)  # noqa: S603 - argv is built, not shell
    except OSError as exc:
        log.error("cannot start llama-server: %s", exc)
        return EXIT_UPSTREAM

    proxy = None
    if rag_enabled and upstream_port:
        proxy = _start_proxy(config, upstream_port, result)

    stopping = threading.Event()

    def _forward(signum: int, _frame: FrameType | None) -> None:
        if not stopping.is_set():
            stopping.set()
            log.info("received signal %s; shutting down", signal.Signals(signum).name)
            child.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _forward)
        except (ValueError, OSError):
            continue

    try:
        code = child.wait()
    except KeyboardInterrupt:
        child.terminate()
        code = child.wait()
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
    if code < 0:
        # Popen reports a signal death as a negative number. Returning it as-is
        # wraps around into systemd's reserved exit-status range, where 225
        # means "failed to set up networking" -- a diagnosis that sends the
        # next person debugging this in entirely the wrong direction.
        signame = signal.Signals(-code).name
        if not stopping.is_set():
            log.error("llama-server was killed by %s", signame)
        return 128 - code
    if code != 0 and not stopping.is_set():
        log.error("llama-server exited with status %s", code)
    return code


def _start_proxy(config: dict[str, Any], upstream_port: int, result: Plan) -> Any:
    from .rag.index import load_index
    from .rag.proxy import RagState, context_budget, serve

    rag_config = config.get("rag") or {}
    model_dir = Path(result.facts["model"]).parent if result.facts.get("model") else None
    index = None
    index_path = _resolve_index(config, model_dir)
    if index_path and Path(index_path).exists():
        try:
            index = load_index(index_path)
            log.info("loaded %d documentation chunks from %s", index.size, index_path)
        except (OSError, ValueError) as exc:
            log.warning("ignoring unusable RAG index %s: %s", index_path, exc)
    else:
        log.info("no RAG index found; the proxy will relay requests unaugmented")

    system_prompt = _system_prompt(config, model_dir)

    # Retrieval must not crowd the context window: a request that arrives
    # larger than the server's context is rejected outright, which is a worse
    # outcome than answering without documentation. The default system prompt
    # is injected unconditionally, so it comes out of the same budget.
    configured = int(rag_config.get("maxChars") or 6000)
    max_chars = context_budget(
        result.facts.get("context_size"), configured, len(system_prompt or "")
    )
    if max_chars < configured:
        log.info(
            "retrieval budget reduced to %d characters to fit a %s-token context",
            max_chars,
            result.facts.get("context_size"),
        )
    if max_chars == 0:
        log.warning(
            "the context window leaves no room for retrieved documentation; "
            "retrieval will be inert until the context size is raised"
        )

    state = RagState(
        index,
        top_k=int(rag_config.get("topK") or 4),
        max_chars=max_chars,
        min_score=float(rag_config.get("minScore") or 1.0),
        system_prompt=system_prompt,
        inject_into=str(rag_config.get("injectInto") or "system"),
    )
    server = serve(config["host"], int(config["port"]), "127.0.0.1", upstream_port, state)
    thread = threading.Thread(target=server.serve_forever, name="rag-proxy", daemon=True)
    thread.start()
    return server


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that everything needed to serve is actually in place."""
    config = _load_config(args)
    problems: list[str] = []
    notes: list[str] = []

    binary = shutil.which(config.get("llamaServer") or "llama-server")
    if binary:
        notes.append(f"llama-server: {binary}")
    else:
        problems.append("llama-server is not on PATH")

    model_dir: Path | None = None
    try:
        model = resolve_model(config)
        model_dir = model.parent
        size_gib = model.stat().st_size / 1024**3
        notes.append(f"model: {model} ({size_gib:.2f} GiB)")
        gguf = read_model_metadata(model)
        if gguf:
            notes.append(
                f"model geometry: {gguf.architecture}, {gguf.n_layer} layers, "
                f"trained context {gguf.n_ctx_train}"
            )
        else:
            problems.append(f"{model} is not readable as GGUF")
    except ModelNotFoundError as exc:
        problems.append(str(exc))

    state = _state_dir(config)
    if state.exists() and not os.access(state, os.W_OK):
        problems.append(f"state directory {state} is not writable")
    else:
        notes.append(f"state directory: {state}")

    index_path = _resolve_index(config, model_dir)
    if (config.get("rag") or {}).get("enable", True):
        if index_path and Path(index_path).exists():
            notes.append(f"RAG index: {index_path}")
        else:
            notes.append("RAG index: none found (retrieval will be inert)")

    for line in notes:
        print(f"ok    {line}")
    for line in problems:
        print(f"FAIL  {line}")
    return 1 if problems else 0


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("model", "model_dir", "host", "port", "acceleration", "state_dir"):
        value = getattr(args, key, None)
        if value is not None:
            camel = {"model_dir": "modelDir", "state_dir": "stateDir"}.get(key, key)
            overrides[camel] = value
    tuning: dict[str, Any] = {}
    for key, name in (
        ("ctx_size", "contextSize"),
        ("threads", "threads"),
        ("gpu_layers", "gpuLayers"),
    ):
        value = getattr(args, key, None)
        if value is not None:
            tuning[name] = value
    if tuning:
        overrides["tuning"] = tuning
    if getattr(args, "no_rag", False):
        overrides["rag"] = {"enable": False}
    if getattr(args, "extra", None):
        overrides["extraFlags"] = list(args.extra)
    return load(config_file=args.config, overrides=overrides)


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nix-llama",
        description="Run llama-server with the best defaults for this machine.",
    )
    parser.add_argument("--version", action="version", version=f"nix-llama {__version__}")
    parser.add_argument("-c", "--config", help="path to a JSON configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS keeps an unspecified subcommand flag from clobbering the value
    # given before the subcommand (`nix-llama -c x.json probe`).
    common.add_argument("-c", "--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    common.add_argument("-m", "--model", help="path to a .gguf file or a model directory")
    common.add_argument("--model-dir", help="directory to search for a .gguf file")
    common.add_argument("--host", help="address to listen on")
    common.add_argument("--port", type=int, help="port to listen on")
    common.add_argument(
        "--acceleration",
        choices=["auto", "none", "cuda", "rocm", "vulkan", "metal", "sycl", "opencl"],
        help="force a backend instead of detecting one",
    )
    common.add_argument("--state-dir", help="writable directory for caches and slot data")
    common.add_argument("--ctx-size", type=int, help="override the detected context size")
    common.add_argument("--threads", type=int, help="override the detected thread count")
    common.add_argument("--gpu-layers", type=int, help="override the number of offloaded layers")
    common.add_argument("--no-rag", action="store_true", help="disable the retrieval proxy")

    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", parents=[common], help="run the server (default)")
    serve.add_argument("--dry-run", action="store_true", help="print the command and exit")
    serve.add_argument("extra", nargs="*", help="extra flags passed through to llama-server")
    serve.set_defaults(func=cmd_serve)

    probe_cmd = sub.add_parser("probe", parents=[common], help="print detected hardware facts")
    probe_cmd.add_argument("--root", default="/", help="filesystem root to probe (for testing)")
    probe_cmd.add_argument("--format", choices=["text", "json"], default="text")
    probe_cmd.set_defaults(func=cmd_probe)

    plan_cmd = sub.add_parser("plan", parents=[common], help="show the llama-server command")
    plan_cmd.add_argument("--format", choices=["text", "json"], default="text")
    plan_cmd.set_defaults(func=cmd_plan)

    inspect = sub.add_parser("inspect", help="print GGUF metadata")
    inspect.add_argument("gguf")
    inspect.add_argument("--format", choices=["text", "json"], default="text")
    inspect.add_argument(
        "--max-bytes", type=int, default=0, help="fail if the file is larger than this"
    )
    inspect.set_defaults(func=cmd_inspect)

    doctor = sub.add_parser("doctor", parents=[common], help="check this installation")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # `nix-llama --port 9000` and `nix run .#nix-llama -- -m model.gguf` should
    # both work, so an absent subcommand means "serve".
    known = {"serve", "probe", "plan", "doctor", "inspect"}
    passthrough = {"-h", "--help", "--version"}
    # `nix-llama`, `nix-llama --port 9000` and `nix-llama -m m.gguf` all mean
    # "serve"; only an explicit subcommand or a help flag escapes that.
    if not argv or (argv[0] not in known and argv[0] not in passthrough):
        argv = ["serve", *argv]
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    func = getattr(args, "func", cmd_serve)
    try:
        return func(args)
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        return 130


def probe_main(argv: list[str] | None = None) -> int:
    """Entry point for the standalone ``nix-llama-probe`` command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return main(["probe", *argv])


if __name__ == "__main__":
    sys.exit(main())
