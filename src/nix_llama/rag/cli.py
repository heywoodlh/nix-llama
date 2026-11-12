"""``nix-llama-rag`` -- build, query and serve the documentation index."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .. import __version__
from ..config import load
from ..dataset import build_dataset, write_jsonl
from .corpus import LicenseError, attribution, build_corpus, load_manifest
from .index import build_index, index_paths, load_index
from .mcp import McpServer
from .proxy import RagState, serve
from .retrieve import format_context, search

log = logging.getLogger("nix-llama-rag")

EXIT_USAGE = 2
EXIT_NO_INDEX = 3
EXIT_LICENSE = 4


def _default_index(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("NIX_LLAMA_RAG_INDEX")
    if env:
        return Path(env)
    model_dir = os.environ.get("NIX_LLAMA_MODEL_DIR")
    if model_dir:
        candidate = Path(model_dir) / "rag" / "index.json.gz"
        if candidate.is_file():
            return candidate
    for candidate in (
        Path("rag/index.json.gz"),
        Path.home() / ".local/share/nix-llama/rag/index.json.gz",
        Path("/var/lib/nix-llama/rag/index.json.gz"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _load(explicit: str | None):
    path = _default_index(explicit)
    if path is None or not path.exists():
        log.error(
            "no documentation index found%s; build one with `nix-llama-rag index`",
            f" at {path}" if path else "",
        )
        return None
    return load_index(path)


def cmd_index(args: argparse.Namespace) -> int:
    chunks = []
    metadata: dict[str, dict[str, str]] = {}
    if args.manifest:
        try:
            sources = load_manifest(args.manifest)
            chunks, metadata = build_corpus(sources, verbose=args.verbose)
        except LicenseError as exc:
            log.error("%s", exc)
            return EXIT_LICENSE
        except (OSError, ValueError) as exc:
            log.error("cannot read manifest %s: %s", args.manifest, exc)
            return EXIT_USAGE
    for spec in args.path or []:
        # NAME:LICENSE:DIR, or just DIR.
        parts = spec.split(":", 2)
        if len(parts) == 3:
            name, license_id, directory = parts
        else:
            name, license_id, directory = "local", "unknown", spec
        found = index_paths([directory], source=name, license=license_id)
        metadata[name] = {
            "license": license_id,
            "homepage": "",
            "rev": "",
            "files": "",
            "chunks": str(len(found)),
            "share_alike": "no",
        }
        chunks.extend(found)

    if not chunks:
        log.error("nothing to index: pass --manifest and/or one or more paths")
        return EXIT_USAGE

    index = build_index(chunks, metadata)
    out = Path(args.output)
    index.save(out)
    log.info("indexed %d chunks from %d sources -> %s", index.size, len(metadata), out)
    if args.attribution:
        Path(args.attribution).write_text(attribution(metadata), encoding="utf-8")
        log.info("wrote attribution to %s", args.attribution)
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    """Derive the fine-tuning dataset from the licensed corpus."""
    try:
        sources = load_manifest(args.manifest)
        records, stats = build_dataset(sources)
    except LicenseError as exc:
        log.error("%s", exc)
        return EXIT_LICENSE
    except (OSError, ValueError) as exc:
        log.error("cannot read manifest %s: %s", args.manifest, exc)
        return EXIT_USAGE
    if not records:
        log.error("the manifest produced no training records")
        return EXIT_USAGE
    write_jsonl(records, args.output)
    log.info(
        "wrote %d records (%d text, %d chat) to %s",
        stats["records"],
        stats["by_kind"].get("text", 0),
        stats["by_kind"].get("chat", 0),
        args.output,
    )
    if args.stats:
        Path(args.stats).write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = _load(args.index)
    if index is None:
        return EXIT_NO_INDEX
    hits = search(index, args.query, top_k=args.top_k, min_score=args.min_score)
    if args.json:
        print(json.dumps([h.as_dict() for h in hits], indent=2))
        return 0
    if not hits:
        print("no matches")
        return 0
    if args.context:
        print(format_context(hits, max_chars=args.max_chars))
        return 0
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] {hit.score:.2f}  {hit.chunk.citation()}")
        snippet = " ".join(hit.chunk.text.split())[:200]
        print(f"     {snippet}...")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    index = _load(args.index)
    if index is None:
        return EXIT_NO_INDEX
    print(f"chunks     {index.size}")
    print(f"terms      {len(index.postings)}")
    print(f"avg length {index.avg_length:.1f} tokens")
    for name, meta in sorted(index.sources.items()):
        print(f"source     {name} ({meta.get('license', '?')}, {meta.get('chunks', '?')} chunks)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    index = _load(args.index)
    if index is None and not args.allow_empty:
        return EXIT_NO_INDEX
    config = load(config_file=args.config) if args.config else None
    rag = (config or {}).get("rag", {}) if config else {}
    state = RagState(
        index,
        top_k=args.top_k or int(rag.get("topK") or 4),
        max_chars=args.max_chars or int(rag.get("maxChars") or 6000),
        min_score=args.min_score,
        system_prompt=Path(args.system_prompt).read_text(encoding="utf-8")
        if args.system_prompt
        else None,
    )
    upstream_host, _, upstream_port = args.upstream.rpartition(":")
    server = serve(args.host, args.port, upstream_host or "127.0.0.1", int(upstream_port), state)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    index = _load(args.index)
    if index is None and not args.allow_empty:
        return EXIT_NO_INDEX
    # stdout is the transport, so logging must not touch it.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    return McpServer(index, default_top_k=args.top_k).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nix-llama-rag",
        description="Documentation retrieval for nix-llama.",
    )
    parser.add_argument("--version", action="version", version=f"nix-llama-rag {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--index", help="path to index.json.gz")
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="build an index from a manifest and/or paths")
    # Accepted here as well as before the subcommand: `nix-llama-rag index -v`
    # is what everyone types. SUPPRESS keeps it from clobbering the global flag.
    index_cmd.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    index_cmd.add_argument("-o", "--output", default="index.json.gz")
    index_cmd.add_argument("--manifest", help="JSON manifest of licensed sources")
    index_cmd.add_argument("--attribution", help="write an ATTRIBUTION.md here")
    index_cmd.add_argument("path", nargs="*", help="NAME:LICENSE:DIR or DIR")
    index_cmd.set_defaults(func=cmd_index)

    corpus_cmd = sub.add_parser(
        "corpus", help="derive a fine-tuning dataset from a licensed manifest"
    )
    corpus_cmd.add_argument("--manifest", required=True)
    corpus_cmd.add_argument("-o", "--output", default="corpus.jsonl")
    corpus_cmd.add_argument("--stats", help="write dataset statistics here as JSON")
    corpus_cmd.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    corpus_cmd.set_defaults(func=cmd_corpus)

    search_cmd = sub.add_parser("search", help="query the index")
    search_cmd.add_argument("query", nargs="+")
    search_cmd.add_argument("-k", "--top-k", type=int, default=5)
    search_cmd.add_argument("--min-score", type=float, default=0.0)
    search_cmd.add_argument("--max-chars", type=int, default=6000)
    search_cmd.add_argument("--context", action="store_true", help="render as a prompt block")
    search_cmd.add_argument("--json", action="store_true")
    search_cmd.set_defaults(func=cmd_search)

    stats_cmd = sub.add_parser("stats", help="describe the index")
    stats_cmd.set_defaults(func=cmd_stats)

    serve_cmd = sub.add_parser("serve", help="run the augmenting OpenAI-compatible proxy")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)
    serve_cmd.add_argument("--upstream", default="127.0.0.1:8081")
    serve_cmd.add_argument("-k", "--top-k", type=int, default=0)
    serve_cmd.add_argument("--max-chars", type=int, default=0)
    serve_cmd.add_argument("--min-score", type=float, default=1.0)
    serve_cmd.add_argument("--system-prompt", help="file injected when the client sends none")
    serve_cmd.add_argument("--config", help="nix-llama JSON config to read rag settings from")
    serve_cmd.add_argument("--allow-empty", action="store_true", help="start without an index")
    serve_cmd.set_defaults(func=cmd_serve)

    mcp_cmd = sub.add_parser("mcp", help="run the stdio MCP documentation server")
    mcp_cmd.add_argument("-k", "--top-k", type=int, default=4)
    mcp_cmd.add_argument("--allow-empty", action="store_true")
    mcp_cmd.set_defaults(func=cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "mcp":
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
    if isinstance(getattr(args, "query", None), list):
        args.query = " ".join(args.query)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
