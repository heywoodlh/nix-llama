"""A stdio MCP server exposing the documentation index as a tool.

The proxy injects context automatically; this is the other half -- it lets a
harness hand the model a tool so it can look something up *because it knows it
does not know*. Newline-delimited JSON-RPC 2.0 over stdin/stdout, which is what
MCP's stdio transport specifies.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, TextIO

from .. import __version__
from .index import Index
from .retrieve import search

log = logging.getLogger("nix-llama.mcp")

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search the nix-llama documentation corpus (Nix, NixOS, nix-darwin, "
            "home-manager, Hyprland, Niri, Kubernetes, CI/CD, Proton/Linux gaming, "
            "macOS tooling) and return the most relevant excerpts with their sources. "
            "Use this whenever you are unsure about an option name, a flag, or current "
            "upstream behaviour instead of guessing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up. Include exact option or flag names.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many excerpts to return (default 4, max 20).",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_doc_sources",
        "description": (
            "List the documentation sources in the index, with the licence and upstream "
            "revision each was built from. Use this to tell the user how current the "
            "indexed documentation is."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class McpServer:
    def __init__(self, index: Index | None, default_top_k: int = 4) -> None:
        self.index = index
        self.default_top_k = default_top_k

    # -- tool implementations ---------------------------------------------- #

    def _search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return self._error_content("search_docs requires a non-empty 'query'")
        if self.index is None:
            return self._error_content(
                "no documentation index is loaded; run `nix-llama-rag index` first"
            )
        top_k = max(1, min(20, int(args.get("top_k") or self.default_top_k)))
        hits = search(self.index, query, top_k=top_k, min_score=0.0)
        if not hits:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No documentation matched {query!r}. "
                        f"Say so rather than inventing an answer.",
                    }
                ]
            }
        blocks = [
            f"## [{i}] {hit.chunk.citation()}\n(licence: {hit.chunk.license or 'unknown'}, "
            f"score {hit.score:.2f})\n\n{hit.chunk.text.strip()}"
            for i, hit in enumerate(hits, start=1)
        ]
        return {"content": [{"type": "text", "text": "\n\n".join(blocks)}]}

    def _list_sources(self, _args: dict[str, Any]) -> dict[str, Any]:
        if self.index is None:
            return self._error_content("no documentation index is loaded")
        if not self.index.sources:
            return {
                "content": [
                    {"type": "text", "text": f"{self.index.size} chunks, source metadata absent"}
                ]
            }
        lines = [f"{self.index.size} indexed chunks from:"]
        for name, meta in sorted(self.index.sources.items()):
            detail = ", ".join(f"{k}={v}" for k, v in sorted(meta.items()) if k != "name")
            lines.append(f"- {name}: {detail}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @staticmethod
    def _error_content(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    # -- JSON-RPC ---------------------------------------------------------- #

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            wanted = params.get("protocolVersion")
            version = wanted if wanted in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
            return self._ok(
                request_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "nix-llama-docs", "version": __version__},
                    "instructions": (
                        "Call search_docs before answering questions about Nix, NixOS, "
                        "nix-darwin, home-manager, Hyprland, Niri, Kubernetes, CI/CD or "
                        "Proton when you are not certain of the current answer."
                    ),
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return self._ok(request_id, {})
        if method == "tools/list":
            return self._ok(request_id, {"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "search_docs":
                return self._ok(request_id, self._search(args))
            if name == "list_doc_sources":
                return self._ok(request_id, self._list_sources(args))
            return self._fail(request_id, -32602, f"unknown tool {name!r}")
        if isinstance(method, str) and method.startswith("notifications/"):
            return None
        return self._fail(request_id, -32601, f"method {method!r} is not supported")

    @staticmethod
    def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _fail(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def run(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                response = self._fail(None, -32700, "parse error")
            else:
                if isinstance(request, list):
                    # Batched requests: respond with the non-notification results.
                    responses = [r for r in (self.handle(item) for item in request) if r]
                    if not responses:
                        continue
                    stdout.write(json.dumps(responses) + "\n")
                    stdout.flush()
                    continue
                try:
                    response = self.handle(request)
                except Exception as exc:
                    log.exception("MCP request failed")
                    response = self._fail(request.get("id"), -32603, str(exc))
            if response is None:
                continue
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
        return 0
