"""An OpenAI-compatible reverse proxy that augments requests with retrieved docs.

llama-server listens on loopback; this proxy owns the public port. Chat
completion requests get documentation excerpts injected into the system message
when the question looks like one the corpus can answer. Every other path --
``/v1/models``, ``/health``, ``/metrics``, ``/props``, the web UI, SSE streams
-- is relayed byte-for-byte.

Failure is always non-fatal: if retrieval raises, the original request is
forwarded unchanged. A broken index must never take the model offline.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__
from .index import Index
from .retrieve import format_context, search, should_retrieve

log = logging.getLogger("nix-llama.rag")

# Headers that describe a single hop and must not be relayed.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

CHAT_PATHS = frozenset({"/v1/chat/completions", "/chat/completions", "/api/chat/completions"})

# Rough character-per-token ratio for English prose and code. Used only to
# bound the injected context -- the proxy deliberately does not tokenise, since
# that would mean loading the model's vocabulary into a second process.
CHARS_PER_TOKEN = 3.5

# The largest share of the context window retrieval may claim. Retrieval that
# fills the window is worse than no retrieval: the request fails outright
# instead of being answered from the model's own knowledge.
CONTEXT_FRACTION = 0.25


def context_budget(
    context_tokens: int | None,
    configured_max_chars: int,
    reserved_chars: int = 0,
) -> int:
    """Clamp the injected-context budget to a share of the context window.

    ``reserved_chars`` accounts for text the proxy injects unconditionally --
    the default system prompt -- which competes for the same window.
    """
    if not context_tokens:
        return configured_max_chars
    allowed = int(context_tokens * CHARS_PER_TOKEN * CONTEXT_FRACTION) - reserved_chars
    return max(0, min(configured_max_chars, allowed))


# Bodies larger than this are relayed without inspection. A multi-megabyte
# request is a file upload or a pasted repository, not a question needing docs.
MAX_INSPECT_BYTES = 8 * 1024 * 1024


class RagState:
    """Shared, hot-reloadable retrieval state."""

    def __init__(
        self,
        index: Index | None,
        *,
        top_k: int = 4,
        max_chars: int = 6000,
        min_score: float = 1.0,
        system_prompt: str | None = None,
        inject_into: str = "system",
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.max_chars = max_chars
        self.min_score = min_score
        self.system_prompt = system_prompt
        self.inject_into = inject_into
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "augmented": 0, "errors": 0}

    def retrieve(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        if self.index is None or not query:
            return "", []
        hits = search(self.index, query, top_k=self.top_k, min_score=self.min_score)
        if not hits:
            return "", []
        return format_context(hits, max_chars=self.max_chars), [h.as_dict() for h in hits]


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # The multimodal content form: a list of typed parts.
            return "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "input_text")
            )
    return ""


def augment(
    payload: dict[str, Any], state: RagState
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inject a default system prompt and retrieved context into a chat request."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return payload, []

    if state.system_prompt and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": state.system_prompt})

    query = _last_user_text(messages)
    if not should_retrieve(query):
        return payload, []
    block, hits = state.retrieve(query)
    if not block:
        return payload, []

    if state.inject_into == "user":
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                message["content"] = f"{block}\n\n---\n\n{message['content']}"
                break
    else:
        for message in messages:
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                message["content"] = f"{message['content']}\n\n{block}"
                break
        else:
            messages.insert(0, {"role": "system", "content": block})
    payload["messages"] = messages
    return payload, hits


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"nix-llama/{__version__}"
    # Populated by serve().
    upstream: tuple[str, int] = ("127.0.0.1", 8081)
    state: RagState | None = None
    timeout = 3600

    # -- plumbing ---------------------------------------------------------- #

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if length is not None:
            try:
                return self.rfile.read(int(length))
            except (ValueError, OSError):
                return b""
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            chunks = bytearray()
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if size == 0:
                    self.rfile.readline()
                    break
                chunks += self.rfile.read(size)
                self.rfile.readline()
            return bytes(chunks)
        return b""

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- request entry points ---------------------------------------------- #

    def do_GET(self) -> None:
        if self._handle_local(b""):
            return
        self._relay("GET", b"")

    def do_POST(self) -> None:
        body = self._read_body()
        if self._handle_local(body):
            return
        path = urlparse(self.path).path
        if path in CHAT_PATHS and self.state is not None and len(body) <= MAX_INSPECT_BYTES:
            body = self._augment_body(body)
        self._relay("POST", body)

    def do_DELETE(self) -> None:
        self._relay("DELETE", self._read_body())

    def do_PUT(self) -> None:
        self._relay("PUT", self._read_body())

    def do_OPTIONS(self) -> None:
        self._relay("OPTIONS", b"")

    def do_HEAD(self) -> None:
        self._relay("HEAD", b"")

    # -- behaviour --------------------------------------------------------- #

    def _augment_body(self, body: bytes) -> bytes:
        state = self.state
        assert state is not None
        state.stats["requests"] += 1
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                return body
            payload, hits = augment(payload, state)
            if hits:
                state.stats["augmented"] += 1
                log.info(
                    "augmented request with %d excerpt(s): %s",
                    len(hits),
                    ", ".join(h["id"] for h in hits),
                )
            return json.dumps(payload).encode("utf-8")
        except Exception:
            state.stats["errors"] += 1
            log.exception("retrieval failed; forwarding the request unchanged")
            return body

    def _handle_local(self, body: bytes) -> bool:
        """Serve the proxy's own endpoints. Returns True if handled."""
        parsed = urlparse(self.path)
        if parsed.path not in ("/v1/rag/search", "/rag/search", "/v1/rag/stats"):
            return False
        state = self.state
        if state is None or state.index is None:
            self._json(503, {"error": "retrieval is not configured"})
            return True
        if parsed.path.endswith("/stats"):
            self._json(
                200,
                {
                    "chunks": state.index.size,
                    "sources": state.index.sources,
                    "top_k": state.top_k,
                    **state.stats,
                },
            )
            return True
        query, top_k = "", state.top_k
        if body:
            try:
                data = json.loads(body)
                query = str(data.get("query") or data.get("q") or "")
                top_k = int(data.get("top_k") or top_k)
            except (ValueError, TypeError):
                self._json(400, {"error": "body must be JSON with a 'query' field"})
                return True
        else:
            params = parse_qs(parsed.query)
            query = (params.get("query") or params.get("q") or [""])[0]
            top_k = int((params.get("top_k") or [str(top_k)])[0])
        hits = search(state.index, query, top_k=top_k, min_score=0.0)
        self._json(200, {"query": query, "results": [h.as_dict() for h in hits]})
        return True

    def _relay(self, method: str, body: bytes) -> None:
        host, port = self.upstream
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() != "content-length"
        }
        if body:
            headers["Content-Length"] = str(len(body))
        try:
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
            conn.request(method, self.path, body=body or None, headers=headers)
            response = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            log.error("upstream llama-server unreachable: %s", exc)
            self._json(502, {"error": f"upstream llama-server unreachable: {exc}"})
            return

        try:
            self.send_response(response.status, response.reason)
            streaming = False
            for key, value in response.getheaders():
                lowered = key.lower()
                if lowered in HOP_BY_HOP:
                    continue
                if lowered == "content-length":
                    continue
                if lowered == "content-type" and "text/event-stream" in value.lower():
                    streaming = True
                self.send_header(key, value)
            length = response.getheader("Content-Length")
            if streaming or length is None:
                # Relay token-by-token so the client sees the same latency it
                # would talking to llama-server directly.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self._pump_chunked(response)
            else:
                self.send_header("Content-Length", length)
                self.end_headers()
                if method != "HEAD":
                    self.wfile.write(response.read())
        except (BrokenPipeError, ConnectionResetError):
            log.debug("client disconnected mid-response")
        finally:
            conn.close()

    def _pump_chunked(self, response: http.client.HTTPResponse) -> None:
        try:
            while True:
                data = response.read(4096)
                if not data:
                    break
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log.debug("client disconnected mid-stream")


class Proxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    address_family = socket.AF_INET


def serve(
    host: str,
    port: int,
    upstream_host: str,
    upstream_port: int,
    state: RagState,
) -> Proxy:
    """Start the proxy. Returns the server; call ``serve_forever`` on it."""
    handler = type(
        "BoundHandler",
        (Handler,),
        {"upstream": (upstream_host, upstream_port), "state": state},
    )
    try:
        server = Proxy((host, port), handler)
    except OSError as exc:
        raise OSError(f"cannot bind {host}:{port}: {exc}") from exc
    log.info(
        "RAG proxy listening on %s:%d -> %s:%d (%s)",
        host,
        port,
        upstream_host,
        upstream_port,
        f"{state.index.size} chunks indexed" if state.index else "retrieval disabled",
    )
    return server
