"""The augmenting proxy: request rewriting and transparent relaying."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from nix_llama.rag.index import build_index, index_paths
from nix_llama.rag.proxy import HOP_BY_HOP, RagState, augment, context_budget, serve


@pytest.fixture
def index(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "nixos.md").write_text(
        "# services.openssh.enable\n\nWhether to enable the OpenSSH daemon, which allows "
        "secure remote logins over an encrypted channel.\n",
        encoding="utf-8",
    )
    return build_index(index_paths([root], source="nixos", license="MIT"), {"nixos": {}})


@pytest.fixture
def state(index) -> RagState:
    return RagState(index, top_k=2, min_score=0.0)


# --- request rewriting ---------------------------------------------------- #


def test_documentation_is_injected_into_the_system_message(state: RagState) -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "how do I enable openssh in NixOS?"},
        ]
    }
    rewritten, hits = augment(payload, state)
    assert hits
    system = rewritten["messages"][0]["content"]
    assert system.startswith("You are helpful.")
    assert "OpenSSH" in system
    # The user's own message is untouched.
    assert rewritten["messages"][1]["content"] == "how do I enable openssh in NixOS?"


def test_a_system_message_is_created_when_the_client_sends_none(state: RagState) -> None:
    payload = {"messages": [{"role": "user", "content": "how do I enable openssh?"}]}
    rewritten, hits = augment(payload, state)
    assert hits
    assert rewritten["messages"][0]["role"] == "system"


def test_injection_into_the_user_message(index) -> None:
    state = RagState(index, top_k=2, min_score=0.0, inject_into="user")
    payload = {"messages": [{"role": "user", "content": "how do I enable openssh?"}]}
    rewritten, _ = augment(payload, state)
    assert "OpenSSH" in rewritten["messages"][0]["content"]
    assert rewritten["messages"][0]["content"].endswith("how do I enable openssh?")


def test_the_default_system_prompt_is_injected_once(index) -> None:
    state = RagState(index, system_prompt="You are nix-llama.", min_score=0.0)
    payload = {"messages": [{"role": "user", "content": "how do I enable openssh?"}]}
    rewritten, _ = augment(payload, state)
    systems = [m for m in rewritten["messages"] if m["role"] == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].startswith("You are nix-llama.")


def test_a_clients_own_system_prompt_is_respected(index) -> None:
    state = RagState(index, system_prompt="You are nix-llama.", min_score=0.0)
    payload = {
        "messages": [
            {"role": "system", "content": "You are something else."},
            {"role": "user", "content": "how do I enable openssh?"},
        ]
    }
    rewritten, _ = augment(payload, state)
    assert rewritten["messages"][0]["content"].startswith("You are something else.")
    assert "You are nix-llama." not in rewritten["messages"][0]["content"]


def test_chit_chat_is_not_augmented(state: RagState) -> None:
    payload = {"messages": [{"role": "user", "content": "thanks!"}]}
    rewritten, hits = augment(payload, state)
    assert hits == []
    assert len(rewritten["messages"]) == 1


def test_a_query_with_no_matches_is_not_augmented(state: RagState) -> None:
    payload = {"messages": [{"role": "user", "content": "qqzz nonexistent xyzzy term"}]}
    _rewritten, hits = augment(payload, state)
    assert hits == []


def test_multimodal_content_lists_are_understood(state: RagState) -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "how do I enable openssh in NixOS?"}],
            }
        ]
    }
    _rewritten, hits = augment(payload, state)
    assert hits


def test_a_request_without_messages_is_passed_through(state: RagState) -> None:
    payload = {"prompt": "raw completion"}
    rewritten, hits = augment(payload, state)
    assert rewritten == payload
    assert hits == []


def test_retrieval_disabled_means_no_rewriting() -> None:
    state = RagState(None)
    payload = {"messages": [{"role": "user", "content": "how do I enable openssh?"}]}
    rewritten, hits = augment(payload, state)
    assert hits == []
    assert rewritten["messages"] == [{"role": "user", "content": "how do I enable openssh?"}]


def test_the_retrieval_budget_is_capped_by_the_context_window() -> None:
    # A 2048-token context cannot afford 6000 characters of documentation: the
    # request would be rejected for exceeding the context instead of answered.
    assert context_budget(2048, 6000) == int(2048 * 3.5 * 0.25)
    # A large context leaves the configured budget alone.
    assert context_budget(32768, 6000) == 6000
    # An unknown context size means no clamping to apply.
    assert context_budget(None, 6000) == 6000
    # The unconditional system prompt competes for the same window.
    assert context_budget(2048, 6000, reserved_chars=1000) == int(2048 * 3.5 * 0.25) - 1000
    # When nothing is left, retrieval goes inert rather than overflowing.
    assert context_budget(512, 6000, reserved_chars=5000) == 0


# --- relaying ------------------------------------------------------------- #


class FakeUpstream(BaseHTTPRequestHandler):
    """Stands in for llama-server, recording what it was sent."""

    received: dict[str, Any] = {}
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        FakeUpstream.received = {"method": "GET", "path": self.path, "headers": dict(self.headers)}
        body = json.dumps({"models": [{"name": "fake"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            # llama-server would answer 400; what matters for the proxy test is
            # that the bytes arrived unmodified.
            parsed = None
        FakeUpstream.received = {
            "method": "POST",
            "path": self.path,
            "headers": dict(self.headers),
            "json": parsed,
            "raw": raw,
        }
        if self.path.endswith("/stream"):
            # Real llama-server chunks its SSE stream; emulate that so the
            # client sees a terminated response.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                payload = f"data: chunk{i}\n\n".encode()
                self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
                self.wfile.flush()
            payload = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stack(state: RagState):
    """A live proxy in front of a live fake upstream."""
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    upstream_port = upstream.server_address[1]
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    proxy = serve("127.0.0.1", 0, "127.0.0.1", upstream_port, state)
    proxy_port = proxy.server_address[1]
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{proxy_port}"
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def post(url: str, payload: dict, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read()


def test_unrelated_paths_are_relayed(stack: str) -> None:
    status, body, _ = get(f"{stack}/v1/models")
    assert status == 200
    assert json.loads(body)["models"][0]["name"] == "fake"
    assert FakeUpstream.received["path"] == "/v1/models"


def test_authorization_headers_reach_the_upstream(stack: str) -> None:
    get(f"{stack}/v1/models", headers={"Authorization": "Bearer secret"})
    assert FakeUpstream.received["headers"]["Authorization"] == "Bearer secret"


def test_hop_by_hop_headers_are_not_relayed(stack: str) -> None:
    get(f"{stack}/v1/models", headers={"X-Keep": "yes"})
    relayed = {k.lower() for k in FakeUpstream.received["headers"]}
    assert "x-keep" in relayed
    # `connection` is regenerated per hop by the client library; the proxy must
    # never copy the inbound value through.
    assert not (relayed & (HOP_BY_HOP - {"connection", "host"}))


def test_chat_requests_arrive_at_the_upstream_augmented(stack: str) -> None:
    status, _ = post(
        f"{stack}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "how do I enable openssh in NixOS?"}]},
    )
    assert status == 200
    sent = FakeUpstream.received["json"]
    assert sent["messages"][0]["role"] == "system"
    assert "OpenSSH" in sent["messages"][0]["content"]


def test_a_malformed_chat_body_is_relayed_unchanged(stack: str) -> None:
    request = urllib.request.Request(
        f"{stack}/v1/chat/completions",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 200
    # Retrieval must never be the reason a request fails, and the bytes must
    # reach llama-server exactly as the client sent them.
    assert FakeUpstream.received["json"] is None
    assert FakeUpstream.received["raw"] == b"{not json"


def test_streaming_responses_are_relayed(stack: str) -> None:
    request = urllib.request.Request(
        f"{stack}/stream",
        data=json.dumps({"stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert "text/event-stream" in response.headers["Content-Type"]
        data = response.read()
    assert b"data: chunk0" in data
    assert b"data: [DONE]" in data


def test_the_search_endpoint_is_served_locally(stack: str) -> None:
    status, body, _ = get(f"{stack}/v1/rag/search?q=openssh")
    assert status == 200
    results = json.loads(body)["results"]
    assert results and "OpenSSH" in results[0]["text"]


def test_the_search_endpoint_accepts_a_json_body(stack: str) -> None:
    status, body = post(f"{stack}/v1/rag/search", {"query": "openssh", "top_k": 1})
    assert status == 200
    assert len(json.loads(body)["results"]) == 1


def test_the_stats_endpoint_counts_augmented_requests(stack: str) -> None:
    post(
        f"{stack}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "how do I enable openssh in NixOS?"}]},
    )
    _status, body, _ = get(f"{stack}/v1/rag/stats")
    stats = json.loads(body)
    assert stats["chunks"] >= 1
    assert stats["requests"] >= 1
    assert stats["augmented"] >= 1
    assert stats["errors"] == 0


def test_an_unreachable_upstream_is_a_502(state: RagState) -> None:
    # Port 1 is reserved and nothing listens there.
    proxy = serve("127.0.0.1", 0, "127.0.0.1", 1, state)
    port = proxy.server_address[1]
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        status, body, _ = get(f"http://127.0.0.1:{port}/v1/models")
        assert status == 502
        assert b"unreachable" in body
    finally:
        proxy.shutdown()
        proxy.server_close()
