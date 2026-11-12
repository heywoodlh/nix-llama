"""The MCP server, driven over its real JSON-RPC transport."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nix_llama.rag.index import build_index, index_paths
from nix_llama.rag.mcp import PROTOCOL_VERSION, TOOLS, McpServer


@pytest.fixture
def server(tmp_path: Path) -> McpServer:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "nixos.md").write_text(
        "# services.openssh.enable\n\nWhether to enable the OpenSSH daemon for secure "
        "remote logins over an encrypted channel.\n",
        encoding="utf-8",
    )
    chunks = index_paths([root], source="nixos", license="MIT")
    index = build_index(chunks, {"nixos": {"license": "MIT", "chunks": str(len(chunks))}})
    return McpServer(index)


def run(server: McpServer, requests: list[dict]) -> list[dict]:
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    server.run(stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_initialize_reports_tool_capability(server: McpServer) -> None:
    (response,) = run(server, [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert response["result"]["capabilities"]["tools"] == {"listChanged": False}
    assert response["result"]["serverInfo"]["name"] == "nix-llama-docs"
    assert "search_docs" in response["result"]["instructions"]


def test_initialize_echoes_a_supported_protocol_version(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        ],
    )
    assert response["result"]["protocolVersion"] == "2024-11-05"


def test_notifications_get_no_response(server: McpServer) -> None:
    assert run(server, [{"jsonrpc": "2.0", "method": "notifications/initialized"}]) == []


def test_tools_list_matches_the_declared_tools(server: McpServer) -> None:
    (response,) = run(server, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    names = [t["name"] for t in response["result"]["tools"]]
    assert names == [t["name"] for t in TOOLS]
    for tool in response["result"]["tools"]:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


def test_search_returns_cited_excerpts(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_docs", "arguments": {"query": "openssh enable"}},
            }
        ],
    )
    text = response["result"]["content"][0]["text"]
    assert "OpenSSH" in text
    assert "licence: MIT" in text
    assert not response["result"].get("isError")


def test_search_with_no_match_says_so_instead_of_inventing(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_docs", "arguments": {"query": "qqzzxnonexistent"}},
            }
        ],
    )
    text = response["result"]["content"][0]["text"]
    assert "No documentation matched" in text
    assert "rather than inventing" in text


def test_search_without_a_query_is_an_error(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_docs", "arguments": {}},
            }
        ],
    )
    assert response["result"]["isError"] is True


def test_top_k_is_clamped(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_docs",
                    "arguments": {"query": "openssh", "top_k": 9999},
                },
            }
        ],
    )
    assert not response["result"].get("isError")


def test_list_sources_reports_licences(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_doc_sources", "arguments": {}},
            }
        ],
    )
    assert "license=MIT" in response["result"]["content"][0]["text"]


def test_unknown_tool_is_a_protocol_error(server: McpServer) -> None:
    (response,) = run(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "rm_rf", "arguments": {}},
            }
        ],
    )
    assert response["error"]["code"] == -32602


def test_unknown_method_is_a_protocol_error(server: McpServer) -> None:
    (response,) = run(server, [{"jsonrpc": "2.0", "id": 1, "method": "resources/list"}])
    assert response["error"]["code"] == -32601


def test_malformed_json_does_not_kill_the_session(server: McpServer) -> None:
    stdin = io.StringIO('{not json\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n')
    stdout = io.StringIO()
    server.run(stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == -32700
    assert responses[1]["result"] == {}


def test_server_without_an_index_explains_itself() -> None:
    (response,) = run(
        McpServer(None),
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_docs", "arguments": {"query": "anything"}},
            }
        ],
    )
    assert response["result"]["isError"] is True
    assert "nix-llama-rag index" in response["result"]["content"][0]["text"]


def test_batched_requests(server: McpServer) -> None:
    stdin = io.StringIO(
        json.dumps(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    server.run(stdin, stdout)
    batch = json.loads(stdout.getvalue())
    assert [r["id"] for r in batch] == [1, 2]
