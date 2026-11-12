"""Chunking, BM25 indexing, retrieval and the licence gate."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from nix_llama.rag.corpus import (
    LICENSE_ALLOWLIST,
    SHARE_ALIKE,
    LicenseError,
    Source,
    attribution,
    build_corpus,
    load_manifest,
)
from nix_llama.rag.index import (
    build_index,
    chunk_document,
    index_paths,
    load_index,
    tokenize,
)
from nix_llama.rag.retrieve import format_context, search, should_retrieve

NIXOS_DOC = """# NixOS services

## services.openssh.enable

Whether to enable the OpenSSH secure shell daemon, which allows secure remote logins
over an encrypted channel. Set `services.openssh.settings.PermitRootLogin` to control
root access.

## services.nginx.enable

Whether to enable the nginx web server and reverse proxy.
"""

HYPRLAND_DOC = """# Hyprland variables

## bindm

Mouse bindings. Use `bindm = SUPER, mouse:272, movewindow` to drag windows with the
super key held down. This is distinct from `bind`, which handles key bindings.
"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "nixos.md").write_text(NIXOS_DOC, encoding="utf-8")
    (root / "hyprland.md").write_text(HYPRLAND_DOC, encoding="utf-8")
    return root


@pytest.fixture
def index(corpus_dir: Path):
    chunks = index_paths([corpus_dir], source="docs", license="MIT")
    return build_index(chunks, {"docs": {"license": "MIT", "chunks": str(len(chunks))}})


def test_tokenizer_keeps_dotted_identifiers_and_their_parts() -> None:
    tokens = tokenize("services.openssh.enable")
    # The whole path, so an exact query matches exactly.
    assert "services.openssh.enable" in tokens
    # Its dotted prefixes, so a query for `services.openssh` reaches every
    # option underneath it.
    assert "services.openssh" in tokens
    # And its parts, so "openssh" alone matches too.
    assert "openssh" in tokens
    assert "enable" in tokens


def test_tokenizer_normalises_plurals() -> None:
    # "which options enable sshd" has to reach a document that says "option".
    assert tokenize("options") == tokenize("option")
    assert tokenize("bindings") == tokenize("binding")
    assert tokenize("dependencies") == tokenize("dependency")
    assert tokenize("boxes") == tokenize("box")
    # ...without conflating things that must stay apart.
    assert tokenize("enabled") != tokenize("enable")
    assert tokenize("class") == ["class"]


def test_tokenizer_leaves_option_paths_unstemmed() -> None:
    # `programs.foo.settings` must not become `programs.foo.setting`, or an
    # exact option path would stop matching itself.
    assert "programs.foo.settings" in tokenize("programs.foo.settings")


def test_tokenizer_drops_stopwords_and_single_characters() -> None:
    tokens = tokenize("the a I enable")
    assert tokens == ["enable"]


def test_chunking_splits_on_headings_and_keeps_them_searchable() -> None:
    chunks = chunk_document(NIXOS_DOC, doc_id="d", title="NixOS", source="nixos")
    assert len(chunks) >= 2
    assert any("services.openssh.enable" in c.text for c in chunks)
    # The heading is carried into the chunk body, which is what makes an
    # option-reference page findable by its option name.
    openssh = next(c for c in chunks if "OpenSSH" in c.text)
    assert "services.openssh.enable" in openssh.text


def test_chunking_does_not_split_inside_a_code_fence() -> None:
    text = (
        "# Title\n\n```nix\n"
        "# not a heading, despite the hash\n"
        "services.foo.enable = true;\n"
        "services.foo.settings.port = 8080;\n"
        "```\n"
    )
    chunks = chunk_document(text, doc_id="d", title="Example")
    assert len(chunks) == 1
    assert "# not a heading, despite the hash" in chunks[0].text


def test_retrieval_ranks_the_relevant_chunk_first(index) -> None:
    hits = search(index, "how do I drag windows with the mouse in hyprland", top_k=3)
    assert hits
    assert "bindm" in hits[0].chunk.text


def test_retrieval_finds_an_option_by_its_exact_path(index) -> None:
    hits = search(index, "services.openssh.enable", top_k=2)
    assert hits
    assert "OpenSSH" in hits[0].chunk.text


def test_retrieval_of_an_unknown_term_returns_nothing(index) -> None:
    assert search(index, "zzzzqqqqnonexistentterm", top_k=4) == []


def test_min_score_filters_weak_matches(index) -> None:
    assert search(index, "enable", top_k=4, min_score=1000.0) == []


def test_formatted_context_cites_sources_and_respects_the_budget(index) -> None:
    hits = search(index, "hyprland bindm", top_k=2)
    block = format_context(hits, max_chars=400)
    assert "Prefer them over recollection" in block
    assert "[1]" in block
    assert len(block) < 1200


def test_empty_hits_produce_no_context() -> None:
    assert format_context([]) == ""


def test_a_zero_budget_produces_no_context(index) -> None:
    hits = search(index, "hyprland bindm", top_k=2)
    assert hits
    assert format_context(hits, max_chars=0) == ""


def test_a_budget_smaller_than_the_preamble_produces_no_context(index) -> None:
    # The preamble is charged against the budget; a budget that cannot even
    # hold it would otherwise overflow the context window it was meant to fit.
    hits = search(index, "hyprland bindm", top_k=2)
    assert format_context(hits, max_chars=50) == ""


def test_the_formatted_block_respects_the_total_budget(index) -> None:
    hits = search(index, "hyprland bindm nixos openssh enable service", top_k=4)
    assert hits
    for budget in (400, 800, 2000):
        assert len(format_context(hits, max_chars=budget)) <= budget


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hi", False),
        ("thanks!", False),
        ("ok", False),
        ("nixos", True),  # short, but names something we have docs for
        ("how do I enable sshd", True),
        ("", False),
    ],
)
def test_retrieval_heuristic(query: str, expected: bool) -> None:
    assert should_retrieve(query) is expected


def test_index_round_trips_through_disk(index, tmp_path: Path) -> None:
    path = index.save(tmp_path / "index.json.gz")
    reloaded = load_index(path)
    assert reloaded.size == index.size
    assert reloaded.avg_length == index.avg_length
    assert (
        search(reloaded, "openssh", top_k=1)[0].chunk.id
        == search(index, "openssh", top_k=1)[0].chunk.id
    )


def test_index_is_byte_reproducible(index, tmp_path: Path) -> None:
    first = index.save(tmp_path / "a.json.gz").read_bytes()
    second = index.save(tmp_path / "b.json.gz").read_bytes()
    # No timestamps anywhere: the index has to be a pure function of its input
    # for the Nix derivation that builds it to be cacheable.
    assert first == second


def test_loading_a_future_index_version_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "index.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"version": 999, "chunks": []}, fh)
    with pytest.raises(ValueError, match="is not supported"):
        load_index(path)


def test_directory_traversal_is_sorted_and_therefore_reproducible(corpus_dir: Path) -> None:
    (corpus_dir / "zzz.md").write_text("# Z\n\n" + "z" * 200, encoding="utf-8")
    (corpus_dir / "aaa.md").write_text("# A\n\n" + "a" * 200, encoding="utf-8")
    first = [c.id for c in index_paths([corpus_dir], source="s")]
    second = [c.id for c in index_paths([corpus_dir], source="s")]
    assert first == second
    assert first == sorted(first, key=first.index)  # order is stable, not arbitrary


# --- the licence gate ---------------------------------------------------- #


def test_a_permissive_source_is_accepted(corpus_dir: Path) -> None:
    source = Source(name="docs", license="MIT", root=str(corpus_dir))
    chunks, metadata = build_corpus([source])
    assert chunks
    assert metadata["docs"]["license"] == "MIT"
    assert metadata["docs"]["share_alike"] == "no"


def test_a_disallowed_licence_fails_the_build(corpus_dir: Path) -> None:
    source = Source(name="arch-wiki", license="GFDL-1.3", root=str(corpus_dir))
    with pytest.raises(LicenseError, match="allowlist"):
        build_corpus([source])


def test_proprietary_and_noncommercial_licences_are_not_on_the_allowlist() -> None:
    for forbidden in ("GFDL-1.3", "CC-BY-NC-4.0", "CC-BY-ND-4.0", "PROPRIETARY", "UNKNOWN"):
        assert forbidden.upper() not in LICENSE_ALLOWLIST


def test_share_alike_licences_are_recorded_as_such(corpus_dir: Path) -> None:
    source = Source(name="nix-dev", license="CC-BY-SA-4.0", root=str(corpus_dir))
    _chunks, metadata = build_corpus([source])
    assert metadata["nix-dev"]["share_alike"] == "yes"
    assert "CC-BY-SA-4.0" in SHARE_ALIKE


def test_attribution_names_every_source_and_the_share_alike_obligation() -> None:
    metadata = {
        "mit-source": {
            "license": "MIT",
            "homepage": "https://x",
            "rev": "abc",
            "share_alike": "no",
        },
        "sa-source": {
            "license": "CC-BY-SA-4.0",
            "homepage": "https://y",
            "rev": "def",
            "share_alike": "yes",
        },
    }
    text = attribution(metadata)
    assert "mit-source" in text
    assert "sa-source" in text
    assert "Share-alike notice" in text
    assert "CC-BY-SA-4.0" in text


def test_include_and_exclude_globs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs" / "a.md").write_text("# A\n\n" + "a" * 200, encoding="utf-8")
    (root / "tests" / "b.md").write_text("# B\n\n" + "b" * 200, encoding="utf-8")
    source = Source(
        name="repo",
        license="MIT",
        root=str(root),
        include=["**/*.md"],
        exclude=["tests/**"],
    )
    files = [p.name for p in source.files()]
    assert files == ["a.md"]


def test_url_template_builds_a_citable_link(tmp_path: Path) -> None:
    source = Source(
        name="repo",
        license="MIT",
        root=str(tmp_path),
        rev="deadbeef",
        url_template="https://example.invalid/blob/{rev}/{path}",
    )
    assert source.file_url("docs/a.md") == "https://example.invalid/blob/deadbeef/docs/a.md"


def test_a_source_that_matches_nothing_fails_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="matched no files"):
        build_corpus([Source(name="empty", license="MIT", root=str(empty))])


def test_manifest_round_trip(tmp_path: Path, corpus_dir: Path) -> None:
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps({"sources": [{"name": "docs", "license": "MIT", "root": str(corpus_dir)}]}),
        encoding="utf-8",
    )
    sources = load_manifest(manifest)
    assert [s.name for s in sources] == ["docs"]


def test_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps({"sources": [{"name": "d", "license": "MIT", "root": "/", "nope": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown manifest keys"):
        load_manifest(manifest)


def test_nix_modules_are_indexed_as_individual_options(tmp_path: Path) -> None:
    """A module's options must be retrievable by their own paths.

    Indexing 1400-character windows of surrounding Nix buries the one line
    someone searching for `services.widget.port` actually wants.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "widget.nix").write_text(
        """{ lib, ... }:
        {
          options.services.widget = {
            enable = lib.mkEnableOption "the widget daemon";
            port = lib.mkOption {
              type = lib.types.port;
              default = 8080;
              description = "Port the widget daemon listens on for requests.";
            };
          };
        }
        """,
        encoding="utf-8",
    )
    source = Source(name="widget", license="MIT", root=str(root), include=["**/*.nix"])
    chunks, _metadata = build_corpus([source])
    titles = {c.title for c in chunks}
    assert "services.widget.port" in titles
    assert "services.widget.enable" in titles

    index = build_index(chunks)
    hits = search(index, "services.widget.port", top_k=1)
    assert hits and hits[0].chunk.title == "services.widget.port"
    assert "8080" in hits[0].chunk.text


def test_template_shortcodes_are_stripped_before_indexing(tmp_path: Path) -> None:
    root = tmp_path / "hugo"
    root.mkdir()
    (root / "page.md").write_text(
        "---\ntitle: Rollouts\n---\n\n# Rollouts\n\n"
        '{{% heading "examples" %}}\n\n'
        "{{< tabs >}}\nRestart a deployment with `kubectl rollout restart`, which replaces\n"
        "every pod in a controlled way rather than all at once.\n{{< /tabs >}}\n",
        encoding="utf-8",
    )
    chunks, _ = build_corpus([Source(name="k8s", license="CC-BY-4.0", root=str(root))])
    text = "\n".join(c.text for c in chunks)
    assert "{{" not in text
    assert "title: Rollouts" not in text
    assert "kubectl rollout restart" in text
