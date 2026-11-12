"""Corpus chunking and BM25 index construction.

The on-disk format is gzipped JSON with sorted keys and no timestamps, so
building the same corpus twice produces byte-identical output -- a hard
requirement for the index to live inside a Nix derivation.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

INDEX_VERSION = 3

# Target chunk size in characters. Big enough to carry a whole option
# description or config snippet, small enough that four of them fit in a
# prompt without crowding out the user's actual code.
CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200
# Minimum length of a finished chunk, heading included. Option-reference pages
# are full of one-sentence descriptions ("Whether to enable the nginx web
# server.") that are short but exactly what someone searches for, so the floor
# only has to exclude genuine fragments.
MIN_CHUNK_CHARS = 60

# BM25 parameters; the usual Robertson/Sparck-Jones defaults.
BM25_K1 = 1.2
BM25_B = 0.75

# Field weighting: a chunk's title is counted this many extra times when the
# index is built. A query like "home-manager programs.git settings" otherwise
# loses to a long introduction page that happens to say "home-manager" twenty
# times, because BM25 has no notion of which part of a document a term is in.
# Titles here are option paths and section headings -- exactly what people
# search by -- so they earn the weight.
TITLE_WEIGHT = 3

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*")
_SPLIT_RE = re.compile(r"[._+-]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "it's",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "i",
        "me",
        "my",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "can",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "s",
        "t",
        "don",
        "now",
    ]
)


_FRONT_MATTER_LIMIT = 4000
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Hugo and Docusaurus shortcodes: `{{< tabs >}}`, `{{% heading "examples" %}}`.
# Kubernetes, GitHub and GitLab documentation are full of them, and left in
# place they pollute both chunk titles and retrieved text.
_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def clean_markup(text: str) -> str:
    """Strip front matter, HTML comments and template shortcodes.

    Used by both the retrieval index and the training dataset, so a chunk the
    model retrieves looks like what it was trained on.
    """
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if 0 < end < _FRONT_MATTER_LIMIT:
            text = text[end + 4 :]
    text = _HTML_COMMENT_RE.sub("", text)
    text = _SHORTCODE_RE.sub("", text)
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


# Minimum stem length. "keys" -> "key" is worth having; "yes" -> "ye" is not.
_MIN_STEM = 3
# Plural endings that need the whole "es" removed, per the English rule.
_ES_STEMS = ("s", "x", "z", "ch", "sh")


def stem(token: str) -> str:
    """Normalise an English plural. Nothing else.

    Without this, "which options enable sshd" misses every document that says
    "option". Verb inflections (-ing, -ed) were tried and removed: no rule
    short of a real stemmer maps "enabled" and "enable" together without also
    mapping things apart that must match, and a stemmer that disagrees with
    itself is worse than none.

    Applied identically when indexing and when querying, so the two sides
    always agree.
    """
    if "." in token:
        # An option path or a filename; mangling it would break exact matching.
        return token
    if token.endswith("ies") and len(token) - 3 >= _MIN_STEM:
        return token[:-3] + "y"
    if (
        token.endswith("es")
        and len(token) - 2 >= _MIN_STEM
        and any(token[:-2].endswith(e) for e in _ES_STEMS)
    ):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) - 1 >= _MIN_STEM:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, identifier-aware, lightly stemmed tokenisation.

    ``services.nix-llama.enable`` yields the full dotted token *and* its parts,
    so a query for either the whole option path or just ``nix-llama`` matches.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("._+-")
        if not token or len(token) < 2:
            continue
        if token not in STOPWORDS:
            tokens.append(stem(token))
        # Dotted prefixes, so a query for `programs.git` reaches every
        # `programs.git.*` option rather than only the exact path typed.
        segments = token.split(".")
        if len(segments) > 2:
            for i in range(2, len(segments)):
                tokens.append(".".join(segments[:i]))
        parts = [p for p in _SPLIT_RE.split(token) if len(p) >= 2 and p not in STOPWORDS]
        if len(parts) > 1:
            tokens.extend(stem(p) for p in parts)
    return tokens


@dataclass
class Chunk:
    """One retrievable passage."""

    id: str
    text: str
    title: str = ""
    source: str = ""
    license: str = ""
    url: str = ""
    path: str = ""

    def citation(self) -> str:
        bits = [b for b in (self.source, self.title) if b]
        label = " / ".join(bits) if bits else self.id
        return f"{label} <{self.url}>" if self.url else label


@dataclass
class Index:
    """A BM25 index over a chunk collection."""

    version: int = INDEX_VERSION
    chunks: list[Chunk] = field(default_factory=list)
    # term -> [[chunk_index, term_frequency], ...]
    postings: dict[str, list[list[int]]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avg_length: float = 0.0
    sources: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        postings = self.postings.get(term)
        if not postings:
            return 0.0
        n, df = len(self.chunks), len(postings)
        # BM25 probabilistic idf, floored so a term in almost every document
        # cannot contribute a negative score.
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "chunks": [asdict(c) for c in self.chunks],
            "postings": self.postings,
            "lengths": self.lengths,
            "avg_length": self.avg_length,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Index:
        version = int(data.get("version", 0))
        if version != INDEX_VERSION:
            raise ValueError(
                f"index format version {version} is not supported "
                f"(expected {INDEX_VERSION}); rebuild with `nix-llama-rag index`"
            )
        return cls(
            version=version,
            chunks=[Chunk(**c) for c in data.get("chunks", [])],
            postings={
                k: [list(map(int, p)) for p in v] for k, v in data.get("postings", {}).items()
            },
            lengths=list(map(int, data.get("lengths", []))),
            avg_length=float(data.get("avg_length", 0.0)),
            sources=data.get("sources", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        # mtime=0 keeps the gzip header free of timestamps, which keeps the
        # output bit-for-bit reproducible.
        with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as fh:
            fh.write(payload.encode("utf-8"))
        return path


def load_index(path: str | Path) -> Index:
    path = Path(path)
    if path.is_dir():
        path = path / "index.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Index.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def _split_sections(text: str) -> Iterator[tuple[str, str]]:
    """Split markdown-ish text on headings, yielding ``(heading, body)``."""
    heading = ""
    buffer: list[str] = []
    in_code = False
    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_code = not in_code
        match = None if in_code else _HEADING_RE.match(line)
        if match:
            if any(b.strip() for b in buffer):
                yield heading, "\n".join(buffer)
            heading = match.group(2)
            buffer = []
        else:
            buffer.append(line)
    if any(b.strip() for b in buffer):
        yield heading, "\n".join(buffer)


def _window(text: str) -> Iterator[str]:
    """Slice an over-long section into overlapping windows on blank lines."""
    if len(text) <= CHUNK_CHARS:
        yield text
        return
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    for para in paragraphs:
        if len(para) > CHUNK_CHARS:
            if current:
                yield current
                current = ""
            for i in range(0, len(para), CHUNK_CHARS - CHUNK_OVERLAP):
                yield para[i : i + CHUNK_CHARS]
            continue
        if len(current) + len(para) + 2 > CHUNK_CHARS and current:
            yield current
            # Carry the tail forward so a fact split across the boundary is
            # still retrievable from at least one chunk.
            current = current[-CHUNK_OVERLAP:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        yield current


def chunk_document(
    text: str,
    *,
    doc_id: str,
    title: str = "",
    source: str = "",
    license: str = "",
    url: str = "",
    path: str = "",
) -> list[Chunk]:
    """Break one document into retrievable chunks."""
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        for piece in _window(body):
            piece = piece.strip()
            full_title = " > ".join(t for t in (title, heading) if t)
            # Length is measured on the finished chunk: a terse description
            # under a descriptive heading is still worth retrieving.
            if len(piece) + len(full_title) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{len(chunks)}",
                    # The heading is prepended so it is searchable alongside
                    # the body, which matters a lot for option reference pages.
                    text=f"{full_title}\n\n{piece}" if full_title else piece,
                    title=heading or title,
                    source=source,
                    license=license,
                    url=url,
                    path=path,
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def build_index(chunks: Iterable[Chunk], sources: dict[str, dict[str, str]] | None = None) -> Index:
    """Build a BM25 index. Deterministic for a given chunk ordering."""
    index = Index(chunks=list(chunks), sources=dict(sources or {}))
    postings: dict[str, dict[int, int]] = {}
    for i, chunk in enumerate(index.chunks):
        tokens = tokenize(chunk.text)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        # The title is already part of `text`; these are the extra copies.
        for token in tokenize(chunk.title) * (TITLE_WEIGHT - 1):
            counts[token] = counts.get(token, 0) + 1
        # Document length stays the body length: inflating it with the title
        # copies would cancel out the weighting through BM25's normalisation.
        index.lengths.append(len(tokens))
        for term, tf in counts.items():
            postings.setdefault(term, {})[i] = tf
    index.postings = {
        term: [[doc, tf] for doc, tf in sorted(docs.items())]
        for term, docs in sorted(postings.items())
    }
    index.avg_length = (sum(index.lengths) / len(index.lengths)) if index.lengths else 0.0
    return index


def index_paths(
    paths: Iterable[str | Path],
    *,
    source: str = "local",
    license: str = "unknown",
    suffixes: tuple[str, ...] = (".md", ".markdown", ".txt", ".rst", ".adoc", ".nix", ".org"),
) -> list[Chunk]:
    """Chunk every matching file under ``paths``.

    Files are visited in sorted order so the resulting index is reproducible.
    """
    chunks: list[Chunk] = []
    for raw in paths:
        root = Path(raw)
        files = (
            sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
            if root.is_dir()
            else [root]
        )
        for file in files:
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(file.relative_to(root)) if root.is_dir() else file.name
            chunks.extend(
                chunk_document(
                    text,
                    doc_id=f"{source}:{rel}",
                    title=file.stem.replace("-", " ").replace("_", " "),
                    source=source,
                    license=license,
                    path=rel,
                )
            )
    return chunks
