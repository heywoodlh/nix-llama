"""BM25 query execution and prompt formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import BM25_B, BM25_K1, Chunk, Index, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "id": self.chunk.id,
            "title": self.chunk.title,
            "source": self.chunk.source,
            "license": self.chunk.license,
            "url": self.chunk.url,
            "text": self.chunk.text,
        }


# A dotted query token is an attribute path -- `services.openssh.enable`,
# `system.defaults.dock` -- and someone who types one means exactly it. The
# ordinary words around it ("which", "option", "settings") are context.
#
# Only dots count. Hyphens were tried and made things worse: "nix-darwin" and
# "fail-fast" are hyphenated *words*, so boosting them promoted whichever
# README says the project's own name most often.
PATH_WEIGHT = 2.5


def _term_weight(term: str) -> float:
    return PATH_WEIGHT if "." in term else 1.0


def search(index: Index, query: str, top_k: int = 4, min_score: float = 0.0) -> list[Hit]:
    """Return the ``top_k`` highest scoring chunks for ``query``."""
    terms = tokenize(query)
    if not terms or not index.chunks:
        return []
    scores: dict[int, float] = {}
    avgdl = index.avg_length or 1.0
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        idf = index.idf(term)
        if idf <= 0.0:
            continue
        weight = idf * _term_weight(term)
        for doc, tf in index.postings.get(term, ()):
            length = index.lengths[doc] if doc < len(index.lengths) else int(avgdl)
            norm = tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avgdl))
            scores[doc] = scores.get(doc, 0.0) + weight * norm
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    hits = [Hit(chunk=index.chunks[doc], score=score) for doc, score in ranked[:top_k]]
    return [h for h in hits if h.score >= min_score]


# Minimum useful excerpt: below this a truncated chunk is noise rather than
# evidence, so it is dropped instead of included.
MIN_EXCERPT_CHARS = 80
_TRUNCATION_MARK = "\n[excerpt truncated]"


def format_context(hits: list[Hit], max_chars: int = 6000) -> str:
    """Render hits as a compact, attributed block for the system prompt.

    The returned string is never longer than ``max_chars``, separators and
    preamble included. That matters: llama-server rejects a request that
    exceeds the context window outright rather than truncating it, so an
    over-budget block costs the user an error instead of a longer answer.
    """
    if not hits or max_chars <= 0:
        return ""

    preamble = (
        "The following documentation excerpts were retrieved for this request. "
        "Prefer them over recollection, cite the source when you rely on one, and say so "
        "plainly if they do not answer the question."
    )
    separator = "\n\n"
    budget = max_chars - len(preamble)
    if budget <= len(separator) + MIN_EXCERPT_CHARS:
        return ""

    parts = [preamble]
    for i, hit in enumerate(hits, start=1):
        available = budget - len(separator)
        header = f"[{i}] {hit.chunk.citation()}"
        body = hit.chunk.text.strip()
        block = f"{header}\n{body}"
        if len(block) > available:
            keep = available - len(header) - 1 - len(_TRUNCATION_MARK)
            if keep < MIN_EXCERPT_CHARS:
                break
            block = f"{header}\n{body[:keep]}{_TRUNCATION_MARK}"
        parts.append(block)
        budget -= len(block) + len(separator)
        if budget <= len(separator) + MIN_EXCERPT_CHARS:
            break
    if len(parts) == 1:
        return ""
    return separator.join(parts)


# Requests that obviously do not need documentation: pure code transformation,
# chit-chat, or anything already carrying a large body of pasted context. The
# heuristic errs toward retrieving, since an unused excerpt costs a few hundred
# tokens while a missing one costs a wrong answer.
_SKIP_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|continue|go on)\b[\s!.?]*$",
    re.IGNORECASE,
)

_DOMAIN_TERMS = frozenset(
    [
        "nix",
        "nixos",
        "nixpkgs",
        "flake",
        "flakes",
        "home-manager",
        "nix-darwin",
        "darwin",
        "derivation",
        "overlay",
        "systemd",
        "hyprland",
        "niri",
        "wayland",
        "hypr",
        "waybar",
        "kubernetes",
        "kubectl",
        "k8s",
        "helm",
        "kustomize",
        "argocd",
        "pod",
        "deployment",
        "statefulset",
        "ingress",
        "proton",
        "steam",
        "wine",
        "lutris",
        "dxvk",
        "vkd3d",
        "gamescope",
        "launchd",
        "homebrew",
        "brew",
        "plist",
        "macos",
        "ci",
        "cd",
        "github",
        "gitlab",
        "codeberg",
        "tangled",
        "actions",
        "workflow",
        "runner",
        "pipeline",
        "forgejo",
        "woodpecker",
        "systemctl",
        "journalctl",
        "nixos-rebuild",
        "darwin-rebuild",
        "option",
        "options",
        "module",
        "modules",
    ]
)


def should_retrieve(query: str, min_words: int = 3) -> bool:
    """Decide whether a user turn is worth a documentation lookup."""
    text = (query or "").strip()
    if not text or _SKIP_RE.match(text):
        return False
    words = text.split()
    if len(words) < min_words:
        # Very short turns are only worth a lookup if they name something we
        # actually have documentation about.
        return any(w.strip(".,:;!?`").lower() in _DOMAIN_TERMS for w in words)
    return True
