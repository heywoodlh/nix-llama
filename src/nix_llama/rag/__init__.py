"""Retrieval-augmented generation for nix-llama.

A small model that has been taught Nix, NixOS and friends still cannot know
what upstream changed yesterday. The RAG layer closes that gap with a lexical
index over permissively licensed documentation, exposed three ways:

* ``nix-llama-rag search`` -- command line lookup
* ``nix-llama-rag serve``  -- an OpenAI-compatible proxy that injects retrieved
  passages into requests whose answer plausibly lives in the docs
* ``nix-llama-rag mcp``    -- an MCP server, so a harness can let the model ask
  for documentation itself when it does not know something

Retrieval is BM25, not embeddings. That is a deliberate trade: no second model
to load, no GPU memory spent on embeddings, a fully deterministic (and
therefore Nix-cacheable) index, and strong behaviour on the identifier-heavy
queries this corpus is made of -- ``services.openssh.enable``,
``hyprland bindm``, ``kubectl rollout``.
"""

from .index import Chunk, Index, build_index, load_index  # noqa: F401
from .retrieve import Hit, format_context, search  # noqa: F401
