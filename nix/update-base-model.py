#!/usr/bin/env python3
"""Pin a Hugging Face base model into nix/model/base.lock.json.

The training pipeline needs the base model's safetensors, tokenizer and config
as a fixed-output derivation. Every file is pinned individually so the fetch is
reproducible and resumable, and so a repository that gains unrelated files later
does not change our hash.

Usage::

    nix run .#update-base-model -- Qwen/Qwen2.5-Coder-1.5B-Instruct
    nix run .#update-base-model -- Qwen/Qwen2.5-Coder-7B-Instruct --as default
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


def _repo_path(*parts: str) -> pathlib.Path:
    """Locate a repository file whether running from a checkout or the store.

    `nix run .#update-sources` executes this script from /nix/store, where it
    has no repository around it -- and the lock file it produces has to land in
    the working tree anyway. So the working directory wins, and the path next to
    the script is only a fallback for running it directly.
    """
    relative = pathlib.Path("nix").joinpath(*parts)
    if relative.exists():
        return relative
    beside = pathlib.Path(__file__).resolve().parent.joinpath(*parts)
    if beside.exists():
        return beside
    # Neither exists yet (a lock file being created for the first time): prefer
    # the working tree, which is where it belongs.
    return relative


LOCK = _repo_path("model", "base.lock.json")

# Files a transformers/peft fine-tune actually needs. Anything else in the
# repository (ONNX exports, duplicate GGUFs, images) is skipped.
WANTED_EXACT = {
    "config.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "LICENSE",
    "README.md",
}
WANTED_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def fetch_json(url: str) -> Any:
    # https URLs built from literals above; no user-controlled scheme.
    request = urllib.request.Request(url, headers={"User-Agent": "nix-llama"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - https only
        return json.load(response)


def prefetch(url: str) -> str:
    proc = subprocess.run(  # noqa: S603
        # `nix` comes from PATH on purpose: this runs under `nix run`.
        ["nix", "store", "prefetch-file", "--json", url],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)["hash"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face repo id, e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--as",
        dest="variant",
        default="small",
        help="which nix-llama variant this is the base for (default: small)",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--lock", default=str(LOCK))
    args = parser.parse_args(argv)

    info = fetch_json(f"https://huggingface.co/api/models/{args.repo}?revision={args.revision}")
    sha = info.get("sha") or args.revision
    license_id = (info.get("cardData") or {}).get("license", "unknown")
    if license_id.lower() not in ("apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0"):
        print(
            f"refusing to pin {args.repo}: licence {license_id!r} is not a permissive licence "
            f"nix-llama is willing to build on",
            file=sys.stderr,
        )
        return 2

    files = [
        f["rfilename"]
        for f in info.get("siblings", [])
        if f["rfilename"] in WANTED_EXACT or WANTED_RE.match(f["rfilename"])
    ]
    if not any(f.endswith(".safetensors") for f in files):
        print(f"{args.repo} has no safetensors weights", file=sys.stderr)
        return 2

    entry: dict[str, Any] = {
        "repo": args.repo,
        "revision": sha,
        "license": license_id,
        "files": {},
    }
    for name in sorted(files):
        url = f"https://huggingface.co/{args.repo}/resolve/{sha}/{name}"
        print(f"prefetching {name} ...", flush=True)
        entry["files"][name] = prefetch(url)

    lock_path = Path(args.lock)
    lock = {"bases": {}}
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.setdefault("bases", {})[args.variant] = entry
    lock["bases"] = dict(sorted(lock["bases"].items()))
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pinned {len(files)} files for {args.variant} -> {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
