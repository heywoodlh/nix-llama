#!/usr/bin/env python3
"""Refresh nix/model/sources.lock.json.

Resolves each documentation source in ``sources.json`` to a concrete commit and
computes the fixed-output hash Nix needs to fetch it reproducibly. Run it via
``nix run .#update-sources``; CI runs it on a schedule so the corpus (and the
"latest upstream docs" promise for Hyprland and Niri) stays current.

Hashes are obtained by building each fetcher with a deliberately wrong hash and
reading the hash Nix reports back. That is ugly, but it is the only method that
works identically for plain fetches and sparse checkouts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


SOURCES = _repo_path("model", "sources.json")
LOCK = _repo_path("model", "sources.lock.json")

FAKE_HASH = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
_GOT_RE = re.compile(r"got:\s+(sha256-[A-Za-z0-9+/=]+)")

API = {
    "github": "https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
    "codeberg": "https://codeberg.org/api/v1/repos/{owner}/{repo}/commits?limit=1&sha={branch}",
    "gitlab": "https://gitlab.com/api/v4/projects/{owner}%2F{repo}/repository/commits/{branch}",
}

CLONE_URL = {
    "github": "https://github.com/{owner}/{repo}.git",
    "codeberg": "https://codeberg.org/{owner}/{repo}.git",
    "gitlab": "https://gitlab.com/{owner}/{repo}.git",
}

# A GitHub wiki is a separate git repository with no API of its own, so its
# revision has to come from the remote directly. Niri's documentation lives
# there, which is why this exists.
WIKI_URL = "https://github.com/{owner}/{repo}.wiki.git"


def fetch_json(url: str) -> Any:
    # https URLs built from literals above; no user-controlled scheme.
    request = urllib.request.Request(url, headers={"User-Agent": "nix-llama"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - https only
        return json.load(response)


def clone_url(source: dict[str, Any]) -> str:
    template = WIKI_URL if source.get("wiki") else CLONE_URL[source.get("forge", "github")]
    return template.format(owner=source["owner"], repo=source["repo"])


def resolve_rev(source: dict[str, Any]) -> str:
    if source.get("wiki"):
        proc = subprocess.run(  # noqa: S603
            # `git` comes from PATH on purpose: this runs under `nix run`.
            ["git", "ls-remote", clone_url(source), "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        head = proc.stdout.split()
        if not head:
            raise RuntimeError(f"{source['name']}: wiki remote reported no HEAD")
        return head[0]
    forge = source.get("forge", "github")
    url = API[forge].format(
        owner=source["owner"], repo=source["repo"], branch=source.get("branch", "main")
    )
    data = fetch_json(url)
    if forge == "codeberg":
        return data[0]["sha"]
    if forge == "gitlab":
        return data["id"]
    return data["sha"]


def nix_expr(source: dict[str, Any], rev: str) -> str:
    forge = source.get("forge", "github")
    sparse = source.get("sparse")
    url = clone_url(source)
    if sparse:
        paths = " ".join(json.dumps(p) for p in sparse)
        return (
            "with import <nixpkgs> { }; fetchgit { "
            f'url = "{url}"; rev = "{rev}"; hash = "{FAKE_HASH}"; '
            f"sparseCheckout = [ {paths} ]; }}"
        )
    if forge == "github" and not source.get("wiki"):
        return (
            "with import <nixpkgs> { }; fetchFromGitHub { "
            f'owner = "{source["owner"]}"; repo = "{source["repo"]}"; '
            f'rev = "{rev}"; hash = "{FAKE_HASH}"; }}'
        )
    return (
        "with import <nixpkgs> { }; fetchgit { "
        f'url = "{url}"; rev = "{rev}"; hash = "{FAKE_HASH}"; }}'
    )


def compute_hash(source: dict[str, Any], rev: str) -> str:
    expr = nix_expr(source, rev)
    proc = subprocess.run(  # noqa: S603
        # `nix` comes from PATH on purpose: this runs under `nix run`.
        ["nix", "build", "--impure", "--no-link", "--expr", expr],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    match = _GOT_RE.search(proc.stderr) or _GOT_RE.search(proc.stdout)
    if match:
        return match.group(1)
    if proc.returncode == 0:
        raise RuntimeError(
            f"{source['name']}: the placeholder hash was accepted, which should be impossible"
        )
    raise RuntimeError(f"{source['name']}: could not determine hash\n{proc.stderr[-2000:]}")


def update_one(source: dict[str, Any]) -> tuple[str, dict[str, str] | None, str | None]:
    name = source["name"]
    if source.get("from"):
        # Sourced from a flake input; nothing to pin here.
        return name, None, None
    try:
        rev = resolve_rev(source)
        return name, {"rev": rev, "hash": compute_hash(source, rev)}, None
    except (urllib.error.URLError, OSError, RuntimeError, KeyError, IndexError) as exc:
        return name, None, str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="update just these sources")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--sources", default=str(SOURCES))
    parser.add_argument("--lock", default=str(LOCK))
    args = parser.parse_args(argv)

    declared = json.loads(Path(args.sources).read_text(encoding="utf-8"))["sources"]
    if args.only:
        declared = [s for s in declared if s["name"] in set(args.only)]

    lock_path = Path(args.lock)
    lock: dict[str, Any] = {"sources": {}}
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, entry, error in pool.map(update_one, declared):
            if error:
                failures[name] = error
                print(f"FAIL {name}: {error.splitlines()[0]}", file=sys.stderr)
                continue
            if entry is None:
                continue
            previous = lock["sources"].get(name, {})
            verb = "unchanged" if previous.get("rev") == entry["rev"] else "updated "
            lock["sources"][name] = entry
            print(f"{verb} {name} {entry['rev'][:12]} {entry['hash']}")

    lock["sources"] = dict(sorted(lock["sources"].items()))
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {lock_path}")

    optional_only = {s["name"] for s in declared if s.get("optional")}
    fatal = set(failures) - optional_only
    if fatal:
        print(f"\nrequired sources failed: {', '.join(sorted(fatal))}", file=sys.stderr)
        return 1
    if failures:
        print(
            f"\noptional sources failed and were left as-is: {', '.join(sorted(failures))}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
