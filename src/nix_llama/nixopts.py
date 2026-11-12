"""Extraction of documented options from Nix modules.

NixOS, home-manager and nix-darwin all declare their configuration surface the
same way -- ``name = mkOption { description = "..."; type = ...; }`` nested
under an attribute path -- and that surface is the single most valuable thing a
Nix-specialised model can know. Both consumers of the corpus use this module:
the retrieval index turns each option into its own searchable chunk, and the
training dataset turns each one into a question/answer pair.

There is no Nix evaluator here on purpose. Evaluating nixpkgs' module system to
get `options.json` costs minutes and a lot of memory; scanning the source
recovers the same option paths, descriptions, types and defaults in
milliseconds, and degrades gracefully on modules it cannot parse.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# A `name = mkOption { ... };` declaration in a NixOS/home-manager/nix-darwin
# module. The qualifier is optional because modules spell it `mkOption`,
# `lib.mkOption` and `lib.options.mkOption` interchangeably. Nested braces in
# the body are why the block itself is scanned by hand below rather than
# matched in one regex.
_OPTION_START_RE = re.compile(
    r"(?P<name>[A-Za-z_][\w.'\"-]*)\s*=\s*(?:[\w.]*\.)?"
    r"(?P<kind>mkOption|mkEnableOption|mkPackageOption)\b",
    re.MULTILINE,
)
# The string argument of `mkEnableOption "the widget daemon"`.
_ENABLE_ARG_RE = re.compile(r"\s*(?P<quote>''|\")(?P<body>.*?)(?P=quote)", re.DOTALL)
_DESCRIPTION_RE = re.compile(
    r"description\s*=\s*(?:lib\.)?(?:mdDoc\s*)?(?P<quote>\"\"\"|''|\")(?P<body>.*?)(?P=quote)",
    re.DOTALL,
)
_TYPE_RE = re.compile(r"type\s*=\s*(?P<body>[^;]+);")
_DEFAULT_RE = re.compile(r"default\s*=\s*(?P<body>[^;]+);")
_EXAMPLE_RE = re.compile(r"example\s*=\s*(?P<body>[^;]+);")


def _nix_string(raw: str) -> str:
    raw = raw.strip()
    for quote in ('"""', "''", '"'):
        if raw.startswith(quote) and raw.endswith(quote) and len(raw) > 2 * len(quote):
            raw = raw[len(quote) : -len(quote)]
            break
    return " ".join(raw.split())


def _balanced_block(text: str, start: int) -> str:
    """Return the ``{ ... }`` block beginning at or after ``start``.

    Returns empty if the declaration ends (``;``) before a block opens, so a
    valueless declaration cannot steal the *next* option's body.
    """
    open_at = text.find("{", start)
    if open_at < 0:
        return ""
    terminator = text.find(";", start)
    if 0 <= terminator < open_at:
        return ""
    depth = 0
    for i in range(open_at, min(len(text), open_at + 8000)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    return ""


# `services.foo = { ... }` -- an attribute whose value opens a set. Used to
# reconstruct the path an option is declared under.
_ATTR_OPEN_RE = re.compile(r"(?:^|[{;])\s*(?P<attr>[A-Za-z_][\w.'\"-]*)\s*=\s*\{", re.MULTILINE)


def _enclosing_paths(text: str) -> dict[int, str]:
    """Map each option declaration's offset to the attribute path enclosing it.

    An option is only useful as training data under its real path:
    `services.openssh.enable` teaches something, a bare `enable` teaches
    nothing. There is no Nix parser here, so this tracks brace depth and
    remembers which attribute opened each set -- which is accurate for the
    `options.services.foo = { ... }` shape nixpkgs modules actually use.
    """
    opens = {match.end() - 1: match.group("attr") for match in _ATTR_OPEN_RE.finditer(text)}
    events: list[tuple[int, str]] = []
    for i, char in enumerate(text):
        if char == "{":
            events.append((i, "open"))
        elif char == "}":
            events.append((i, "close"))
    declarations = {m.start(): m for m in _OPTION_START_RE.finditer(text)}

    paths: dict[int, str] = {}
    stack: list[str | None] = []
    pending = sorted(declarations)
    cursor = 0
    for position, kind in events:
        while cursor < len(pending) and pending[cursor] < position:
            segments = [s for s in stack if s]
            paths[pending[cursor]] = ".".join(segments)
            cursor += 1
        if kind == "open":
            stack.append(opens.get(position))
        elif stack:
            stack.pop()
    while cursor < len(pending):
        paths[pending[cursor]] = ".".join(s for s in stack if s)
        cursor += 1
    return paths


def _qualify(path: str, name: str) -> str:
    """Join an enclosing path and an option name into a usable option path."""
    # Modules nest declarations under `options`, sometimes as `options.<path>`;
    # neither prefix belongs in the answer.
    segments = [s for s in path.split(".") if s and s != "options"]
    segments = [s for s in segments if s not in ("config", "mkIf", "mkMerge")]
    return ".".join([*segments, name]) if segments else name


def options_from_nix(text: str, *, prefix_hint: str = "") -> Iterator[dict[str, str]]:
    """Extract documented options from a Nix module."""
    enclosing = _enclosing_paths(text)
    for match in _OPTION_START_RE.finditer(text):
        name = _qualify(enclosing.get(match.start(), ""), match.group("name").strip("\"'"))
        kind = match.group("kind")

        if kind == "mkEnableOption":
            # `mkEnableOption "the widget daemon"` carries its description as a
            # bare string argument and generates the rest.
            argument = _ENABLE_ARG_RE.match(text, match.end())
            if not argument:
                continue
            thing = _nix_string(argument.group("body"))
            if len(thing) < 3:
                continue
            yield {
                "name": name,
                "description": f"Whether to enable {thing}.",
                "namespace": prefix_hint,
                "type": "boolean",
                "default": "false",
            }
            continue

        block = _balanced_block(text, match.end())
        if not block:
            continue
        description = _DESCRIPTION_RE.search(block)
        if not description:
            continue
        body = _nix_string(description.group("body"))
        if len(body) < 24:
            continue
        option: dict[str, str] = {
            "name": name,
            "description": body,
            "namespace": prefix_hint,
        }
        for key, pattern in (
            ("type", _TYPE_RE),
            ("default", _DEFAULT_RE),
            ("example", _EXAMPLE_RE),
        ):
            found = pattern.search(block)
            if found:
                option[key] = " ".join(found.group("body").split())[:200]
        yield option
