"""Splits source files into function/class-level chunks using tree-sitter.

MVP targets Python only (see Master Document section 7.1: "pick one language
to start"). Adding another language means adding an entry to _LANGUAGES plus
a node-type mapping — the walker itself is language-agnostic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

_PY_LANGUAGE = Language(tspython.language())

# node types that count as a chunk boundary, per language
_LANGUAGES = {
    ".py": {
        "language": _PY_LANGUAGE,
        "function_types": {"function_definition"},
        "class_types": {"class_definition"},
    }
}


@dataclass(frozen=True)
class Chunk:
    file_path: str          # path relative to the repo root
    symbol: str             # e.g. "foo" or "Bar.method"
    kind: str               # "function" | "class" | "method"
    start_line: int         # 1-indexed, inclusive
    end_line: int           # 1-indexed, inclusive
    code: str


def supported_extensions() -> set[str]:
    return set(_LANGUAGES.keys())


def _node_name(node: Node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return source[name_node.start_byte:name_node.end_byte].decode("utf-8")


def _chunk_from_node(
    node: Node, source: bytes, file_path: str, kind: str, qualified_name: str
) -> Chunk:
    return Chunk(
        file_path=file_path,
        symbol=qualified_name,
        kind=kind,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        code=source[node.start_byte:node.end_byte].decode("utf-8"),
    )


def _walk(
    node: Node,
    source: bytes,
    file_path: str,
    lang_cfg: dict,
    chunks: list[Chunk],
    scope: str | None = None,
) -> None:
    function_types = lang_cfg["function_types"]
    class_types = lang_cfg["class_types"]

    for child in node.children:
        if child.type in class_types:
            name = _node_name(child, source)
            if name is not None:
                chunks.append(_chunk_from_node(child, source, file_path, "class", name))
                _walk(child, source, file_path, lang_cfg, chunks, scope=name)
                continue
        if child.type in function_types:
            name = _node_name(child, source)
            if name is not None:
                qualified = f"{scope}.{name}" if scope else name
                kind = "method" if scope else "function"
                chunks.append(_chunk_from_node(child, source, file_path, kind, qualified))
                # do not recurse into nested functions/classes as separate scope chunks
                continue
        _walk(child, source, file_path, lang_cfg, chunks, scope=scope)


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    """Chunk a single source file into function/class-level Chunks.

    Returns an empty list for unsupported file types.
    """
    lang_cfg = _LANGUAGES.get(path.suffix)
    if lang_cfg is None:
        return []

    source = path.read_bytes()
    parser = Parser(lang_cfg["language"])
    tree = parser.parse(source)

    file_path = str(path.relative_to(repo_root))
    chunks: list[Chunk] = []
    _walk(tree.root_node, source, file_path, lang_cfg, chunks)
    return chunks


_EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".chroma"}


def iter_source_files(repo_root: Path):
    """Yield every supported source file under repo_root, skipping common junk dirs."""
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix not in _LANGUAGES:
            continue
        if _EXCLUDED_DIRS & set(path.relative_to(repo_root).parts[:-1]):
            continue
        yield path


def chunk_repo(repo_root: Path) -> list[Chunk]:
    """Chunk every supported source file under repo_root.

    A file whose bytes aren't valid UTF-8 (Epic A6) is skipped entirely with
    a warning rather than crashing chunking for every other file: tree-sitter
    parses raw bytes regardless of encoding, so parsing itself never fails,
    but _node_name/_chunk_from_node's source[...].decode("utf-8") does, as
    soon as a chunk's byte range covers non-UTF-8 content. Skipping the whole
    file (rather than guessing an encoding to decode it with, or decoding
    with errors="replace" and silently corrupting the chunk's text) matches
    "don't guess-decode source code you can't read correctly".
    """
    chunks: list[Chunk] = []
    for path in iter_source_files(repo_root):
        try:
            chunks.extend(chunk_file(path, repo_root))
        except UnicodeDecodeError as error:
            rel = path.relative_to(repo_root)
            warnings.warn(f"skipping {rel}: not valid UTF-8 ({error})", stacklevel=2)
    return chunks
