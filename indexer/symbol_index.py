"""Lightweight symbol map: symbol name -> file:line locations.

Built directly from chunker output, so a symbol's location always matches
its chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from indexer.chunker import Chunk, chunk_repo


@dataclass(frozen=True)
class SymbolLocation:
    file_path: str
    start_line: int
    end_line: int
    kind: str


class SymbolIndex:
    """Maps symbol name -> list of locations (a name may be defined more than once)."""

    def __init__(self) -> None:
        self._symbols: dict[str, list[SymbolLocation]] = {}

    def add(self, chunk: Chunk) -> None:
        locations = self._symbols.setdefault(chunk.symbol, [])
        locations.append(
            SymbolLocation(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                kind=chunk.kind,
            )
        )
        # a method is also reachable by its bare name (e.g. "method" as well
        # as "Bar.method"), which matters for exact lookups where the caller
        # doesn't know the enclosing class.
        if "." in chunk.symbol:
            bare_name = chunk.symbol.rsplit(".", 1)[1]
            bare_locations = self._symbols.setdefault(bare_name, [])
            loc = SymbolLocation(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                kind=chunk.kind,
            )
            if loc not in bare_locations:
                bare_locations.append(loc)

    def lookup(self, symbol: str) -> list[SymbolLocation]:
        return list(self._symbols.get(symbol, []))

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._symbols

    def all_symbols(self) -> list[str]:
        return list(self._symbols.keys())


def build_symbol_index(repo_root: Path, chunks: list[Chunk] | None = None) -> SymbolIndex:
    """Build a SymbolIndex for repo_root. Reuses `chunks` if already computed."""
    if chunks is None:
        chunks = chunk_repo(repo_root)
    index = SymbolIndex()
    for chunk in chunks:
        index.add(chunk)
    return index
