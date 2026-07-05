"""Given a task description, decides which files are relevant.

Implements the retrieval steps from Master Document section 7.3:
1. embed the task description
2. retrieve top-K similar chunks from the vector store
3. boost files whose symbol names are named explicitly in the task text
4. pull in one hop of directly related files (Python import edges)
5. rank and return the top-N files

Ranking/truncation to a token budget (7.3 step 6) is deferred to the
reasoning module, which knows the actual context window budget; this module
only decides *which* files matter and in what order.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from indexer.chunker import iter_source_files
from indexer.embedder import Embedder
from indexer.pipeline import IndexResult

_DEFAULT_TOP_K_CHUNKS = 10
_DEFAULT_TOP_N_FILES = 3

# Fixed bonus applied when the task text names a real symbol whose chunk
# lives in a given file — outranks pure embedding similarity, since an exact
# name match is a much stronger signal than semantic closeness.
_SYMBOL_MATCH_SCORE = 100.0
_ONE_HOP_SCORE = 0.1


@dataclass(frozen=True)
class FileScore:
    file_path: str
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RetrievalResult:
    files: list[FileScore]
    related_files: list[FileScore]

    @property
    def file_paths(self) -> list[str]:
        return [f.file_path for f in self.files]


def _symbol_matches(task: str, all_symbols: list[str]) -> set[str]:
    """Return the subset of known symbol names that appear as a whole word
    in the task text (case-insensitive)."""
    matches = set()
    for symbol in all_symbols:
        bare = symbol.rsplit(".", 1)[-1]
        if re.search(rf"\b{re.escape(bare)}\b", task, flags=re.IGNORECASE):
            matches.add(symbol)
    return matches


def _module_name(file_path: str) -> str:
    return file_path[: -len(".py")].replace("/", ".") if file_path.endswith(".py") else file_path


def build_import_graph(repo_root: Path) -> dict[str, set[str]]:
    """Best-effort forward import graph: file_path -> set of repo-relative
    file paths it imports, resolved via Python `import`/`from ... import`
    statements. Only intra-repo, absolute (non-relative) imports are
    resolved; anything else (stdlib, third-party, relative imports) is
    ignored since it has no corresponding file in the repo.

    Uses indexer.chunker.iter_source_files rather than a raw `rglob("*.py")`
    (Epic A6) so this walks the same file set chunk_repo already does --
    skipping `.venv`/`node_modules`/`.git`/etc -- instead of also parsing
    every dependency vendored under the repo root. That exclusion is the
    actual fix for the crash this was built to prevent: a real run against
    Solvix's own repo hit a `.venv`-vendored joblib test fixture deliberately
    encoded as Big5 (a `# -*- coding: big5 -*-` file testing joblib's own
    handling of non-UTF-8 source), which should never have been scanned as
    part of *this* repo's import graph in the first place.

    A file that still can't be decoded as UTF-8 even after that exclusion
    (a genuinely non-UTF-8 file inside the repo proper) is skipped with a
    warning rather than crashing graph-building for every other file.
    """
    py_files = list(iter_source_files(repo_root))
    module_to_path = {_module_name(str(p.relative_to(repo_root))): str(p.relative_to(repo_root)) for p in py_files}

    graph: dict[str, set[str]] = {str(p.relative_to(repo_root)): set() for p in py_files}
    for p in py_files:
        rel = str(p.relative_to(repo_root))
        try:
            source_text = p.read_text()
        except UnicodeDecodeError as error:
            warnings.warn(f"skipping {rel} in import-graph building: not valid UTF-8 ({error})", stacklevel=2)
            continue
        try:
            tree = ast.parse(source_text, filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                module_names.append(node.module)
            for module_name in module_names:
                target = module_to_path.get(module_name)
                if target and target != rel:
                    graph[rel].add(target)
    return graph


def _one_hop_files(seed_files: set[str], import_graph: dict[str, set[str]]) -> set[str]:
    related: set[str] = set()
    reverse: dict[str, set[str]] = {}
    for src, targets in import_graph.items():
        for tgt in targets:
            reverse.setdefault(tgt, set()).add(src)

    for f in seed_files:
        related |= import_graph.get(f, set())  # files this one imports
        related |= reverse.get(f, set())  # files that import this one
    return related - seed_files


def retrieve_relevant_files(
    task: str,
    index_result: IndexResult,
    embedder: Embedder,
    top_n: int = _DEFAULT_TOP_N_FILES,
    top_k_chunks: int = _DEFAULT_TOP_K_CHUNKS,
) -> RetrievalResult:
    """Given a task description and a built repo index, return the files most
    likely to require changes, ranked highest-relevance first.
    """
    scores: dict[str, dict[str, object]] = {}

    def _record(file_path: str, score: float, reason: str) -> None:
        entry = scores.setdefault(file_path, {"score": 0.0, "reasons": []})
        entry["score"] = max(entry["score"], score)
        entry["reasons"].append(reason)

    query_embedding = embedder.embed_texts([task])[0]
    for hit in index_result.vector_store.query(query_embedding, top_k=top_k_chunks):
        file_path = hit["metadata"]["file_path"]
        similarity = 1.0 / (1.0 + max(hit["distance"], 0.0))
        _record(file_path, similarity, f"embedding_similarity:{hit['metadata']['symbol']}")

    matched_symbols = _symbol_matches(task, index_result.symbol_index.all_symbols())
    for symbol in matched_symbols:
        for location in index_result.symbol_index.lookup(symbol):
            _record(location.file_path, _SYMBOL_MATCH_SCORE, f"symbol_match:{symbol}")

    ranked = sorted(
        (FileScore(file_path=fp, score=v["score"], reasons=tuple(v["reasons"])) for fp, v in scores.items()),
        key=lambda f: f.score,
        reverse=True,
    )
    top_files = ranked[:top_n]

    import_graph = build_import_graph(index_result.repo_root)
    seed = {f.file_path for f in top_files}
    one_hop = _one_hop_files(seed, import_graph)
    related = [
        FileScore(file_path=fp, score=_ONE_HOP_SCORE, reasons=("one_hop_import",))
        for fp in sorted(one_hop)
        if fp not in seed
    ]

    return RetrievalResult(files=top_files, related_files=related)
