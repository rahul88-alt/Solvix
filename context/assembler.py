"""Given a task description, decides which files are relevant.

Implements the retrieval steps from Master Document section 7.3:
1. embed the task description
2. retrieve top-K similar chunks from the vector store
3. boost files whose symbol names are named explicitly in the task text
4. pull in one hop of directly related files (Python import edges)
5. rank and return the top-N files
6. rank all underlying evidence (exact symbol match > high-similarity chunk >
   one-hop related file) and truncate to a token budget (SLX-A5), stopping
   before whole chunks/files that would push the total over budget rather
   than ever slicing one apart

qwen2.5-coder:14b (like most model families) doesn't expose a convenient
local tokenizer, and OpenAI's tiktoken targets a different model family
entirely -- so token counts here are a ~4-chars-per-token approximation, a
widely used rule of thumb for English/code text. It only needs to be in the
right ballpark to keep the prompt within the model's context window with
headroom to spare, not exact.
"""

from __future__ import annotations

import ast
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from indexer.chunker import iter_source_files
from indexer.embedder import Embedder
from indexer.pipeline import IndexResult

_DEFAULT_TOP_K_CHUNKS = 10
_DEFAULT_TOP_N_FILES = 3

# See config._DEFAULT_CONTEXT_TOKEN_BUDGET for the reasoning behind this
# number; duplicated here (rather than imported) so this module has a
# sensible standalone default with no dependency on config.py. Public (no
# leading underscore) since reasoning.task_input re-exports it as its own
# default.
DEFAULT_TOKEN_BUDGET = 10000

_CHARS_PER_TOKEN = 4

# Fixed bonus applied when the task text names a real symbol whose chunk
# lives in a given file — outranks pure embedding similarity, since an exact
# name match is a much stronger signal than semantic closeness.
_SYMBOL_MATCH_SCORE = 100.0
_ONE_HOP_SCORE = 0.1

# Evidence tiers, in priority order: an exact symbol match is a much
# stronger relevance signal than a high-similarity embedding chunk, which in
# turn is real (if indirect) evidence, unlike a one-hop related file, which
# is included only for surrounding context.
_TIER_SYMBOL_MATCH = 0
_TIER_EMBEDDING_CHUNK = 1
_TIER_ONE_HOP_FILE = 2


def estimate_tokens(text: str) -> int:
    """Rough token count for `text`, ~4 characters per token.

    An approximation, not an exact tokenizer -- see module docstring.
    """
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


@dataclass(frozen=True)
class FileScore:
    file_path: str
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RetrievalResult:
    files: list[FileScore]
    related_files: list[FileScore]
    estimated_tokens: int = 0

    @property
    def file_paths(self) -> list[str]:
        return [f.file_path for f in self.files]


@dataclass(frozen=True)
class _Evidence:
    """One atomic, whole unit of context: a single chunk's code, or (for a
    one-hop related file, which contributes no chunk content) just its file
    path. Truncation only ever includes or excludes a whole _Evidence item.
    """

    tier: int
    file_path: str
    tokens: int


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


def _read_chunk_text(
    repo_root: Path, file_path: str, start_line: int, end_line: int, cache: dict[str, list[str] | None]
) -> str | None:
    if file_path not in cache:
        try:
            cache[file_path] = (repo_root / file_path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            cache[file_path] = None

    lines = cache[file_path]
    if lines is None:
        return None
    return "\n".join(lines[start_line - 1 : end_line])


def retrieve_relevant_files(
    task: str,
    index_result: IndexResult,
    embedder: Embedder,
    top_n: int = _DEFAULT_TOP_N_FILES,
    top_k_chunks: int = _DEFAULT_TOP_K_CHUNKS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> RetrievalResult:
    """Given a task description and a built repo index, return the files most
    likely to require changes, ranked highest-relevance first, with the
    underlying evidence truncated (whole chunks/files only) to fit
    `token_budget`.

    The single highest-priority piece of evidence is always kept, even if it
    alone exceeds token_budget -- an empty result (e.g. from a misconfigured,
    too-small budget) would leave the reasoning model with no context at all,
    which is worse than a plan grounded in one slightly-over-budget match.
    """
    scores: dict[str, dict[str, object]] = {}

    def _record(file_path: str, score: float, reason: str) -> None:
        entry = scores.setdefault(file_path, {"score": 0.0, "reasons": []})
        entry["score"] = max(entry["score"], score)
        entry["reasons"].append(reason)

    query_embedding = embedder.embed_texts([task])[0]
    hits = index_result.vector_store.query(query_embedding, top_k=top_k_chunks)
    for hit in hits:
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

    # Build the ordered, atomic evidence list (tier 0 first, then 1, then 2)
    # that token-budget truncation walks. Each item is a whole chunk's code
    # (tiers 0-1) or a bare file-path reference (tier 2, since no file
    # content is loaded for one-hop files at this stage) -- truncation can
    # only include/exclude a whole item, never slice inside one.
    line_cache: dict[str, list[str] | None] = {}
    seen_chunks: set[tuple[str, int, int]] = set()
    evidence: list[_Evidence] = []

    for symbol in sorted(matched_symbols):
        for location in index_result.symbol_index.lookup(symbol):
            if location.file_path not in seed:
                continue
            key = (location.file_path, location.start_line, location.end_line)
            if key in seen_chunks:
                continue
            text = _read_chunk_text(
                index_result.repo_root, location.file_path, location.start_line, location.end_line, line_cache
            )
            if text is None:
                continue
            seen_chunks.add(key)
            evidence.append(_Evidence(tier=_TIER_SYMBOL_MATCH, file_path=location.file_path, tokens=estimate_tokens(text)))

    for hit in hits:
        metadata = hit["metadata"]
        file_path = metadata["file_path"]
        if file_path not in seed:
            continue
        key = (file_path, metadata["start_line"], metadata["end_line"])
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        evidence.append(_Evidence(tier=_TIER_EMBEDDING_CHUNK, file_path=file_path, tokens=estimate_tokens(hit["code"])))

    for f in related:
        evidence.append(_Evidence(tier=_TIER_ONE_HOP_FILE, file_path=f.file_path, tokens=estimate_tokens(f.file_path)))

    running_tokens = 0
    kept_files: set[str] = set()
    kept_related: set[str] = set()
    for i, item in enumerate(evidence):
        # The single highest-priority item is always included, even if it
        # alone exceeds the budget: a plan built from one exact symbol match
        # that's a bit over budget beats a plan built from nothing at all,
        # which is what a too-small (e.g. misconfigured) budget would
        # otherwise silently produce.
        if i > 0 and running_tokens + item.tokens > token_budget:
            break
        running_tokens += item.tokens
        if item.tier == _TIER_ONE_HOP_FILE:
            kept_related.add(item.file_path)
        else:
            kept_files.add(item.file_path)

    final_files = [f for f in top_files if f.file_path in kept_files]
    final_related = [f for f in related if f.file_path in kept_related]

    return RetrievalResult(files=final_files, related_files=final_related, estimated_tokens=running_tokens)
