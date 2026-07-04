from pathlib import Path

from indexer.chunker import chunk_file, chunk_repo

SAMPLE_REPO = Path(__file__).parent.parent / "sample_repo"


def test_chunk_file_finds_top_level_functions_and_classes():
    chunks = chunk_file(SAMPLE_REPO / "calculator.py", SAMPLE_REPO)
    symbols = {c.symbol: c for c in chunks}

    assert "add" in symbols
    assert "subtract" in symbols
    assert "Calculator" in symbols

    assert symbols["add"].kind == "function"
    assert symbols["Calculator"].kind == "class"


def test_chunk_file_finds_methods_with_qualified_names():
    chunks = chunk_file(SAMPLE_REPO / "calculator.py", SAMPLE_REPO)
    symbols = {c.symbol: c for c in chunks}

    assert "Calculator.__init__" in symbols
    assert "Calculator.add" in symbols
    assert "Calculator.subtract" in symbols
    assert symbols["Calculator.add"].kind == "method"


def test_chunk_line_ranges_are_correct():
    chunks = chunk_file(SAMPLE_REPO / "calculator.py", SAMPLE_REPO)
    symbols = {c.symbol: c for c in chunks}

    add_chunk = symbols["add"]
    assert add_chunk.start_line == 4
    assert add_chunk.end_line == 6
    assert "return a + b" in add_chunk.code
    # chunk should not bleed into the next function
    assert "def subtract" not in add_chunk.code


def test_chunk_file_records_correct_file_path():
    chunks = chunk_file(SAMPLE_REPO / "calculator.py", SAMPLE_REPO)
    assert all(c.file_path == "calculator.py" for c in chunks)


def test_chunk_repo_walks_nested_directories():
    chunks = chunk_repo(SAMPLE_REPO)
    file_paths = {c.file_path for c in chunks}
    symbols = {c.symbol for c in chunks}

    assert "utils/strings.py" in file_paths
    assert "slugify" in symbols
    assert "truncate" in symbols


def test_chunk_file_returns_empty_for_unsupported_extension(tmp_path):
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("just some text, not code")
    assert chunk_file(unsupported, tmp_path) == []


def test_chunk_repo_chunk_count_matches_known_symbols():
    chunks = chunk_repo(SAMPLE_REPO)
    # calculator.py: add, subtract, Calculator, __init__, add(method), subtract(method) = 6
    # utils/strings.py: slugify, truncate, reverse = 3
    # report.py: format_summary = 1
    # tests/test_calculator.py: test_add, test_subtract,
    #   test_subtract_negative_result, test_calculator_add_tracks_running_total,
    #   test_calculator_subtract_tracks_running_total = 5
    assert len(chunks) == 15
