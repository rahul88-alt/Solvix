from pathlib import Path

from indexer.symbol_index import build_symbol_index

SAMPLE_REPO = Path(__file__).parent.parent / "sample_repo"


def test_lookup_finds_top_level_function():
    index = build_symbol_index(SAMPLE_REPO)
    locations = index.lookup("add")

    # "add" exists as both a top-level function and Calculator.add's bare name
    assert len(locations) == 2
    files = {loc.file_path for loc in locations}
    assert files == {"calculator.py"}


def test_lookup_qualified_method_name_is_precise():
    index = build_symbol_index(SAMPLE_REPO)
    locations = index.lookup("Calculator.add")

    assert len(locations) == 1
    loc = locations[0]
    assert loc.file_path == "calculator.py"
    assert loc.kind == "method"


def test_lookup_unknown_symbol_returns_empty():
    index = build_symbol_index(SAMPLE_REPO)
    assert index.lookup("does_not_exist") == []


def test_lookup_class_symbol():
    index = build_symbol_index(SAMPLE_REPO)
    locations = index.lookup("Calculator")

    assert len(locations) == 1
    assert locations[0].kind == "class"


def test_index_contains_symbols_from_nested_directories():
    index = build_symbol_index(SAMPLE_REPO)
    assert "slugify" in index
    assert "truncate" in index

    locations = index.lookup("slugify")
    assert locations[0].file_path == "utils/strings.py"


def test_symbol_location_line_numbers_are_1_indexed_and_span_the_definition():
    index = build_symbol_index(SAMPLE_REPO)
    locations = index.lookup("Calculator.__init__")

    assert len(locations) == 1
    loc = locations[0]
    assert loc.start_line < loc.end_line
    assert loc.start_line > 0
