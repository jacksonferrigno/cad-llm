from pathlib import Path

from cad_llm.docs.loader import extract_symbols, load_and_split_html


def test_load_and_split_cadquery_html() -> None:
    html_path = Path("data/cadquery-latest/index.html")
    if not html_path.exists():
        return

    chunks = load_and_split_html(html_path)
    assert len(chunks) > 100
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert any("workplane" in " ".join(chunk.metadata.get("symbols", [])) for chunk in chunks)


def test_extract_symbols_finds_cadquery_terms() -> None:
    symbols = extract_symbols("Use cq.Workplane('XY').circle(10).extrude(20).cutThruAll()")
    assert "workplane" in symbols
    assert "circle" in symbols
    assert "extrude" in symbols
