from cad_llm.inference.extract import extract_python_code


def test_extract_fenced_python() -> None:
    text = """Here is the code:

```python
import cadquery as cq
result = cq.Workplane("XY").box(1, 2, 3)
```
"""
    assert "import cadquery" in extract_python_code(text)
    assert "```" not in extract_python_code(text)


def test_extract_raw_python() -> None:
    text = 'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 1, 1)'
    assert extract_python_code(text) == text


def test_extract_prefers_longest_block() -> None:
    text = """```python
x = 1
```
```python
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
```"""
    code = extract_python_code(text)
    assert "box(10, 10, 10)" in code


def test_extract_strips_thinking_block() -> None:
    text = (
        "<think>\nPlan the hex prism first.\n</think>\n\n"
        'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 1, 1)'
    )
    code = extract_python_code(text)
    assert code.startswith("import cadquery")
    assert "Plan the hex" not in code


def test_extract_strips_leading_thinking_close() -> None:
    text = (
        "</think>\n\n"
        'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 1, 1)'
    )
    code = extract_python_code(text)
    assert code.startswith("import cadquery")
