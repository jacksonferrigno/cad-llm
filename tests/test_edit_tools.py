from pathlib import Path

import pytest

from cad_llm.tools.common.paths import PathEscapeError
from cad_llm.tools.edit import (
    delete_file,
    get_edit_tools,
    grep,
    read_file,
    search_replace,
    write_file,
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def test_path_escape_rejected(project_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    with pytest.raises(PathEscapeError):
        read_file(project_root, "../outside.txt")

    with pytest.raises(PathEscapeError):
        write_file(project_root, "../outside.txt", "nope")


def test_read_write_and_line_slice(project_root: Path) -> None:
    (project_root / "a.txt").write_text("one\ntwo\nthree\n")

    assert read_file(project_root, "a.txt") == "one\ntwo\nthree\n"
    assert read_file(project_root, "a.txt", start_line=2, end_line=2) == "two\n"

    rel = write_file(project_root, "b.txt", "hello")
    assert rel == "b.txt"
    assert (project_root / "b.txt").read_text() == "hello"


def test_search_replace(project_root: Path) -> None:
    write_file(project_root, "edit.txt", "foo bar foo")

    assert search_replace(project_root, "edit.txt", "foo", "baz") == "ok"
    assert read_file(project_root, "edit.txt") == "baz bar baz"
    assert search_replace(project_root, "edit.txt", "missing", "x") == "no_match"


def test_grep_file_and_project(project_root: Path) -> None:
    write_file(project_root, "src/x.py", "import cadquery\n# TODO\n")
    write_file(project_root, "src/y.py", "print('cadquery')\n")

    file_hits = grep(project_root, "cadquery", path="src/x.py")
    assert file_hits == "src/x.py:1:import cadquery"

    all_hits = grep(project_root, "cadquery")
    assert "src/x.py:1:import cadquery" in all_hits.splitlines()
    assert "src/y.py:1:print('cadquery')" in all_hits.splitlines()


def test_delete_file(project_root: Path) -> None:
    write_file(project_root, "drop.txt", "gone")
    target = project_root / "drop.txt"
    assert target.is_file()

    assert delete_file(project_root, "drop.txt") == "ok"
    assert not target.exists()


def test_write_file_rejects_invalid_python(project_root: Path) -> None:
    result = write_file(project_root, "src/main.py", "def oops(\n")
    assert result.startswith("error: Python syntax error")
    assert not (project_root / "src" / "main.py").exists()


def test_get_edit_tools_exports(project_root: Path) -> None:
    tools = get_edit_tools()
    names = {t.name for t in tools}
    assert names == {"read_file", "grep", "write_file", "search_replace", "delete_file"}
