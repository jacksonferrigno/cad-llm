from pathlib import Path

from cad_llm.app.mesh import latest_mesh


def test_latest_mesh_picks_newest(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    old = outputs / "a.stl"
    new = outputs / "b.stl"
    old.write_text("old")
    new.write_text("new")

    assert latest_mesh(outputs) == new


def test_latest_mesh_none_when_empty(tmp_path: Path) -> None:
    assert latest_mesh(tmp_path / "missing") is None
