import json
from pathlib import Path

from cad_llm.tools.workspace import create_chat, create_project, list_projects, load_project


def test_create_project_layout(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="bracket", project_id="testproj01")

    assert project.root == projects_root / "testproj01"
    assert (project.meta_dir / "project.json").is_file()
    assert (project.src_dir / "parts").is_dir()
    assert project.outputs_dir.is_dir()

    meta = json.loads((project.meta_dir / "project.json").read_text())
    assert meta["id"] == "testproj01"
    assert meta["name"] == "bracket"


def test_create_chat_under_project(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo")
    chat = create_chat(project, title="first pass", chat_id="chat001")

    assert chat.root == project.chats_dir / "chat001"
    assert chat.meta_path.is_file()
    assert chat.transcript_path.is_file()

    meta = json.loads(chat.meta_path.read_text())
    assert meta["id"] == "chat001"
    assert meta["title"] == "first pass"


def test_list_and_load_projects(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    create_project(projects_root, name="alpha", project_id="alpha01")
    create_project(projects_root, name="beta", project_id="beta01")

    layouts = list_projects(projects_root)
    assert [layout.project_id for layout in layouts] == ["alpha01", "beta01"]

    loaded = load_project(projects_root, "alpha01")
    meta = json.loads((loaded.meta_dir / "project.json").read_text(encoding="utf-8"))
    assert meta["name"] == "alpha"
