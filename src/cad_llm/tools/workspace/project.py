"""Project and chat workspace layout."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    project_id: str
    root: Path
    meta_dir: Path
    src_dir: Path
    chats_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class ChatLayout:
    chat_id: str
    root: Path
    meta_path: Path
    transcript_path: Path


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def project_layout(projects_root: Path, project_id: str) -> ProjectLayout:
    root = projects_root / project_id
    return ProjectLayout(
        project_id=project_id,
        root=root,
        meta_dir=root / ".cad-llm",
        src_dir=root / "src",
        chats_dir=root / "chats",
        outputs_dir=root / "outputs",
    )


def chat_layout(project: ProjectLayout, chat_id: str) -> ChatLayout:
    root = project.chats_dir / chat_id
    return ChatLayout(
        chat_id=chat_id,
        root=root,
        meta_path=root / "meta.json",
        transcript_path=root / "transcript.jsonl",
    )


def create_project(
    projects_root: Path, *, name: str, project_id: str | None = None
) -> ProjectLayout:
    pid = project_id or uuid.uuid4().hex[:12]
    layout = project_layout(projects_root, pid)
    layout.meta_dir.mkdir(parents=True, exist_ok=True)
    layout.src_dir.mkdir(parents=True, exist_ok=True)
    layout.chats_dir.mkdir(parents=True, exist_ok=True)
    layout.outputs_dir.mkdir(parents=True, exist_ok=True)
    (layout.src_dir / "parts").mkdir(parents=True, exist_ok=True)
    (layout.meta_dir / "project.json").write_text(
        json.dumps({"id": pid, "name": name, "created_at": _now_iso()}, indent=2)
    )
    return layout


def create_chat(project: ProjectLayout, *, title: str, chat_id: str | None = None) -> ChatLayout:
    cid = chat_id or uuid.uuid4().hex[:12]
    layout = chat_layout(project, cid)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.meta_path.write_text(
        json.dumps({"id": cid, "title": title, "created_at": _now_iso()}, indent=2)
    )
    layout.transcript_path.touch()
    return layout


def load_project(projects_root: Path, project_id: str) -> ProjectLayout:
    layout = project_layout(projects_root, project_id)
    if not (layout.meta_dir / "project.json").is_file():
        msg = f"Unknown project: {project_id}"
        raise FileNotFoundError(msg)
    return layout


def list_projects(projects_root: Path) -> list[ProjectLayout]:
    if not projects_root.is_dir():
        return []

    layouts: list[ProjectLayout] = []
    for child in sorted(projects_root.iterdir()):
        meta_path = child / ".cad-llm" / "project.json"
        if not child.is_dir() or not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        project_id = meta.get("id", child.name)
        layouts.append(project_layout(projects_root, project_id))
    return layouts
