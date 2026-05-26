"""Path resolution jailed to a project root."""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(ValueError):
    pass


def resolve_project_path(project_root: Path, relative_path: str | Path) -> Path:
    root = project_root.resolve()
    target = Path(relative_path)
    resolved = target.resolve() if target.is_absolute() else (root / target).resolve()
    if root not in resolved.parents and resolved != root:
        msg = f"Path escapes project root: {relative_path}"
        raise PathEscapeError(msg)
    return resolved
