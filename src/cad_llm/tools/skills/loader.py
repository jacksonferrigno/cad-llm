"""Load agent skill prompts from Markdown files alongside this module."""

from __future__ import annotations

import re
from pathlib import Path

_VALID_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _skills_dir() -> Path:
    return Path(__file__).resolve().parent


def _normalize_name(name: str) -> str:
    return name.strip()


def list_skills() -> list[str]:
    """Return skill ids (stem of each ``*.md`` file in this package)."""
    directory = _skills_dir()
    return sorted(p.stem for p in directory.glob("*.md") if p.is_file())


def load_skill(name: str) -> str:
    """Read the Markdown body for ``name`` (``{name}.md`` in this package).

    Raises:
        ValueError: If ``name`` is not a valid skill identifier.
        FileNotFoundError: If no matching skill file exists.
    """
    normalized = _normalize_name(name)
    if not normalized or _VALID_SKILL_NAME.fullmatch(normalized) is None:
        msg = f"Invalid skill name: {name!r}"
        raise ValueError(msg)

    path = _skills_dir() / f"{normalized}.md"
    if not path.is_file():
        msg = f"Unknown skill: {normalized}"
        raise FileNotFoundError(msg)

    return path.read_text(encoding="utf-8")
