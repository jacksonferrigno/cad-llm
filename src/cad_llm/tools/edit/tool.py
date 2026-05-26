"""LangChain edit tools jailed to a project root via resolve_project_path."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cad_llm.inference.extract import extract_python_code
from cad_llm.tools.common.paths import resolve_project_path


def read_file(
    project_root: Path,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    resolved = resolve_project_path(project_root, path)
    text = resolved.read_text(encoding="utf-8")
    if start_line is None and end_line is None:
        return text
    lines = text.splitlines(keepends=True)
    start_idx = (start_line - 1) if start_line is not None else 0
    end_idx = end_line if end_line is not None else len(lines)
    return "".join(lines[start_idx:end_idx])


def grep(project_root: Path, pattern: str, path: str | None = None) -> str:
    root = project_root.resolve()
    base = resolve_project_path(project_root, path) if path is not None else root
    regex = re.compile(pattern)
    out: list[str] = []

    def scan_file(fp: Path) -> None:
        try:
            content = fp.read_text(encoding="utf-8")
        except OSError:
            return
        for i, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
                out.append(f"{rel}:{i}:{line}")

    if base.is_file():
        scan_file(base)
    else:
        for fp in base.rglob("*"):
            if fp.is_file():
                scan_file(fp)
    return "\n".join(out) if out else ""


def _is_interactive_show_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped.startswith(("cq.show(", "show_object(")):
        return True
    return "Viewer(" in stripped and ".show(" in stripped


def _sanitize_python_content(content: str) -> str:
    """Normalize common model mistakes before syntax validation."""
    if "```" in content:
        content = extract_python_code(content)

    lines = [line.rstrip() for line in content.replace("\r\n", "\n").splitlines()]
    fixed: list[str] = []
    for line in lines:
        if _is_interactive_show_line(line):
            continue
        # Markdown/code-block leaks often prefix top-level lines with a single space.
        if line.startswith(" ") and not line.startswith("  ") and line.strip():
            fixed.append(line[1:])
        else:
            fixed.append(line)

    text = "\n".join(fixed).strip()
    return f"{text}\n" if text else ""


def _validate_python(content: str, *, path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return f"error: Python syntax error in {path}: {exc.msg} (line {exc.lineno})"
    return None


def write_file(project_root: Path, path: str, content: str) -> str:
    if path.endswith(".py"):
        content = _sanitize_python_content(content)

    syntax_error = _validate_python(content, path=path)
    if syntax_error is not None:
        return syntax_error

    resolved = resolve_project_path(project_root, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return str(resolved.relative_to(project_root.resolve()))


def search_replace(project_root: Path, path: str, old: str, new: str) -> str:
    resolved = resolve_project_path(project_root, path)
    text = resolved.read_text(encoding="utf-8")
    if old not in text:
        return "no_match"
    updated = text.replace(old, new)
    if path.endswith(".py"):
        updated = _sanitize_python_content(updated)
    syntax_error = _validate_python(updated, path=path)
    if syntax_error is not None:
        return syntax_error
    resolved.write_text(updated, encoding="utf-8")
    return "ok"


def delete_file(project_root: Path, path: str) -> str:
    resolved = resolve_project_path(project_root, path)
    resolved.unlink()
    return "ok"


# --- StructuredTool wrappers ---


class ReadFileInput(BaseModel):
    project_root: str = Field(description="Absolute filesystem path to the project root.")
    path: str = Field(description="Path relative to project root, or absolute if under root.")
    start_line: int | None = Field(default=None, description="1-based start line (inclusive).")
    end_line: int | None = Field(default=None, description="1-based end line (inclusive).")


class GrepInput(BaseModel):
    project_root: str = Field(description="Absolute filesystem path to the project root.")
    pattern: str = Field(description="Regular expression searched per line.")
    path: str | None = Field(
        default=None,
        description="File or directory under project_root; omit to search entire project.",
    )


class WriteFileInput(BaseModel):
    project_root: str = Field(description="Absolute filesystem path to the project root.")
    path: str = Field(description="Path relative to project root, or absolute if under root.")
    content: str = Field(description="Full file contents to write.")


class SearchReplaceInput(BaseModel):
    project_root: str = Field(description="Absolute filesystem path to the project root.")
    path: str = Field(description="Path relative to project root, or absolute if under root.")
    old: str = Field(description="Substring to replace.")
    new: str = Field(description="Replacement text.")


class DeleteFileInput(BaseModel):
    project_root: str = Field(description="Absolute filesystem path to the project root.")
    path: str = Field(description="Path relative to project root, or absolute if under root.")


def _read_file_tool(
    project_root: str,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    return read_file(Path(project_root), path, start_line=start_line, end_line=end_line)


def _grep_tool(project_root: str, pattern: str, path: str | None = None) -> str:
    return grep(Path(project_root), pattern, path=path)


def _write_file_tool(project_root: str, path: str, content: str) -> str:
    return write_file(Path(project_root), path, content)


def _search_replace_tool(project_root: str, path: str, old: str, new: str) -> str:
    return search_replace(Path(project_root), path, old, new)


def _delete_file_tool(project_root: str, path: str) -> str:
    return delete_file(Path(project_root), path)


def get_edit_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_read_file_tool,
            name="read_file",
            description="Read a text file within the project, optionally slicing by line number.",
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            func=_grep_tool,
            name="grep",
            description="Regex search per line over a file, directory subtree, or whole project.",
            args_schema=GrepInput,
        ),
        StructuredTool.from_function(
            func=_write_file_tool,
            name="write_file",
            description="Create or overwrite a file under the project root.",
            args_schema=WriteFileInput,
        ),
        StructuredTool.from_function(
            func=_search_replace_tool,
            name="search_replace",
            description="Replace all occurrences of a substring in one file.",
            args_schema=SearchReplaceInput,
        ),
        StructuredTool.from_function(
            func=_delete_file_tool,
            name="delete_file",
            description="Delete a single file within the project root.",
            args_schema=DeleteFileInput,
        ),
    ]
