"""Bind project-scoped tools so the agent never supplies ``project_root``."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cad_llm.tools.docs.tool import get_search_cadquery_docs_tool
from cad_llm.tools.edit.tool import delete_file, grep, read_file, search_replace, write_file
from cad_llm.tools.sandbox.tool import run_python
from cad_llm.tools.skills.tool import get_skill_tool


class ReadFileBoundInput(BaseModel):
    path: str = Field(description="Path relative to project root.")
    start_line: int | None = Field(default=None, description="1-based start line (inclusive).")
    end_line: int | None = Field(default=None, description="1-based end line (exclusive).")


class GrepBoundInput(BaseModel):
    pattern: str = Field(description="Regex pattern matched per line.")
    path: str | None = Field(
        default=None,
        description="File or directory under project root; omit to search entire project.",
    )


class WriteFileBoundInput(BaseModel):
    path: str = Field(description="Path relative to project root.")
    content: str = Field(description="Full file contents to write.")


class SearchReplaceBoundInput(BaseModel):
    path: str = Field(description="Path relative to project root.")
    old: str = Field(description="Substring to replace.")
    new: str = Field(description="Replacement text.")


class DeleteFileBoundInput(BaseModel):
    path: str = Field(description="Path relative to project root.")


class RunPythonBoundInput(BaseModel):
    entrypoint: str = Field(
        default="src/main.py",
        description="Python file to run, relative to project root.",
    )
    timeout: int = Field(default=30, description="Subprocess timeout in seconds.")


def get_bound_research_tools(_project_root: Path) -> list[StructuredTool]:
    return [get_search_cadquery_docs_tool()]


def get_bound_implement_tools(project_root: Path) -> list[StructuredTool]:
    names = {
        "read_file",
        "grep",
        "write_file",
        "search_replace",
        "delete_file",
        "run_python_sandbox",
    }
    return [tool for tool in get_bound_agent_tools(project_root) if tool.name in names]


def get_bound_agent_tools(project_root: Path) -> list[StructuredTool]:
    """Return agent tools with ``project_root`` fixed to the active workspace."""
    root = project_root.resolve()
    root_str = str(root)

    def read_file_bound(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        return read_file(root, path, start_line=start_line, end_line=end_line)

    def grep_bound(pattern: str, path: str | None = None) -> str:
        return grep(root, pattern, path=path)

    def write_file_bound(path: str, content: str) -> str:
        return write_file(root, path, content)

    def search_replace_bound(path: str, old: str, new: str) -> str:
        return search_replace(root, path, old, new)

    def delete_file_bound(path: str) -> str:
        return delete_file(root, path)

    def run_python_bound(entrypoint: str = "src/main.py", timeout: int = 30) -> str:
        try:
            return run_python(root, entrypoint=entrypoint, timeout=timeout)
        except ValueError as e:
            return f"error: {e}"

    return [
        get_search_cadquery_docs_tool(),
        StructuredTool.from_function(
            func=read_file_bound,
            name="read_file",
            description=f"Read a text file within project {root_str}.",
            args_schema=ReadFileBoundInput,
        ),
        StructuredTool.from_function(
            func=grep_bound,
            name="grep",
            description=f"Regex search within project {root_str}.",
            args_schema=GrepBoundInput,
        ),
        StructuredTool.from_function(
            func=write_file_bound,
            name="write_file",
            description=f"Create or overwrite a file under project {root_str}.",
            args_schema=WriteFileBoundInput,
        ),
        StructuredTool.from_function(
            func=search_replace_bound,
            name="search_replace",
            description=f"Replace text in one file under project {root_str}.",
            args_schema=SearchReplaceBoundInput,
        ),
        StructuredTool.from_function(
            func=delete_file_bound,
            name="delete_file",
            description=f"Delete a file under project {root_str}.",
            args_schema=DeleteFileBoundInput,
        ),
        StructuredTool.from_function(
            func=run_python_bound,
            name="run_python_sandbox",
            description=(
                f"Run a Python entrypoint under project {root_str} "
                "(cwd and PYTHONPATH set to project root)."
            ),
            args_schema=RunPythonBoundInput,
        ),
        get_skill_tool(),
    ]
