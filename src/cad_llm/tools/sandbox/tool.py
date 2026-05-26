"""LangChain tools for running Python entrypoints inside a project sandbox."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cad_llm.tools.common.paths import PathEscapeError, resolve_project_path


def run_python(
    project_root: str | Path,
    entrypoint: str = "src/main.py",
    timeout: int = 30,
) -> str:
    """Run ``python <entrypoint>`` with cwd and PYTHONPATH set to ``project_root``."""
    root = Path(project_root).expanduser().resolve()

    if not entrypoint or not entrypoint.strip():
        raise ValueError("entrypoint must be non-empty")

    try:
        script = resolve_project_path(root, entrypoint)
    except PathEscapeError as e:
        raise ValueError(str(e)) from e

    if not script.is_file():
        return f"error: entrypoint is not a file: {script}"

    rel_script = script.relative_to(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    try:
        completed = subprocess.run(
            [sys.executable, str(rel_script)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"error: timed out after {timeout}s"

    parts: list[str] = [f"exit_code={completed.returncode}"]
    if completed.stdout:
        parts.append(f"stdout:\n{completed.stdout.rstrip()}")
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr.rstrip()}")
    return "\n\n".join(parts)


class RunPythonInput(BaseModel):
    project_root: str = Field(
        description="Absolute or relative path to the project root directory."
    )
    entrypoint: str = Field(
        default="src/main.py",
        description="Path to the Python file to run, relative to project_root unless already absolute.",
    )
    timeout: int = Field(default=30, description="Subprocess timeout in seconds.")


def _run_python_tool(project_root: str, entrypoint: str = "src/main.py", timeout: int = 30) -> str:
    try:
        return run_python(project_root, entrypoint=entrypoint, timeout=timeout)
    except ValueError as e:
        return f"error: {e}"


def get_sandbox_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_run_python_tool,
            name="run_python_sandbox",
            description=(
                "Run a Python script under project_root with cwd and PYTHONPATH set to that root. "
                "Only the allowlisted interpreter is used; entrypoint must stay under project_root."
            ),
            args_schema=RunPythonInput,
        )
    ]
