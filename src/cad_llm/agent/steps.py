"""Shared agent step types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStep:
    kind: str
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


@dataclass
class RunState:
    tools_called: set[str] = field(default_factory=set)
    has_src_write: bool = False
    src_dirty: bool = False
    docs_injected_for_dirty_src: bool = False

    def record_tool_result(self, name: str, arguments: dict[str, object], output: str) -> None:
        self.tools_called.add(name)
        if name in {"write_file", "search_replace"}:
            path = str(arguments.get("path", ""))
            if path.replace("\\", "/").startswith("src/") and not output.startswith("error:"):
                self.has_src_write = True
                self.src_dirty = True
                self.docs_injected_for_dirty_src = False
        if name == "run_python_sandbox":
            first = output.splitlines()[0] if output else ""
            if first.startswith("exit_code=0"):
                self.src_dirty = False
                self.docs_injected_for_dirty_src = False

    def needs_write(self) -> bool:
        return not self.has_src_write

    def needs_sandbox(self) -> bool:
        return self.src_dirty

    def mark_docs_injected(self) -> None:
        self.docs_injected_for_dirty_src = True
