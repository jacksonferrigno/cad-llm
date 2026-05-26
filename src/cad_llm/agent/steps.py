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
    phase: str | None = None


@dataclass
class RunState:
    tools_called: set[str] = field(default_factory=set)
    has_src_write: bool = False
    src_dirty: bool = False
    awaiting_src_fix: bool = False
    consecutive_doc_searches: int = 0
    last_doc_query: str = ""

    def record_tool_result(self, name: str, arguments: dict[str, object], output: str) -> None:
        self.tools_called.add(name)
        if name in {"write_file", "search_replace"}:
            path = str(arguments.get("path", ""))
            if path.replace("\\", "/").startswith("src/") and self._src_edit_applied(name, output):
                self.has_src_write = True
                self.src_dirty = True
                self.awaiting_src_fix = False
                self.consecutive_doc_searches = 0
                self.last_doc_query = ""
        if name == "search_cadquery_docs":
            self.consecutive_doc_searches += 1
        if name == "run_python_sandbox":
            first = output.splitlines()[0] if output else ""
            if first.startswith("exit_code="):
                self.src_dirty = False
                if not first.startswith("exit_code=0"):
                    self.awaiting_src_fix = True
                    self.consecutive_doc_searches = 0
                    self.last_doc_query = ""

    def doc_search_loop_detected(self, query: str) -> bool:
        if not self.awaiting_src_fix:
            return False
        normalized = query.strip().lower()
        repeated = bool(normalized) and normalized == self.last_doc_query
        self.last_doc_query = normalized
        return self.consecutive_doc_searches >= 2 or repeated

    @staticmethod
    def _src_edit_applied(name: str, output: str) -> bool:
        if output.startswith("error:") or output in {"no_change", "no_match"}:
            return False
        if name == "search_replace":
            return output == "ok"
        return True

    def needs_write(self) -> bool:
        return not self.has_src_write

    def needs_sandbox(self) -> bool:
        return self.src_dirty
