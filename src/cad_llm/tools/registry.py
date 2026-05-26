"""Central registry for LangChain agent tools."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from cad_llm.tools.docs.tool import get_search_cadquery_docs_tool
from cad_llm.tools.edit.tool import get_edit_tools
from cad_llm.tools.sandbox.tool import get_sandbox_tools
from cad_llm.tools.skills.tool import get_skill_tool


def get_agent_tools() -> list[StructuredTool]:
    """Return all StructuredTools for the CAD agent.

    Edit and sandbox tools take ``project_root`` on each call (see their args
    schemas); this registry does not bind a project. The agent loop should pass
    the active workspace project path when invoking read_file, grep,
    run_python_sandbox, etc.
    """
    return [
        get_search_cadquery_docs_tool(),
        *get_edit_tools(),
        *get_sandbox_tools(),
        get_skill_tool(),
    ]
