"""LangChain tool wrapper for loading Markdown skills."""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cad_llm.tools.skills.loader import load_skill


class LoadSkillInput(BaseModel):
    name: str = Field(
        description=(
            "Skill id to load (e.g. cad-generation, brainstorming, cad-debug); "
            "matches a file under cad_llm/tools/skills/{name}.md."
        ),
    )


def load_skill_tool(name: str) -> str:
    return load_skill(name)


def get_skill_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=load_skill_tool,
        name="load_skill",
        description=(
            "Load Markdown guidance for a named skill (cad-generation, brainstorming, cad-debug)."
        ),
        args_schema=LoadSkillInput,
    )
