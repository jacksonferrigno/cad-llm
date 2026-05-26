"""Build agent system context before the model runs."""

from __future__ import annotations

from cad_llm.agent.prompts import AGENT_SYSTEM_PROMPT, build_system_prompt
from cad_llm.agent.steps import AgentStep
from cad_llm.agent.transcript import append_event
from cad_llm.config import settings
from cad_llm.tools.docs.search import search_cadquery_docs
from cad_llm.tools.skills.loader import load_skill
from cad_llm.tools.workspace.project import ChatLayout

_BOOTSTRAP_SKILLS = ("cad-generation", "brainstorming")


def _load_bootstrap_skills() -> str:
    sections = [f"## {name}\n\n{load_skill(name).strip()}" for name in _BOOTSTRAP_SKILLS]
    return "\n\n".join(sections)


def bootstrap_system_prompt(
    prompt: str,
    chat: ChatLayout,
    *,
    include_skill: bool = True,
    include_docs: bool = True,
    cached_skill: str | None = None,
) -> tuple[str, list[AgentStep], str]:
    """Load skill/docs into the system prompt. Returns (prompt, steps, skill_content)."""
    steps: list[AgentStep] = []

    if include_skill or cached_skill is None:
        skill_content = _load_bootstrap_skills()
        if include_skill:
            for skill_name in _BOOTSTRAP_SKILLS:
                append_event(
                    chat.transcript_path,
                    "bootstrap",
                    name="load_skill",
                    skill=skill_name,
                )
                steps.append(
                    AgentStep(
                        kind="bootstrap",
                        content="ready",
                        tool_name="load_skill",
                        tool_args={"name": skill_name},
                    )
                )
    else:
        skill_content = cached_skill

    doc_context = ""
    if include_docs:
        try:
            doc_context = search_cadquery_docs(
                prompt,
                db_url=settings.docs_db_url,
                collection_name=settings.docs_collection_name,
                embedding_model=settings.docs_embedding_model,
                chunks_cache=settings.resolve(settings.docs_chunks_cache),
                limit=3,
            )
            append_event(
                chat.transcript_path,
                "bootstrap",
                name="search_cadquery_docs",
                query=prompt,
                chars=len(doc_context),
            )
            steps.append(
                AgentStep(
                    kind="bootstrap",
                    content="ready",
                    tool_name="search_cadquery_docs",
                    tool_args={"query": prompt},
                )
            )
        except Exception as exc:  # noqa: BLE001
            doc_context = f"(CadQuery docs unavailable: {exc})"
            append_event(
                chat.transcript_path,
                "bootstrap",
                name="search_cadquery_docs",
                query=prompt,
                error=str(exc),
            )
            steps.append(
                AgentStep(
                    kind="bootstrap",
                    content="unavailable",
                    tool_name="search_cadquery_docs",
                    tool_args={"query": prompt},
                )
            )

    if not include_skill and not include_docs and cached_skill is not None:
        return build_system_prompt(cached_skill, doc_context), steps, skill_content

    if not include_docs and cached_skill is not None:
        return build_system_prompt(cached_skill, doc_context), steps, skill_content

    return build_system_prompt(skill_content, doc_context), steps, skill_content


def minimal_system_prompt(cached_skill: str) -> str:
    return build_system_prompt(cached_skill, "")
