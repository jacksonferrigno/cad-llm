"""Research subagent — docs only, produces a handoff brief."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cad_llm.agent.steps import AgentStep
from cad_llm.agent.subagent import GenerateFn, SubagentResult, run_subagent
from cad_llm.config import settings
from cad_llm.tools.binding import get_bound_research_tools
from cad_llm.tools.docs.search import search_cadquery_docs
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout

RESEARCH_SYSTEM_PROMPT = """You are the CAD research subagent.

Your ONLY job: study the user request and CadQuery documentation, then write a Research Brief
for the implementation subagent.

Rules:
- Use search_cadquery_docs when initial snippets are missing a symbol you need.
- The ## CadQuery APIs section must COPY signatures and examples verbatim from doc snippets.
- Never invent APIs (no Workplane.cylinder, cq.Cylinder, union for holes, exporters.exporter).
- For through-holes: circle(radius).cutThruAll() on an existing solid — not a separate cylinder + union.
- Do NOT write Python code. Do NOT call write_file or run_python_sandbox.

When research is complete, respond with ONLY this structure (no tool calls):

## Requirements
(bullet list — every constraint from the user request)

## CadQuery APIs
(relevant method signatures and short examples copied from docs)

## Implementation plan
(numbered steps using ONLY the APIs above)
"""


def _seed_docs(prompt: str) -> str:
    cfg = settings
    chunks = settings.resolve(cfg.docs_chunks_cache)
    primary = search_cadquery_docs(
        prompt,
        db_url=cfg.docs_db_url,
        collection_name=cfg.docs_collection_name,
        embedding_model=cfg.docs_embedding_model,
        chunks_cache=chunks,
        limit=5,
    )
    secondary = search_cadquery_docs(
        "cadquery Workplane box centered cutThruAll export step",
        db_url=cfg.docs_db_url,
        collection_name=cfg.docs_collection_name,
        embedding_model=cfg.docs_embedding_model,
        chunks_cache=chunks,
        limit=3,
    )
    return f"{primary}\n\n---\n\n{secondary}"


def run_research_agent(
    project: ProjectLayout,
    chat: ChatLayout,
    prompt: str,
    *,
    model: Any,
    tokenizer: Any,
    max_steps: int = 5,
    max_tokens: int = 1024,
    generate_fn: GenerateFn | None = None,
    on_step: Callable[[AgentStep], None] | None = None,
) -> SubagentResult:
    seeded = _seed_docs(prompt)
    user_message = (
        f"User request:\n{prompt}\n\n"
        f"Initial CadQuery doc snippets:\n{seeded}\n\n"
        "Search docs if anything is missing, then output the Research Brief."
    )
    return run_subagent(
        phase="research",
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        user_message=user_message,
        tools=get_bound_research_tools(project.root),
        model=model,
        tokenizer=tokenizer,
        chat=chat,
        max_steps=max_steps,
        max_tokens=max_tokens,
        generate_fn=generate_fn,
        on_step=on_step,
    )
