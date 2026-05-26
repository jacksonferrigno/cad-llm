"""Orchestrator — research then implement, or chat for brainstorms."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from cad_llm.agent.generate import generate_completion
from cad_llm.agent.implement import run_implement_agent
from cad_llm.agent.research import run_research_agent
from cad_llm.agent.runner import AgentRunResult, GenerateFn
from cad_llm.agent.steps import AgentStep
from cad_llm.agent.transcript import append_event, write_run_summary
from cad_llm.inference.extract import strip_model_response
from cad_llm.inference.generate import CadGenerator
from cad_llm.tools.skills.loader import load_skill
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout

_BUILD_RE = re.compile(
    r"\b(build|create|make|cut|add|export|implement|write|generate|model)\b",
    re.IGNORECASE,
)
_BRAINSTORM_RE = re.compile(
    r"\b(what should|what could|ideas?|brainstorm|options?|cook up|suggest)\b",
    re.IGNORECASE,
)

CHAT_SYSTEM_PROMPT = """You are a CAD design assistant.

If the user is brainstorming or asking what to build, reply in plain text with 2–3 concrete
options and one clarifying question. Do not write code or call tools.
"""


def is_build_request(prompt: str) -> bool:
    if _BRAINSTORM_RE.search(prompt):
        return False
    return bool(_BUILD_RE.search(prompt))


def run_orchestrated_agent(
    project: ProjectLayout,
    chat: ChatLayout,
    prompt: str,
    *,
    model_id: str | None = None,
    generator: CadGenerator | None = None,
    max_steps: int = 15,
    max_tokens: int = 2048,
    on_step: Callable[[AgentStep], None] | None = None,
    generate_fn: GenerateFn | None = None,
) -> AgentRunResult:
    gen = generator or CadGenerator(model_id=model_id)
    if gen.model is None:
        gen.load()
    model = gen.model
    tokenizer = gen.tokenizer

    steps: list[AgentStep] = []
    started_at = datetime.now(tz=UTC).isoformat()
    append_event(chat.transcript_path, "user", content=prompt)

    def emit(step: AgentStep) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    skill_content = f"## brainstorming\n\n{load_skill('brainstorming').strip()}"

    if not is_build_request(prompt):
        emit(AgentStep(kind="phase", content="chat", phase="orchestrator"))
        formatted = tokenizer.apply_chat_template(
            [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if generate_fn is not None:
            raw = generate_fn(model, tokenizer, formatted, max_tokens)
        else:
            raw = generate_completion(model, tokenizer, formatted, max_tokens)
        final_response = strip_model_response(raw) or raw.strip()
        append_event(chat.transcript_path, "assistant", content=final_response, phase="chat")
        emit(AgentStep(kind="assistant", content=final_response, phase="chat"))
    else:
        research_steps = max(3, min(5, max_steps // 3))
        implement_steps = max_steps

        emit(AgentStep(kind="phase", content="research", phase="orchestrator"))
        research = run_research_agent(
            project,
            chat,
            prompt,
            model=model,
            tokenizer=tokenizer,
            max_steps=research_steps,
            max_tokens=min(1024, max_tokens),
            generate_fn=generate_fn,
            on_step=on_step,
        )
        steps.extend(research.steps)

        emit(AgentStep(kind="phase", content="implement", phase="orchestrator"))
        implement = run_implement_agent(
            project,
            chat,
            prompt,
            research.final_text,
            model=model,
            tokenizer=tokenizer,
            max_steps=implement_steps,
            max_tokens=max_tokens,
            generate_fn=generate_fn,
            on_step=on_step,
        )
        steps.extend(implement.steps)
        final_response = implement.final_text

    run_summary_path = chat.root / f"run_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_run_summary(
        run_summary_path,
        {
            "project_id": project.project_id,
            "chat_id": chat.chat_id,
            "prompt": prompt,
            "started_at": started_at,
            "finished_at": datetime.now(tz=UTC).isoformat(),
            "final_response": final_response,
            "mode": "build" if is_build_request(prompt) else "chat",
            "steps": [
                {
                    "kind": step.kind,
                    "phase": step.phase,
                    "tool_name": step.tool_name,
                    "content": step.content[:500],
                }
                for step in steps
            ],
        },
    )

    return AgentRunResult(
        project=project,
        chat=chat,
        prompt=prompt,
        final_response=final_response,
        steps=steps,
        transcript_path=chat.transcript_path,
        run_summary_path=run_summary_path,
        skill_content=skill_content,
    )
