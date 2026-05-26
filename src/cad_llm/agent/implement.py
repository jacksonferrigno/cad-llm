"""Implementation subagent — write, sandbox, fix."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from cad_llm.agent.parse import ParsedToolCall
from cad_llm.agent.steps import AgentStep, RunState
from cad_llm.agent.subagent import GenerateFn, SubagentResult, run_subagent
from cad_llm.agent.transcript import append_event
from cad_llm.tools.binding import get_bound_implement_tools
from cad_llm.tools.edit.tool import read_file
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout

IMPLEMENT_SYSTEM_PROMPT = """You are the CAD implementation subagent.

You receive a Research Brief with verified CadQuery APIs and a plan.
Your job: write src/main.py, run sandbox, fix until exit_code=0, verify against the brief.

Rules:
- Your FIRST action must be a write_file tool call for src/main.py — never prose or markdown code.
- Use ONLY APIs listed in the Research Brief.
- write_file content = raw Python only (no markdown fences).
- Do not call search_cadquery_docs — the research agent already verified APIs.
- Sandbox runs automatically after src edits.
- When sandbox passes, compare code to the brief and user request before summarizing.
- Export to outputs/ with cq.exporters.export(result, "outputs/name.step"). Never cq.show or cq.Viewer.

When sandbox passed and code is verified: brief summary only (no code dump).
"""


def _sandbox_failed(output: str) -> bool:
    first = output.splitlines()[0] if output else ""
    return first.startswith("exit_code=") and not first.startswith("exit_code=0")


def _make_after_src_edit(
    *,
    project: ProjectLayout,
    prompt: str,
    tools_by_name: dict[str, StructuredTool],
    chat: ChatLayout,
    phase: str,
    on_step: Callable[[AgentStep], None] | None,
) -> Callable[[RunState, list[dict[str, str]]], None]:
    def _emit(step: AgentStep) -> None:
        step.phase = phase
        if on_step is not None:
            on_step(step)

    def _after(state: RunState, messages: list[dict[str, str]]) -> None:
        if not state.needs_sandbox():
            return
        from cad_llm.agent.subagent import run_tool_call

        append_event(chat.transcript_path, "auto_tool", name="run_python_sandbox", phase=phase)
        output = run_tool_call(
            call=ParsedToolCall(name="run_python_sandbox", arguments={"entrypoint": "src/main.py"}),
            tools_by_name=tools_by_name,
            messages=messages,
            chat=chat,
            emit=_emit,
            phase=phase,
            state=state,
        )
        if not _sandbox_failed(output):
            try:
                src = read_file(project.root, "src/main.py")
            except Exception:  # noqa: BLE001
                return
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Sandbox passed. Verify src/main.py against the Research Brief and user request.\n\n"
                        f"User request: {prompt}\n\n"
                        f"--- src/main.py ---\n{src}\n---\n\n"
                        "Fix with tools if wrong; otherwise brief summary."
                    ),
                }
            )

    return _after


def run_implement_agent(
    project: ProjectLayout,
    chat: ChatLayout,
    prompt: str,
    research_brief: str,
    *,
    model: Any,
    tokenizer: Any,
    max_steps: int = 15,
    max_tokens: int = 2048,
    generate_fn: GenerateFn | None = None,
    on_step: Callable[[AgentStep], None] | None = None,
) -> SubagentResult:
    tools = get_bound_implement_tools(project.root)
    tools_by_name = {tool.name: tool for tool in tools}
    user_message = f"User request:\n{prompt}\n\nResearch Brief:\n{research_brief}"
    return run_subagent(
        phase="implement",
        system_prompt=IMPLEMENT_SYSTEM_PROMPT,
        user_message=user_message,
        tools=tools,
        model=model,
        tokenizer=tokenizer,
        chat=chat,
        max_steps=max_steps,
        max_tokens=max_tokens,
        generate_fn=generate_fn,
        on_step=on_step,
        require_sandbox_clean=True,
        require_write=True,
        after_src_edit=_make_after_src_edit(
            project=project,
            prompt=prompt,
            tools_by_name=tools_by_name,
            chat=chat,
            phase="implement",
            on_step=on_step,
        ),
    )
