"""Run the CAD agent with visible tool steps and transcript logging."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from cad_llm.agent.bootstrap import bootstrap_system_prompt
from cad_llm.agent.generate import generate_completion
from cad_llm.agent.parse import ParsedToolCall, has_incomplete_tool_call, parse_tool_calls
from cad_llm.agent.recovery import (
    build_sandbox_recovery_message,
    docs_query_from_failure,
    fetch_cadquery_docs,
)
from cad_llm.agent.steps import AgentStep, RunState
from cad_llm.agent.summary import summarize_tool_call
from cad_llm.agent.transcript import append_event, write_run_summary
from cad_llm.inference.extract import strip_model_response
from cad_llm.inference.generate import CadGenerator
from cad_llm.tools.binding import get_bound_agent_tools
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout

SANDBOX_NUDGE = (
    "You changed src/ files but sandbox has not passed since the last edit. "
    'Call run_python_sandbox with entrypoint "src/main.py". Use a tool call only.'
)
TRUNCATED_TOOL_NUDGE = (
    "Your last tool call was cut off. Repeat the full <tool_call> block with complete JSON."
)
WRITE_NUDGE = (
    "You have docs but no code yet. Call write_file to create src/main.py now. "
    "Tool call only — no prose."
)
RAMBLE_NUDGE = (
    "Stop explaining. Call write_file, then run_python_sandbox. Tool calls only."
)


def _inject_auto_docs(
    *,
    query: str,
    chat: ChatLayout,
    messages: list[dict[str, str]],
    emit: Callable[[AgentStep], None],
    state: RunState,
    reason: str,
) -> str:
    docs = fetch_cadquery_docs(query)
    append_event(
        chat.transcript_path,
        "auto_docs",
        query=query,
        reason=reason,
        chars=len(docs),
    )
    emit(
        AgentStep(
            kind="bootstrap",
            content=summarize_tool_call("search_cadquery_docs", {"query": query}),
            tool_name="search_cadquery_docs",
            tool_args={"query": query, "auto": True},
        )
    )
    emit(
        AgentStep(
            kind="tool_result",
            content=docs,
            tool_name="search_cadquery_docs",
            tool_args={"query": query},
        )
    )
    message = (
        f"Auto-fetched CadQuery docs for: {query}\n\n"
        f"Read these before editing. Use only APIs shown here.\n\n{docs}"
    )
    messages.append({"role": "user", "content": message})
    state.mark_docs_injected()
    return message


@dataclass
class AgentRunResult:
    project: ProjectLayout
    chat: ChatLayout
    prompt: str
    final_response: str
    steps: list[AgentStep] = field(default_factory=list)
    transcript_path: Path | None = None
    run_summary_path: Path | None = None
    skill_content: str | None = None


GenerateFn = Callable[[Any, Any, str, int], str]


def _tools_for_template(tools: list[StructuredTool]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        if tool.args_schema is not None:
            schema = tool.args_schema.model_json_schema()
            schema.pop("title", None)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return openai_tools


def _execute_tool(tool: StructuredTool, arguments: dict[str, object]) -> str:
    try:
        result = tool.invoke(arguments)
    except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
        return f"error: {exc}"
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _sandbox_failed(output: str) -> bool:
    first = output.splitlines()[0] if output else ""
    return first.startswith("exit_code=") and not first.startswith("exit_code=0")


def _force_sandbox_run(
    *,
    tools_by_name: dict[str, StructuredTool],
    messages: list[dict[str, str]],
    chat: ChatLayout,
    emit: Callable[[AgentStep], None],
    state: RunState,
) -> str:
    call_args = {"entrypoint": "src/main.py"}
    append_event(
        chat.transcript_path,
        "auto_tool",
        name="run_python_sandbox",
        reason="sandbox_required",
    )
    return _run_tool_call(
        call=ParsedToolCall(name="run_python_sandbox", arguments=call_args),
        tools_by_name=tools_by_name,
        messages=messages,
        chat=chat,
        emit=emit,
        record_state=state,
    )


def run_agent(
    project: ProjectLayout,
    chat: ChatLayout,
    prompt: str,
    *,
    model_id: str | None = None,
    generator: CadGenerator | None = None,
    max_steps: int = 12,
    max_tokens: int = 2048,
    on_step: Callable[[AgentStep], None] | None = None,
    generate_fn: GenerateFn | None = None,
    bootstrap: bool = True,
    bootstrap_skill: bool = True,
    bootstrap_docs: bool = True,
    cached_skill: str | None = None,
) -> AgentRunResult:
    """Run a tool-calling agent loop against a project workspace."""
    tools = get_bound_agent_tools(project.root)
    tools_by_name = {tool.name: tool for tool in tools}
    openai_tools = _tools_for_template(tools)

    gen = generator or CadGenerator(model_id=model_id)
    gen.load()
    model = gen.model
    tokenizer = gen.tokenizer

    steps: list[AgentStep] = []
    state = RunState()
    started_at = datetime.now(tz=UTC).isoformat()
    append_event(chat.transcript_path, "user", content=prompt)

    def emit(step: AgentStep) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    skill_content: str | None = cached_skill

    if bootstrap:
        system_prompt, bootstrap_steps, skill_content = bootstrap_system_prompt(
            prompt,
            chat,
            include_skill=bootstrap_skill,
            include_docs=bootstrap_docs,
            cached_skill=cached_skill,
        )
        for step in bootstrap_steps:
            emit(step)
    else:
        from cad_llm.agent.prompts import AGENT_SYSTEM_PROMPT

        system_prompt = AGENT_SYSTEM_PROMPT

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    final_response = ""

    for _ in range(max_steps):
        formatted = tokenizer.apply_chat_template(
            messages,
            tools=openai_tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        if generate_fn is not None:
            raw = generate_fn(model, tokenizer, formatted, max_tokens)
        else:
            raw = generate_completion(model, tokenizer, formatted, max_tokens)

        if has_incomplete_tool_call(raw):
            messages.append({"role": "assistant", "content": raw})
            append_event(chat.transcript_path, "assistant", content=raw, truncated_tool=True)
            messages.append({"role": "user", "content": TRUNCATED_TOOL_NUDGE})
            append_event(chat.transcript_path, "nudge", content=TRUNCATED_TOOL_NUDGE)
            emit(AgentStep(kind="nudge", content=TRUNCATED_TOOL_NUDGE))
            continue

        tool_calls = parse_tool_calls(raw)

        if not tool_calls:
            visible = strip_model_response(raw)

            if state.needs_write():
                messages.append({"role": "assistant", "content": raw})
                append_event(
                    chat.transcript_path,
                    "assistant",
                    content=raw,
                    rejected=True,
                    reason="no_src_write",
                )
                messages.append({"role": "user", "content": WRITE_NUDGE})
                append_event(chat.transcript_path, "nudge", content=WRITE_NUDGE)
                emit(AgentStep(kind="nudge", content=WRITE_NUDGE))
                continue

            if state.needs_sandbox():
                messages.append({"role": "assistant", "content": raw})
                append_event(
                    chat.transcript_path,
                    "assistant",
                    content=raw,
                    rejected=True,
                    reason="sandbox_not_run",
                )
                if not state.docs_injected_for_dirty_src:
                    query = docs_query_from_failure("", user_prompt=prompt)
                    _inject_auto_docs(
                        query=query,
                        chat=chat,
                        messages=messages,
                        emit=emit,
                        state=state,
                        reason="sandbox_not_run",
                    )
                messages.append({"role": "user", "content": SANDBOX_NUDGE})
                append_event(chat.transcript_path, "nudge", content=SANDBOX_NUDGE)
                emit(AgentStep(kind="nudge", content=SANDBOX_NUDGE))
                continue

            final_response = visible or raw.strip()
            append_event(chat.transcript_path, "assistant", content=final_response)
            emit(AgentStep(kind="assistant", content=final_response))
            break

        messages.append({"role": "assistant", "content": raw})
        append_event(chat.transcript_path, "assistant", content=raw, tool_calls=len(tool_calls))

        for call in tool_calls:
            output = _run_tool_call(
                call=call,
                tools_by_name=tools_by_name,
                messages=messages,
                chat=chat,
                emit=emit,
                record_state=state,
            )
            if call.name == "run_python_sandbox" and _sandbox_failed(output):
                short, fail_msg, query, docs = build_sandbox_recovery_message(
                    output,
                    user_prompt=prompt,
                )
                append_event(
                    chat.transcript_path,
                    "auto_docs",
                    query=query,
                    reason="sandbox_failure",
                    chars=len(docs),
                )
                emit(
                    AgentStep(
                        kind="bootstrap",
                        content=f"auto · {query!r}",
                        tool_name="search_cadquery_docs",
                        tool_args={"query": query, "auto": True},
                    )
                )
                emit(
                    AgentStep(
                        kind="tool_result",
                        content=docs,
                        tool_name="search_cadquery_docs",
                        tool_args={"query": query},
                    )
                )
                messages.append({"role": "user", "content": fail_msg})
                append_event(chat.transcript_path, "nudge", content=short)
                emit(AgentStep(kind="nudge", content=short))
                state.mark_docs_injected()
            elif call.name == "search_cadquery_docs" and state.needs_write():
                messages.append({"role": "user", "content": WRITE_NUDGE})
                append_event(chat.transcript_path, "nudge", content=WRITE_NUDGE)
                emit(AgentStep(kind="nudge", content=WRITE_NUDGE))
    else:
        final_response = "Stopped: reached max agent steps without a final reply."
        append_event(chat.transcript_path, "assistant", content=final_response, truncated=True)
        emit(AgentStep(kind="assistant", content=final_response))

    if state.needs_sandbox():
        sandbox_output = _force_sandbox_run(
            tools_by_name=tools_by_name,
            messages=messages,
            chat=chat,
            emit=emit,
            state=state,
        )
        auto_note = f"Auto-ran sandbox.\n{sandbox_output}"
        final_response = f"{final_response}\n\n{auto_note}" if final_response else auto_note
        append_event(chat.transcript_path, "assistant", content=auto_note, auto_sandbox=True)

    run_summary_path = chat.root / f"run_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    summary = {
        "project_id": project.project_id,
        "chat_id": chat.chat_id,
        "prompt": prompt,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "final_response": final_response,
        "tools_called": sorted(state.tools_called),
        "has_src_write": state.has_src_write,
        "src_dirty": state.src_dirty,
        "steps": [
            {
                "kind": step.kind,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "content": step.content[:500],
            }
            for step in steps
        ],
        "paths": {
            "project_root": str(project.root),
            "src_dir": str(project.src_dir),
            "outputs_dir": str(project.outputs_dir),
            "transcript": str(chat.transcript_path),
        },
    }
    write_run_summary(run_summary_path, summary)

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


def _run_tool_call(
    *,
    call: ParsedToolCall,
    tools_by_name: dict[str, StructuredTool],
    messages: list[dict[str, str]],
    chat: ChatLayout,
    emit: Callable[[AgentStep], None],
    record_state: RunState | None = None,
) -> str:
    emit(
        AgentStep(
            kind="tool_call",
            content=summarize_tool_call(call.name, call.arguments),
            tool_name=call.name,
            tool_args=call.arguments,
        )
    )

    append_event(
        chat.transcript_path,
        "tool_call",
        name=call.name,
        arguments=call.arguments,
    )

    tool = tools_by_name.get(call.name)
    if tool is None:
        output = f"error: unknown tool {call.name!r}"
    else:
        output = _execute_tool(tool, call.arguments)

    if record_state is not None:
        record_state.record_tool_result(call.name, call.arguments, output)

    append_event(chat.transcript_path, "tool_result", name=call.name, output=output)
    emit(
        AgentStep(
            kind="tool_result",
            content=output,
            tool_name=call.name,
            tool_args=call.arguments,
        )
    )

    messages.append({"role": "tool", "content": output})
    return output
