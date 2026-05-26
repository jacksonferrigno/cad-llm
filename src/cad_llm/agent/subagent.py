"""Shared tool-calling loop for subagents."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool

from cad_llm.agent.generate import generate_completion
from cad_llm.agent.parse import ParsedToolCall, has_incomplete_tool_call, parse_tool_calls
from cad_llm.agent.steps import AgentStep, RunState
from cad_llm.agent.summary import summarize_tool_call
from cad_llm.agent.transcript import append_event
from cad_llm.inference.extract import strip_model_response
from cad_llm.tools.workspace.project import ChatLayout

GenerateFn = Callable[[Any, Any, str, int], str]

TRUNCATED_TOOL_NUDGE = (
    "Your last tool call was cut off. Repeat the full <tool_call> block with complete JSON."
)
WRITE_NUDGE = (
    "Call write_file to create src/main.py now. Tool call only — no prose, no markdown code blocks."
)
SANDBOX_NUDGE = 'Call run_python_sandbox with entrypoint "src/main.py". Tool call only.'
WRITE_SYNTAX_NUDGE = (
    "Fix the Python and call write_file again. Raw Python only — no markdown fences. Tool call only."
)


@dataclass
class SubagentResult:
    final_text: str
    steps: list[AgentStep]


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
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _write_syntax_failed(output: str) -> bool:
    return output.startswith("error: Python syntax error")


def run_subagent(
    *,
    phase: str,
    system_prompt: str,
    user_message: str,
    tools: list[StructuredTool],
    model: Any,
    tokenizer: Any,
    chat: ChatLayout,
    max_steps: int,
    max_tokens: int,
    generate_fn: GenerateFn | None = None,
    on_step: Callable[[AgentStep], None] | None = None,
    require_sandbox_clean: bool = False,
    require_write: bool = False,
    after_src_edit: Callable[[RunState, list[dict[str, str]]], None] | None = None,
) -> SubagentResult:
    """Run a focused tool loop for one subagent phase."""
    tools_by_name = {tool.name: tool for tool in tools}
    openai_tools = _tools_for_template(tools)
    state = RunState()
    steps: list[AgentStep] = []

    def emit(step: AgentStep) -> None:
        step.phase = phase
        steps.append(step)
        if on_step is not None:
            on_step(step)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    final_text = ""

    for _ in range(max_steps):
        formatted = tokenizer.apply_chat_template(
            messages,
            tools=openai_tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if generate_fn is not None:
            raw = generate_fn(model, tokenizer, formatted, max_tokens)
        else:
            raw = generate_completion(model, tokenizer, formatted, max_tokens)

        if has_incomplete_tool_call(raw):
            messages.append({"role": "assistant", "content": raw})
            append_event(chat.transcript_path, "assistant", content=raw, phase=phase, truncated_tool=True)
            messages.append({"role": "user", "content": TRUNCATED_TOOL_NUDGE})
            emit(AgentStep(kind="nudge", content=TRUNCATED_TOOL_NUDGE, phase=phase))
            continue

        tool_calls = parse_tool_calls(raw)
        if not tool_calls:
            visible = strip_model_response(raw)
            if require_write and state.needs_write():
                messages.append({"role": "assistant", "content": raw})
                append_event(
                    chat.transcript_path,
                    "assistant",
                    content=raw,
                    phase=phase,
                    rejected=True,
                    reason="no_src_write",
                )
                messages.append({"role": "user", "content": WRITE_NUDGE})
                append_event(chat.transcript_path, "nudge", content=WRITE_NUDGE, phase=phase)
                emit(AgentStep(kind="nudge", content=WRITE_NUDGE, phase=phase))
                continue
            if require_sandbox_clean and state.needs_sandbox():
                messages.append({"role": "assistant", "content": raw})
                append_event(
                    chat.transcript_path,
                    "assistant",
                    content=raw,
                    phase=phase,
                    rejected=True,
                    reason="sandbox_not_run",
                )
                messages.append({"role": "user", "content": SANDBOX_NUDGE})
                emit(AgentStep(kind="nudge", content=SANDBOX_NUDGE, phase=phase))
                continue
            final_text = visible or raw.strip()
            append_event(chat.transcript_path, "assistant", content=final_text, phase=phase)
            emit(AgentStep(kind="assistant", content=final_text, phase=phase))
            break

        messages.append({"role": "assistant", "content": raw})
        append_event(chat.transcript_path, "assistant", content=raw, phase=phase, tool_calls=len(tool_calls))

        for call in tool_calls:
            output = _run_tool_call(
                call=call,
                tools_by_name=tools_by_name,
                messages=messages,
                chat=chat,
                emit=emit,
                phase=phase,
                state=state,
            )
            if call.name in {"write_file", "search_replace"} and _write_syntax_failed(output):
                nudge = f"{output}\n\n{WRITE_SYNTAX_NUDGE}"
                messages.append({"role": "user", "content": nudge})
                emit(AgentStep(kind="nudge", content=WRITE_SYNTAX_NUDGE, phase=phase))

        if after_src_edit is not None:
            after_src_edit(state, messages)
    else:
        final_text = f"Stopped: {phase} agent reached max steps."
        append_event(chat.transcript_path, "assistant", content=final_text, phase=phase, truncated=True)
        emit(AgentStep(kind="assistant", content=final_text, phase=phase))

    return SubagentResult(final_text=final_text, steps=steps)


def _run_tool_call(
    *,
    call: ParsedToolCall,
    tools_by_name: dict[str, StructuredTool],
    messages: list[dict[str, str]],
    chat: ChatLayout,
    emit: Callable[[AgentStep], None],
    phase: str,
    state: RunState,
) -> str:
    emit(
        AgentStep(
            kind="tool_call",
            content=summarize_tool_call(call.name, call.arguments),
            tool_name=call.name,
            tool_args=call.arguments,
            phase=phase,
        )
    )
    append_event(chat.transcript_path, "tool_call", name=call.name, phase=phase, arguments=call.arguments)

    tool = tools_by_name.get(call.name)
    if tool is None:
        output = f"error: unknown tool {call.name!r}"
    else:
        output = _execute_tool(tool, call.arguments)

    state.record_tool_result(call.name, call.arguments, output)
    append_event(chat.transcript_path, "tool_result", name=call.name, phase=phase, output=output)
    emit(
        AgentStep(
            kind="tool_result",
            content=output,
            tool_name=call.name,
            tool_args=call.arguments,
            phase=phase,
        )
    )
    messages.append({"role": "tool", "content": output})
    return output


def run_tool_call(
    *,
    call: ParsedToolCall,
    tools_by_name: dict[str, StructuredTool],
    messages: list[dict[str, str]],
    chat: ChatLayout,
    emit: Callable[[AgentStep], None],
    phase: str,
    state: RunState,
) -> str:
    return _run_tool_call(
        call=call,
        tools_by_name=tools_by_name,
        messages=messages,
        chat=chat,
        emit=emit,
        phase=phase,
        state=state,
    )
