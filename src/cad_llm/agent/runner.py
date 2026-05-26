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
from cad_llm.agent.steps import AgentStep, RunState
from cad_llm.agent.summary import summarize_tool_call
from cad_llm.agent.transcript import append_event, read_chat_history, write_run_summary
from cad_llm.config import settings
from cad_llm.inference.extract import strip_model_response
from cad_llm.inference.generate import CadGenerator
from cad_llm.tools.binding import get_bound_agent_tools
from cad_llm.tools.edit.tool import read_file
from cad_llm.tools.skills.loader import load_skill
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout

SANDBOX_PASSED_MESSAGE = "Sandbox passed."
SANDBOX_NUDGE = 'Call run_python_sandbox with entrypoint "src/main.py". Tool call only.'
TRUNCATED_TOOL_NUDGE = (
    "Your last tool call was cut off. Repeat the full <tool_call> block with complete JSON."
)
MALFORMED_TOOL_NUDGE = (
    "Your last tool call was not valid JSON. Repeat it as exactly one <tool_call> block "
    'containing JSON like {"name": "write_file", "arguments": {...}}. Tool call only.'
)
WRITE_SYNTAX_NUDGE = (
    "Fix the Python and call write_file again. Raw Python only — no markdown fences. Tool call only."
)
NO_CHANGE_NUDGE = (
    "search_replace did not change the file. Make a material fix — patch the failing lines "
    "or rewrite src/main.py."
)


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


def _write_syntax_failed(output: str) -> bool:
    return output.startswith("error: Python syntax error")


def _force_sandbox_run(
    *,
    project: ProjectLayout,
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
        project=project,
    )


def _inject_failure_context(
    *,
    project: ProjectLayout,
    sandbox_output: str,
    messages: list[dict[str, str]],
    chat: ChatLayout,
) -> None:
    try:
        src = read_file(project.root, "src/main.py")
    except Exception:  # noqa: BLE001
        src = "(could not read src/main.py)"
    try:
        debug_skill = load_skill("cad-debug").strip()
    except Exception:  # noqa: BLE001
        debug_skill = "(cad-debug skill unavailable)"

    messages.append(
        {
            "role": "user",
            "content": (
                "Sandbox failed. Fix src/main.py using the debug workflow below.\n\n"
                f"--- sandbox output ---\n{sandbox_output}\n---\n\n"
                f"--- src/main.py ---\n{src}\n---\n\n"
                f"--- cad-debug ---\n{debug_skill}\n---"
            ),
        }
    )
    append_event(chat.transcript_path, "nudge", content="sandbox_failure_context")


def _complete_after_sandbox_success(
    *,
    chat: ChatLayout,
    emit: Callable[[AgentStep], None],
) -> str:
    append_event(chat.transcript_path, "assistant", content=SANDBOX_PASSED_MESSAGE)
    emit(AgentStep(kind="assistant", content=SANDBOX_PASSED_MESSAGE))
    return SANDBOX_PASSED_MESSAGE


def _build_initial_messages(
    *,
    system_prompt: str,
    prompt: str,
    chat: ChatLayout,
    project: ProjectLayout,
) -> tuple[list[dict[str, str]], bool]:
    history = read_chat_history(
        chat.transcript_path,
        exclude_last_user=prompt,
    )
    continuing = bool(history)

    user_content = prompt
    if continuing:
        try:
            src = read_file(project.root, "src/main.py")
        except Exception:  # noqa: BLE001
            pass
        else:
            user_content = (
                f"{prompt}\n\n"
                f"--- current src/main.py ---\n{src}\n---"
            )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages, continuing


def _auto_sandbox_after_src_edit(
    *,
    project: ProjectLayout,
    tools_by_name: dict[str, StructuredTool],
    messages: list[dict[str, str]],
    chat: ChatLayout,
    emit: Callable[[AgentStep], None],
    state: RunState,
) -> bool:
    if not state.needs_sandbox():
        return False
    output = _force_sandbox_run(
        project=project,
        tools_by_name=tools_by_name,
        messages=messages,
        chat=chat,
        emit=emit,
        state=state,
    )
    if _sandbox_failed(output):
        _inject_failure_context(
            project=project,
            sandbox_output=output,
            messages=messages,
            chat=chat,
        )
        return False
    return True


def run_agent(
    project: ProjectLayout,
    chat: ChatLayout,
    prompt: str,
    *,
    model_id: str | None = None,
    generator: CadGenerator | None = None,
    max_steps: int = 15,
    max_tokens: int | None = None,
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
    token_budget = max_tokens if max_tokens is not None else settings.agent_max_tokens

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

    messages, continuing = _build_initial_messages(
        system_prompt=system_prompt,
        prompt=prompt,
        chat=chat,
        project=project,
    )
    if continuing and (project.root / "src" / "main.py").is_file():
        state.has_src_write = True

    final_response = ""

    for _ in range(max_steps):
        formatted = tokenizer.apply_chat_template(
            messages,
            tools=openai_tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.enable_thinking,
        )

        if generate_fn is not None:
            raw = generate_fn(model, tokenizer, formatted, token_budget)
        else:
            raw = generate_completion(model, tokenizer, formatted, token_budget)

        if has_incomplete_tool_call(raw):
            messages.append({"role": "assistant", "content": raw})
            append_event(chat.transcript_path, "assistant", content=raw, truncated_tool=True)
            messages.append({"role": "user", "content": TRUNCATED_TOOL_NUDGE})
            append_event(chat.transcript_path, "nudge", content=TRUNCATED_TOOL_NUDGE)
            emit(AgentStep(kind="nudge", content=TRUNCATED_TOOL_NUDGE))
            continue

        try:
            tool_calls = parse_tool_calls(raw)
        except ValueError as exc:
            messages.append({"role": "assistant", "content": raw})
            append_event(
                chat.transcript_path,
                "assistant",
                content=raw,
                rejected=True,
                reason="malformed_tool_call",
                error=str(exc),
            )
            messages.append({"role": "user", "content": MALFORMED_TOOL_NUDGE})
            append_event(chat.transcript_path, "nudge", content=MALFORMED_TOOL_NUDGE)
            emit(AgentStep(kind="nudge", content=MALFORMED_TOOL_NUDGE))
            continue

        if not tool_calls:
            visible = strip_model_response(raw)

            if state.needs_sandbox():
                messages.append({"role": "assistant", "content": raw})
                append_event(
                    chat.transcript_path,
                    "assistant",
                    content=raw,
                    rejected=True,
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
                project=project,
            )
            if call.name == "run_python_sandbox" and _sandbox_failed(output):
                _inject_failure_context(
                    project=project,
                    sandbox_output=output,
                    messages=messages,
                    chat=chat,
                )
            if call.name in {"write_file", "search_replace"} and output == "no_change":
                messages.append({"role": "user", "content": NO_CHANGE_NUDGE})
                append_event(chat.transcript_path, "nudge", content=NO_CHANGE_NUDGE)
                emit(AgentStep(kind="nudge", content=NO_CHANGE_NUDGE))
            elif call.name in {"write_file", "search_replace"} and _write_syntax_failed(output):
                nudge = f"{output}\n\n{WRITE_SYNTAX_NUDGE}"
                messages.append({"role": "user", "content": nudge})
                append_event(chat.transcript_path, "nudge", content=WRITE_SYNTAX_NUDGE)
                emit(AgentStep(kind="nudge", content=WRITE_SYNTAX_NUDGE))

        if _auto_sandbox_after_src_edit(
            project=project,
            tools_by_name=tools_by_name,
            messages=messages,
            chat=chat,
            emit=emit,
            state=state,
        ):
            final_response = _complete_after_sandbox_success(chat=chat, emit=emit)
            break
    else:
        final_response = "Stopped: reached max agent steps without a final reply."
        append_event(chat.transcript_path, "assistant", content=final_response, truncated=True)
        emit(AgentStep(kind="assistant", content=final_response))

    if state.needs_sandbox():
        if _auto_sandbox_after_src_edit(
            project=project,
            tools_by_name=tools_by_name,
            messages=messages,
            chat=chat,
            emit=emit,
            state=state,
        ):
            final_response = _complete_after_sandbox_success(chat=chat, emit=emit)
        elif not final_response:
            final_response = "Stopped: reached max agent steps."

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
    project: ProjectLayout | None = None,
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
