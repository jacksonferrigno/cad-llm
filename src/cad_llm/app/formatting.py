"""Format agent steps as terminal-style log lines."""

from __future__ import annotations

import re

from cad_llm.agent.steps import AgentStep
from cad_llm.agent.summary import summarize_tool_call, summarize_tool_result
from cad_llm.inference.extract import strip_model_response

_FENCE_BLOCK = re.compile(r"```(?:python)?\s*\n.*?```", re.DOTALL | re.IGNORECASE)
_INLINE_CODE = re.compile(r"`([^`]+)`")


def format_assistant_reply(text: str, *, max_chars: int = 480) -> str:
    """Short, readable agent reply — no code dumps."""
    cleaned = strip_model_response(text)
    cleaned = _FENCE_BLOCK.sub("", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return ""
    first_block = cleaned.split("\n\n")[0].strip()
    summary = first_block if len(first_block) < len(cleaned) * 0.85 else cleaned
    if len(summary) > max_chars:
        return summary[: max_chars - 1].rstrip() + "…"
    return summary


def format_step(step: AgentStep) -> tuple[str, str]:
    """Return (tag, line) for the transcript pane. tag drives color."""
    prefix = f"{step.phase} › " if step.phase else ""

    if step.kind == "phase":
        return "setup", f"setup  {step.content}"

    if step.kind == "bootstrap":
        label = "skill" if step.tool_name == "load_skill" else "docs"
        auto = (step.tool_args or {}).get("auto")
        suffix = " (auto)" if auto else ""
        return "setup", f"setup  {label}{suffix} ready"

    if step.kind == "tool_call":
        assert step.tool_name is not None
        summary = summarize_tool_call(step.tool_name, step.tool_args or {})
        return "tool_out", f"{prefix}→ {summary}"

    if step.kind == "tool_result":
        assert step.tool_name is not None
        summary = summarize_tool_result(step.tool_name, step.content)
        failed = "error" in summary.lower() or "exit_code=1" in summary
        return "error" if failed else "tool_in", f"{prefix}← {summary}"

    if step.kind == "nudge":
        return "nudge", f"… {prefix}{step.content[:100]}"

    if step.kind == "assistant":
        text = format_assistant_reply(step.content)
        if text:
            label = prefix.rstrip(" › ") or "agent"
            return "done", f"{label} › {text}" if prefix else text
        return "muted", ""

    if step.content:
        return "muted", step.content
    return "muted", ""
