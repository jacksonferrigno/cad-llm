"""Rich console rendering for agent runs."""

from __future__ import annotations

from rich.console import Console

from cad_llm.agent.steps import AgentStep
from cad_llm.agent.summary import summarize_tool_call, summarize_tool_result
from cad_llm.inference.extract import ReasoningStream, extract_reasoning_text, strip_model_response


class AgentConsole:
    """Compact tool output with optional dim reasoning stream."""

    def __init__(
        self, console: Console, *, stream: bool = True, show_reasoning: bool = False
    ) -> None:
        self.console = console
        self.stream = stream
        self.show_reasoning = show_reasoning
        self._reasoning = ReasoningStream()
        self._reasoning_started = False

    def begin_generation(self) -> None:
        self._reasoning = ReasoningStream()
        self._reasoning_started = False

    def on_token(self, token: str) -> None:
        if not self.stream or not self.show_reasoning:
            return
        delta = self._reasoning.feed(token)
        if not delta:
            return
        if not self._reasoning_started:
            self.console.print("[dim]think ›[/dim] ", end="", highlight=False)
            self._reasoning_started = True
        self.console.print(delta, end="", style="dim", highlight=False, soft_wrap=True)

    def end_generation(self) -> None:
        if not self.stream or not self.show_reasoning:
            return
        if not self._reasoning_started:
            tail = extract_reasoning_text(self._reasoning.buffer)
            if tail:
                self.console.print(f"[dim]think ›[/dim] {tail}", highlight=False)
                self._reasoning_started = True
        elif self._reasoning_started:
            self.console.print()

    def print_step(self, step: AgentStep) -> None:
        if step.kind == "bootstrap":
            label = "skill" if step.tool_name == "load_skill" else "docs"
            auto = (step.tool_args or {}).get("auto")
            suffix = " (auto)" if auto else ""
            self.console.print(f"[dim]setup[/dim]  {label}{suffix} ready")
            return

        if step.kind == "tool_call":
            assert step.tool_name is not None
            args = step.tool_args or {}
            summary = summarize_tool_call(step.tool_name, args)
            self.console.print(f"[cyan]→[/cyan] {summary}")
            return

        if step.kind == "tool_result":
            assert step.tool_name is not None
            summary = summarize_tool_result(step.tool_name, step.content)
            failed = "error" in summary.lower() or "exit_code=1" in summary
            style = "red" if failed else "green"
            self.console.print(f"[{style}]←[/{style}] {summary}")
            return

        if step.kind == "nudge":
            self.console.print(f"[dim]… {step.content[:80]}[/dim]")
            return

        if step.kind == "assistant":
            text = strip_model_response(step.content)
            if text:
                self.console.print(f"\n[bold]done[/bold]  {text}")
            return

        self.console.print(step.content)


def print_step(console: Console, step: AgentStep) -> None:
    AgentConsole(console, stream=False).print_step(step)


def print_paths(
    console: Console,
    *,
    project_root: str,
    src_dir: str,
    outputs_dir: str,
    transcript: str,
) -> None:
    console.print(
        f"[dim]project[/dim] {project_root}\n"
        f"[dim]src[/dim]     {src_dir}  ·  "
        f"[dim]outputs[/dim] {outputs_dir}\n"
        f"[dim]log[/dim]     {transcript}"
    )
