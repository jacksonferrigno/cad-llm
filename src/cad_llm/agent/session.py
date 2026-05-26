"""Interactive agent session (model loaded once, many turns)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cad_llm.agent.runner import AgentRunResult, GenerateFn, run_agent
from cad_llm.inference.generate import CadGenerator
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout


@dataclass
class AgentSession:
    project: ProjectLayout
    chat: ChatLayout
    generator: CadGenerator
    max_steps: int = 15
    max_tokens: int = 2048
    _turn_count: int = field(default=0, repr=False)
    _skill_content: str | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        project: ProjectLayout,
        chat: ChatLayout,
        *,
        model_id: str | None = None,
        generator: CadGenerator | None = None,
        max_steps: int = 15,
        max_tokens: int = 2048,
    ) -> AgentSession:
        gen = generator or CadGenerator(model_id=model_id)
        if gen.model is None:
            gen.load()
        return cls(
            project=project,
            chat=chat,
            generator=gen,
            max_steps=max_steps,
            max_tokens=max_tokens,
        )

    def run_turn(
        self,
        prompt: str,
        *,
        on_step: Any = None,
        generate_fn: GenerateFn | None = None,
    ) -> AgentRunResult:
        first_turn = self._turn_count == 0
        self._turn_count += 1

        result = run_agent(
            self.project,
            self.chat,
            prompt,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            on_step=on_step,
            generate_fn=generate_fn,
            bootstrap=True,
            bootstrap_docs=first_turn,
            bootstrap_skill=first_turn,
            cached_skill=self._skill_content,
            generator=self.generator,
        )
        if result.skill_content:
            self._skill_content = result.skill_content
        return result
