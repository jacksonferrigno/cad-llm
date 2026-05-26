import json
from pathlib import Path

from cad_llm.agent.subagent import run_subagent
from cad_llm.tools.binding import get_bound_implement_tools
from cad_llm.tools.workspace import create_chat, create_project


class _FakeTokenizer:
    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "prompt"


def test_run_subagent_write_gate(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="sub1")
    chat = create_chat(project, title="test", chat_id="chat_sub1")

    responses = [
        "I'll write the code now.",
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "print(1)\\n"}}\n</tool_call>'
        ),
        "Done.",
    ]
    call_count = {"n": 0}

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[min(idx, len(responses) - 1)]

    result = run_subagent(
        phase="implement",
        system_prompt="test",
        user_message="build",
        tools=get_bound_implement_tools(project.root),
        model=object(),
        tokenizer=_FakeTokenizer(),
        chat=chat,
        max_steps=5,
        max_tokens=512,
        generate_fn=fake_generate,
        require_write=True,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "nudge" and "write_file" in event.get("content", "") for event in events)
    assert result.final_text == "Done."
