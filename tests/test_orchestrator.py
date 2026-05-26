import json
from pathlib import Path
from unittest.mock import patch

from cad_llm.agent.runner import run_agent


@patch("cad_llm.agent.runner.generate_completion")
def test_run_agent_brainstorm_reply(mock_generate, tmp_path: Path) -> None:
    from cad_llm.tools.workspace import create_chat, create_project

    mock_generate.return_value = "Option A or B?"

    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="agent1")
    chat = create_chat(project, title="test", chat_id="chat_agent1")

    result = run_agent(
        project,
        chat,
        "what should we build",
        generate_fn=lambda _m, _t, _p, _n: "Option A or B?",
    )

    assert "Option A or B?" in result.final_response
    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "user" for event in events)


@patch("cad_llm.agent.runner.generate_completion")
def test_run_agent_build_writes_code(mock_generate, tmp_path: Path) -> None:
    from cad_llm.tools.workspace import create_chat, create_project

    responses = [
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

    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="agent2")
    chat = create_chat(project, title="test", chat_id="chat_agent2")

    result = run_agent(
        project,
        chat,
        "Build a 10mm cube",
        generate_fn=fake_generate,
    )

    assert result.final_response == "Done."
    assert (project.src_dir / "main.py").read_text(encoding="utf-8") == "print(1)\n"
