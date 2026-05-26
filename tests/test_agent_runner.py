import json
from pathlib import Path

from cad_llm.agent.runner import run_agent
from cad_llm.tools.workspace import create_chat, create_project


def test_run_agent_auto_sandbox_after_write(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo02")
    chat = create_chat(project, title="test", chat_id="chat02")

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

    result = run_agent(
        project,
        chat,
        "Build a cube",
        generate_fn=fake_generate,
        max_steps=3,
        bootstrap=False,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "auto_tool" for event in events)
    assert "Done." in result.final_response


def test_run_agent_accepts_brainstorm_without_write_nudge(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo03")
    chat = create_chat(project, title="test", chat_id="chat03")

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        return "Here are three options: bracket, spacer, or plate. Which do you want?"

    result = run_agent(
        project,
        chat,
        "what should we cook up",
        generate_fn=fake_generate,
        max_steps=3,
        bootstrap=False,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert "Which do you want?" in result.final_response
    assert not any(
        event.get("type") == "nudge" and "write_file" in event.get("content", "")
        for event in events
    )


def test_run_agent_sandbox_nudge_if_model_replies_before_auto_sandbox(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo04")
    chat = create_chat(project, title="test", chat_id="chat04")

    responses = [
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "print(1)\\n"}}\n</tool_call>'
        ),
        "Done!",
        (
            '<tool_call>\n{"name": "run_python_sandbox", "arguments": '
            '{"entrypoint": "src/main.py"}}\n</tool_call>'
        ),
        "Finished.",
    ]
    call_count = {"n": 0}

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[min(idx, len(responses) - 1)]

    result = run_agent(
        project,
        chat,
        "Build a cube",
        generate_fn=fake_generate,
        max_steps=5,
        bootstrap=False,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert not any(event.get("type") == "auto_docs" for event in events)
    assert "Finished." in result.final_response
