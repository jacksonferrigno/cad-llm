import json
from pathlib import Path

from cad_llm.agent.runner import run_agent
from cad_llm.tools.workspace import create_chat, create_project


def test_run_agent_rejects_early_exit_without_sandbox(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo01")
    chat = create_chat(project, title="test", chat_id="chat01")

    responses = [
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "print(1)\\n"}}\n</tool_call>'
        ),
        "Done! Exported cube.step successfully.",
        (
            '<tool_call>\n{"name": "run_python_sandbox", "arguments": '
            '{"entrypoint": "src/main.py"}}\n</tool_call>'
        ),
        "Sandbox passed.",
    ]
    call_count = {"n": 0}

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[min(idx, len(responses) - 1)]

    result = run_agent(
        project,
        chat,
        "Build a 10mm cube",
        generate_fn=fake_generate,
        max_steps=6,
        bootstrap=False,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "nudge" for event in events)
    assert "run_python_sandbox" in result.final_response.lower() or any(
        event.get("name") == "run_python_sandbox" for event in events if event.get("type") == "tool_call"
    )
    assert call_count["n"] >= 3


def test_run_agent_auto_sandbox_when_model_never_runs(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo02")
    chat = create_chat(project, title="test", chat_id="chat02")

    (project.src_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")

    responses = [
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "print(1)\\n"}}\n</tool_call>'
        ),
        "All done.",
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
        max_steps=2,
        bootstrap=False,
    )

    assert "Auto-ran sandbox" in result.final_response
    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "auto_tool" for event in events)
