import json
from pathlib import Path

from cad_llm.agent.runner import DOC_LOOP_NUDGE, run_agent
from cad_llm.agent.transcript import append_event
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
    assert result.final_response == "Sandbox passed."


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
    assert result.final_response == "Sandbox passed."


def test_run_agent_injects_failure_context_on_sandbox_error(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo05")
    chat = create_chat(project, title="test", chat_id="chat05")

    responses = [
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "raise RuntimeError(\\"boom\\")\\n"}}\n</tool_call>'
        ),
        "Fixed.",
    ]
    captured: list[str] = []
    call_count = {"n": 0}

    def fake_generate(_model, _tokenizer, prompt: str, _max_tokens: int) -> str:
        captured.append(prompt)
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

    assert "Fixed." in result.final_response
    assert any("--- src/main.py ---" in prompt for prompt in captured)
    assert any("CAD debug" in prompt for prompt in captured)
    assert any("RuntimeError" in prompt for prompt in captured)

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(
        event.get("type") == "nudge" and event.get("content") == "sandbox_failure_context"
        for event in events
    )


def test_run_agent_does_not_auto_sandbox_after_docs_without_src_edit(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo06")
    chat = create_chat(project, title="test", chat_id="chat06")

    responses = [
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "raise RuntimeError(\\"boom\\")\\n"}}\n</tool_call>'
        ),
        (
            '<tool_call>\n{"name": "search_cadquery_docs", "arguments": '
            '{"query": "Workplane box"}}\n</tool_call>'
        ),
        "Fixed.",
    ]
    call_count = {"n": 0}

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[min(idx, len(responses) - 1)]

    run_agent(
        project,
        chat,
        "Build a cube",
        generate_fn=fake_generate,
        max_steps=4,
        bootstrap=False,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    sandbox_runs = [
        event
        for event in events
        if event.get("type") == "tool_result" and event.get("name") == "run_python_sandbox"
    ]
    assert len(sandbox_runs) == 1


def test_run_agent_follow_up_turn_includes_history_and_src(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo07")
    chat = create_chat(project, title="test", chat_id="chat07")

    src = (
        "import cadquery as cq\n\n"
        "result = cq.Workplane('XY').box(80, 80, 40).faces('>Z').workplane().circle(20).cutThruAll()\n"
    )
    (project.src_dir / "main.py").write_text(src, encoding="utf-8")

    append_event(chat.transcript_path, "user", content="build block with pocket")
    append_event(
        chat.transcript_path,
        "assistant",
        content="Used cutThruAll for the pocket.",
    )
    append_event(
        chat.transcript_path,
        "user",
        content="that cut through the whole block, not a pocket",
    )
    append_event(
        chat.transcript_path,
        "assistant",
        content="Right, that should be a blind cut with cutBlind instead.",
    )

    captured: list[str] = []

    def fake_generate(_model, _tokenizer, prompt: str, _max_tokens: int) -> str:
        captured.append(prompt)
        return "Updated to cutBlind."

    result = run_agent(
        project,
        chat,
        "make the change",
        generate_fn=fake_generate,
        max_steps=2,
        bootstrap=False,
    )

    assert "Updated to cutBlind." in result.final_response
    assert any("that cut through the whole block" in prompt for prompt in captured)
    assert any("cutBlind instead" in prompt for prompt in captured)
    assert any("--- current src/main.py ---" in prompt for prompt in captured)
    assert any("cutThruAll()" in prompt for prompt in captured)


def test_run_agent_nudges_after_repeated_doc_search_during_recovery(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo08")
    chat = create_chat(project, title="test", chat_id="chat08")

    responses = [
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "raise RuntimeError(\\"boom\\")\\n"}}\n</tool_call>'
        ),
        (
            '<tool_call>\n{"name": "search_cadquery_docs", "arguments": '
            '{"query": "sphere cut"}}\n</tool_call>'
        ),
        (
            '<tool_call>\n{"name": "search_cadquery_docs", "arguments": '
            '{"query": "sphere cut"}}\n</tool_call>'
        ),
        (
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"path": "src/main.py", "content": "print(1)\\n"}}\n</tool_call>'
        ),
    ]
    call_count = {"n": 0}

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[min(idx, len(responses) - 1)]

    run_agent(
        project,
        chat,
        "Build a block",
        generate_fn=fake_generate,
        max_steps=6,
        bootstrap=False,
    )

    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "nudge" and event.get("content") == DOC_LOOP_NUDGE for event in events)
