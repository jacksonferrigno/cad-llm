import json
from pathlib import Path
from unittest.mock import patch

from cad_llm.agent.orchestrator import is_build_request, run_orchestrated_agent


def test_is_build_request() -> None:
    assert is_build_request("Build a 50mm cube")
    assert not is_build_request("what should we cook up")
    assert not is_build_request("what should we build")


@patch("cad_llm.agent.orchestrator.run_implement_agent")
@patch("cad_llm.agent.orchestrator.run_research_agent")
def test_orchestrated_build_runs_research_then_implement(
    mock_research,
    mock_implement,
    tmp_path: Path,
) -> None:
    from cad_llm.agent.steps import AgentStep
    from cad_llm.agent.subagent import SubagentResult
    from cad_llm.tools.workspace import create_chat, create_project

    mock_research.return_value = SubagentResult(
        final_text="## Requirements\n- cube\n## CadQuery APIs\nbox cutThruAll\n## Implementation plan\n1. box",
        steps=[AgentStep(kind="assistant", content="brief", phase="research")],
    )
    mock_implement.return_value = SubagentResult(
        final_text="Cube exported.",
        steps=[AgentStep(kind="assistant", content="Cube exported.", phase="implement")],
    )

    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="orch1")
    chat = create_chat(project, title="test", chat_id="chat_orch1")

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        return "unused"

    result = run_orchestrated_agent(
        project,
        chat,
        "Build a 10mm cube",
        generate_fn=fake_generate,
    )

    mock_research.assert_called_once()
    mock_implement.assert_called_once()
    assert result.final_response == "Cube exported."
    events = [json.loads(line) for line in chat.transcript_path.read_text().strip().splitlines()]
    assert any(event.get("type") == "user" for event in events)


def test_orchestrated_chat_skips_subagents(tmp_path: Path) -> None:
    from cad_llm.tools.workspace import create_chat, create_project

    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="orch2")
    chat = create_chat(project, title="test", chat_id="chat_orch2")

    def fake_generate(_model, _tokenizer, _prompt: str, _max_tokens: int) -> str:
        return "Option A or B?"

    result = run_orchestrated_agent(
        project,
        chat,
        "what should we build",
        generate_fn=fake_generate,
    )

    assert "Option A or B?" in result.final_response
