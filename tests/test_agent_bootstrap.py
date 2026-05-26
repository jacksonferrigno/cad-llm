from unittest.mock import patch

from cad_llm.agent.bootstrap import bootstrap_system_prompt
from cad_llm.tools.workspace import create_chat, create_project


def test_bootstrap_loads_skill_and_docs(tmp_path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo01")
    chat = create_chat(project, title="t", chat_id="c1")

    with patch(
        "cad_llm.agent.bootstrap.search_cadquery_docs",
        return_value="[1] Workplane extrude example",
    ):
        prompt, steps, skill = bootstrap_system_prompt("Build a cube", chat)

    assert "cad-generation skill" in prompt
    assert "Workplane extrude example" in prompt
    assert skill
    assert len(steps) == 2
    assert steps[0].tool_name == "load_skill"
    assert steps[1].tool_name == "search_cadquery_docs"


def test_bootstrap_skips_docs_when_cached(tmp_path) -> None:
    projects_root = tmp_path / "projects"
    project = create_project(projects_root, name="demo", project_id="demo02")
    chat = create_chat(project, title="t", chat_id="c2")
    cached = "cached skill body"

    prompt, steps, skill = bootstrap_system_prompt(
        "Build a cube",
        chat,
        include_skill=False,
        include_docs=False,
        cached_skill=cached,
    )

    assert skill == cached
    assert "cached skill body" in prompt
    assert steps == []
