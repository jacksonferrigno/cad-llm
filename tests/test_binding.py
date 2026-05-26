from pathlib import Path

from cad_llm.tools.binding import get_bound_agent_tools


def test_bound_tools_omit_project_root_from_schema(tmp_path: Path) -> None:
    tools = get_bound_agent_tools(tmp_path)
    names = {tool.name for tool in tools}
    assert names == {
        "search_cadquery_docs",
        "read_file",
        "grep",
        "write_file",
        "search_replace",
        "delete_file",
        "run_python_sandbox",
        "load_skill",
    }

    for tool in tools:
        if tool.name in {"search_cadquery_docs", "load_skill"}:
            continue
        assert tool.args_schema is not None
        schema = tool.args_schema.model_json_schema()
        assert "project_root" not in schema.get("properties", {})


def test_bound_write_file_uses_project_root(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in get_bound_agent_tools(tmp_path)}
    result = tools["write_file"].invoke({"path": "src/main.py", "content": "print('hi')\n"})
    assert result == "src/main.py"
    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "print('hi')\n"
