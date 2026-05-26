"""Tests for agent run state and sandbox gating."""

from cad_llm.agent.steps import RunState


def test_run_state_requires_write_then_sandbox() -> None:
    state = RunState()
    assert state.needs_write()
    assert not state.needs_sandbox()

    state.record_tool_result("write_file", {"path": "src/main.py", "content": "x"}, "src/main.py")
    assert not state.needs_write()
    assert state.needs_sandbox()

    state.record_tool_result(
        "run_python_sandbox",
        {"entrypoint": "src/main.py"},
        "exit_code=0\n\nstdout:\nok",
    )
    assert not state.needs_sandbox()


def test_sandbox_before_write_does_not_satisfy_gate() -> None:
    state = RunState()
    state.record_tool_result(
        "run_python_sandbox",
        {"entrypoint": "src/main.py"},
        "error: entrypoint is not a file: /tmp/src/main.py",
    )
    assert state.needs_write()

    state.record_tool_result("write_file", {"path": "src/main.py", "content": "x"}, "src/main.py")
    assert state.needs_sandbox()


def test_docs_search_alone_does_not_clear_write_gate() -> None:
    state = RunState()
    state.record_tool_result("search_cadquery_docs", {"query": "Workplane.box"}, "hits")
    assert state.needs_write()


def test_failed_write_does_not_mark_src() -> None:
    state = RunState()
    state.record_tool_result(
        "write_file",
        {"path": "src/main.py", "content": "bad"},
        "error: Python syntax error in src/main.py: unexpected indent (line 1)",
    )
    assert state.needs_write()
    assert not state.needs_sandbox()


def test_run_state_ignores_non_src_writes() -> None:
    state = RunState()
    state.record_tool_result(
        "write_file",
        {"path": "outputs/readme.txt", "content": "x"},
        "outputs/readme.txt",
    )
    assert state.needs_write()
    assert not state.needs_sandbox()
