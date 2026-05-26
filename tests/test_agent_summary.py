from cad_llm.agent.summary import summarize_tool_call, summarize_tool_result


def test_summarize_tool_call() -> None:
    assert summarize_tool_call("write_file", {"path": "src/main.py"}) == "write src/main.py"
    assert summarize_tool_call("run_python_sandbox", {}) == "run src/main.py"


def test_summarize_tool_result_sandbox_error() -> None:
    output = "exit_code=1\n\nstderr:\nIndentationError: unexpected indent"
    assert "IndentationError" in summarize_tool_result("run_python_sandbox", output)
