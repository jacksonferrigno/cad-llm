import json

import pytest

from cad_llm.agent.parse import parse_tool_calls


def test_parse_single_tool_call() -> None:
    text = (
        "Some preamble\n<tool_call>\n"
        '{"name": "load_skill", "arguments": {"name": "cad-generation"}}\n'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "load_skill"
    assert calls[0].arguments == {"name": "cad-generation"}


def test_parse_tool_call_with_string_arguments() -> None:
    text = (
        "<tool_call>{"
        '"name": "write_file", '
        '"arguments": "{\\"path\\": \\"src/main.py\\", \\"content\\": \\"print(1)\\"}"'
        "}</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls[0].name == "write_file"
    assert calls[0].arguments["path"] == "src/main.py"


def test_parse_no_tool_calls() -> None:
    assert parse_tool_calls("All done.") == []


def test_has_incomplete_tool_call() -> None:
    from cad_llm.agent.parse import has_incomplete_tool_call

    assert has_incomplete_tool_call('<tool_call>\n{"name": "write_file"')
    assert not has_incomplete_tool_call(
        '<tool_call>\n{"name": "write_file", "arguments": {}}\n</tool_call>'
    )


def test_parse_invalid_payload_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_tool_calls("<tool_call>not-json</tool_call>")
