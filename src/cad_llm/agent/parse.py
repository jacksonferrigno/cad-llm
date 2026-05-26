"""Parse Qwen-style tool calls from model output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_INCOMPLETE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*)\Z", re.DOTALL)
_FUNCTION_BLOCK = re.compile(r"<function=([\w_]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_BLOCK = re.compile(r"<parameter=([\w_]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def has_incomplete_tool_call(text: str) -> bool:
    """True when generation ended mid tool-call."""
    if "<tool_call>" not in text:
        return False
    complete = list(_TOOL_CALL_BLOCK.finditer(text))
    tail = text.rsplit("<tool_call>", maxsplit=1)[-1]
    return "</tool_call>" not in tail and not complete


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, object]


def _parse_tool_call_payload(raw: str) -> ParsedToolCall:
    function_match = _FUNCTION_BLOCK.fullmatch(raw.strip())
    if function_match is not None:
        name = function_match.group(1).strip()
        body = function_match.group(2)
        arguments = {
            match.group(1).strip(): match.group(2).strip()
            for match in _PARAMETER_BLOCK.finditer(body)
        }
        return ParsedToolCall(name=name, arguments=arguments)

    data = json.loads(raw)
    if not isinstance(data, dict):
        msg = f"Tool call payload must be a JSON object, got {type(data).__name__}"
        raise ValueError(msg)

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tool call missing non-empty 'name'")

    arguments = data.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        msg = f"Tool call arguments must be an object, got {type(arguments).__name__}"
        raise ValueError(msg)

    return ParsedToolCall(name=name.strip(), arguments=arguments)


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    """Extract zero or more ``<tool_call>{...}</tool_call>`` blocks."""
    calls: list[ParsedToolCall] = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        payload = match.group(1).strip()
        if not payload:
            continue
        calls.append(_parse_tool_call_payload(payload))
    return calls
