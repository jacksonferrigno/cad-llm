import re

_FENCE_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINKING_BLOCK = re.compile(
    r"<think>.*?(?:</think>)?",
    re.DOTALL,
)
_THINKING_OPEN = "<think>"
_THINKING_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"


def _thinking_bounds(raw: str) -> tuple[int, int] | None:
    start = raw.find(_THINKING_OPEN)
    if start == -1:
        return None
    inner_start = start + len(_THINKING_OPEN)
    end = raw.find(_THINKING_CLOSE, inner_start)
    if end == -1:
        return inner_start, len(raw)
    return inner_start, end


def _strip_thinking(text: str) -> str:
    cleaned = _THINKING_BLOCK.sub("", text)
    if _THINKING_CLOSE in cleaned:
        cleaned = cleaned.rsplit(_THINKING_CLOSE, maxsplit=1)[-1]
    if _THINKING_OPEN in cleaned:
        cleaned = cleaned.split(_THINKING_OPEN, maxsplit=1)[-1]
    return cleaned.strip()


def strip_model_response(text: str) -> str:
    """Remove Qwen thinking blocks and return user-visible text."""
    return _strip_thinking(text)


def extract_reasoning_text(raw: str, *, max_chars: int = 800) -> str:
    """Return reasoning/thinking portion of model output for display."""
    bounds = _thinking_bounds(raw)
    if bounds is not None:
        inner = raw[bounds[0] : bounds[1]].strip()
        if len(inner) > max_chars:
            return inner[: max_chars - 1] + "…"
        return inner

    if _TOOL_CALL_OPEN in raw:
        prefix = raw.split(_TOOL_CALL_OPEN, maxsplit=1)[0].strip()
    else:
        prefix = raw.strip()

    if not prefix:
        return ""

    if len(prefix) > max_chars:
        return prefix[: max_chars - 1] + "…"
    return prefix


class ReasoningStream:
    """Incrementally surface reasoning tokens during generation."""

    def __init__(self, *, max_chars: int = 800) -> None:
        self._buffer = ""
        self._emitted = 0
        self._max_chars = max_chars

    def feed(self, token: str) -> str:
        self._buffer += token
        visible = extract_reasoning_text(self._buffer, max_chars=self._max_chars)
        delta = visible[self._emitted :]
        self._emitted = len(visible)
        return delta

    @property
    def buffer(self) -> str:
        return self._buffer


def extract_python_code(text: str) -> str:
    """Pull Python from model output, omitting Qwen thinking blocks."""
    blocks = _FENCE_PATTERN.findall(text)
    if blocks:
        return _strip_thinking(max(blocks, key=len)).strip()

    return _strip_thinking(text)
