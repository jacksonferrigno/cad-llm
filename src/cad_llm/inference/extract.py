import re

_FENCE_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINKING_BLOCK = re.compile(
    r"<think>.*?</think>",
    re.DOTALL,
)
_THINKING_OPEN = "<think>"
_THINKING_CLOSE = "</think>"


def _strip_thinking(text: str) -> str:
    cleaned = _THINKING_BLOCK.sub("", text)
    if _THINKING_CLOSE in cleaned:
        cleaned = cleaned.rsplit(_THINKING_CLOSE, maxsplit=1)[-1]
    if _THINKING_OPEN in cleaned:
        cleaned = cleaned.split(_THINKING_OPEN, maxsplit=1)[-1]
    return cleaned.strip()


def extract_python_code(text: str) -> str:
    """Pull Python from model output, omitting Qwen thinking blocks."""
    blocks = _FENCE_PATTERN.findall(text)
    if blocks:
        return _strip_thinking(max(blocks, key=len)).strip()

    return _strip_thinking(text)
