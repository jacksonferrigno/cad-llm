import re

_FENCE_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Pull Python from markdown fences, or return trimmed raw text."""
    blocks = _FENCE_PATTERN.findall(text)
    if blocks:
        return max(blocks, key=len).strip()

    return text.strip()
