"""Model generation helpers (streaming and batched)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlx_lm import generate, stream_generate

TokenCallback = Callable[[str], None]


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    *,
    on_token: TokenCallback | None = None,
) -> str:
    if on_token is None:
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    chunks: list[str] = []
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        token = response.text
        chunks.append(token)
        on_token(token)
    return "".join(chunks)
