"""Token counting helpers for training data filtering."""

from __future__ import annotations

from typing import Protocol

from mlx_lm.utils import load_tokenizer

from cad_llm.inference.generate import SYSTEM_PROMPT


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        return_dict: bool = ...,
    ) -> list[int]: ...


def load_training_tokenizer(model_id: str) -> ChatTokenizer:
    return load_tokenizer(model_id)


def build_sft_messages(prompt: str, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def count_sft_messages(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> int:
    tokens = tokenizer.apply_chat_template(messages, return_dict=False)
    return len(tokens)


def count_sft_tokens(tokenizer: ChatTokenizer, prompt: str, completion: str) -> int:
    return count_sft_messages(tokenizer, build_sft_messages(prompt, completion))


def filter_rows_by_token_length(
    rows: list[dict[str, object]],
    tokenizer: ChatTokenizer,
    *,
    max_tokens: int,
) -> tuple[list[dict[str, object]], int]:
    kept: list[dict[str, object]] = []
    dropped = 0
    for row in rows:
        messages = row.get("messages")
        if messages is None:
            token_count = count_sft_tokens(
                tokenizer,
                str(row["prompt"]),
                str(row["completion"]),
            )
        else:
            token_count = count_sft_messages(tokenizer, messages)  # type: ignore[arg-type]
        if token_count <= max_tokens:
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped
