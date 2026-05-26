"""Append-only JSONL transcript for agent runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def append_event(path: Path, event_type: str, **payload: Any) -> dict[str, Any]:
    record = {"ts": _now_iso(), "type": event_type, **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def write_run_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _assistant_history_content(content: str) -> str:
    from cad_llm.inference.extract import strip_model_response

    text = strip_model_response(content)
    if "<tool_call>" in text:
        text = text.split("<tool_call>", maxsplit=1)[0].strip()
    return text


def read_chat_history(
    path: Path,
    *,
    max_messages: int = 8,
    exclude_last_user: str | None = None,
) -> list[dict[str, str]]:
    """Return prior user/assistant turns for session continuation."""
    if not path.is_file():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if exclude_last_user is not None:
        for index in range(len(records) - 1, -1, -1):
            record = records[index]
            if record.get("type") == "user" and record.get("content") == exclude_last_user:
                records = records[:index]
                break

    history: list[dict[str, str]] = []
    for record in records:
        event_type = record.get("type")
        if event_type == "user":
            content = str(record.get("content", "")).strip()
            if content:
                history.append({"role": "user", "content": content})
        elif event_type == "assistant":
            content = _assistant_history_content(str(record.get("content", "")))
            if content:
                history.append({"role": "assistant", "content": content})

    if len(history) > max_messages:
        history = history[-max_messages:]
    return history
