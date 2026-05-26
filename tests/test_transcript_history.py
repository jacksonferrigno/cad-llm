import json

from cad_llm.agent.transcript import append_event, read_chat_history


def test_read_chat_history_returns_prior_user_assistant_turns(tmp_path) -> None:
    path = tmp_path / "transcript.jsonl"
    append_event(path, "user", content="build a cube")
    append_event(
        path,
        "assistant",
        content="Done.\n\n<tool_call>\n{\"name\": \"write_file\"}\n</tool_call>",
    )
    append_event(path, "tool_call", name="write_file")
    append_event(path, "user", content="make it bigger")

    history = read_chat_history(path, exclude_last_user="make it bigger")

    assert history == [
        {"role": "user", "content": "build a cube"},
        {"role": "assistant", "content": "Done."},
    ]


def test_read_chat_history_limits_messages(tmp_path) -> None:
    path = tmp_path / "transcript.jsonl"
    for index in range(6):
        append_event(path, "user", content=f"user-{index}")
        append_event(path, "assistant", content=f"assistant-{index}")

    history = read_chat_history(path, exclude_last_user="user-5", max_messages=2)

    assert history == [
        {"role": "user", "content": "user-4"},
        {"role": "assistant", "content": "assistant-4"},
    ]


def test_read_chat_history_skips_empty_assistant_tool_only(tmp_path) -> None:
    path = tmp_path / "transcript.jsonl"
    append_event(path, "user", content="build")
    append_event(path, "assistant", content="<tool_call>{\"name\": \"write_file\"}</tool_call>")
    append_event(path, "user", content="fix it")

    history = read_chat_history(path, exclude_last_user="fix it")

    assert history == [{"role": "user", "content": "build"}]
