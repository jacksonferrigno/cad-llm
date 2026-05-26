from unittest.mock import patch

from cad_llm.agent.recovery import build_sandbox_recovery_message, docs_query_from_failure


def test_docs_query_from_attribute_error() -> None:
    output = (
        "exit_code=1\n\nstderr:\n"
        "AttributeError: module 'cadquery.occ_impl.exporters' has no attribute 'step'"
    )
    query = docs_query_from_failure(output, user_prompt="export cube step")
    assert "export" in query
    assert "attributeerror" in query or "exporters" in query


def test_docs_query_fallback_to_prompt() -> None:
    query = docs_query_from_failure("exit_code=1", user_prompt="Build a 10mm cube")
    assert "cadquery" in query


@patch("cad_llm.agent.recovery.fetch_cadquery_docs", return_value="[1] cq.exporters.export(...)")
@patch("cad_llm.agent.recovery.load_skill", return_value="# debug")
def test_build_sandbox_recovery_includes_docs(_mock_skill, _mock_docs) -> None:
    output = "exit_code=1\n\nstderr:\nAttributeError: no attribute 'step'"
    short, full, query, docs = build_sandbox_recovery_message(output, user_prompt="export step")
    assert "cq.exporters.export" in docs
    assert query in full
    assert "cad-debug" in full
    assert "sandbox failed" in short
