from cad_llm.agent.recovery import docs_query_from_failure


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
