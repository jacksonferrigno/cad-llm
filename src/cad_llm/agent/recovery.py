"""Auto doc search and recovery context after sandbox failures."""

from __future__ import annotations

import re

from cad_llm.config import settings
from cad_llm.tools.docs.search import search_cadquery_docs
from cad_llm.tools.skills.loader import load_skill

_ERROR_LINE = re.compile(r"^\s*(?:\w+Error|error:)\s*:?\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_API_TOKEN = re.compile(r"cadquery[\w.]*|cq\.\w+|exporters?(?:\.\w+)?|Workplane|\.\w+\(\)")


def docs_query_from_failure(sandbox_output: str, *, user_prompt: str = "") -> str:
    """Build a focused CadQuery docs query from sandbox stderr/stdout."""
    text = sandbox_output.lower()
    terms: list[str] = []

    for kw in (
        "export",
        "exporters",
        "step",
        "stp",
        "stl",
        "attributeerror",
        "syntaxerror",
        "indentationerror",
        "modulenotfounderror",
        "workplane",
        "box",
        "extrude",
    ):
        if kw in text:
            terms.append(kw)

    for match in _ERROR_LINE.finditer(sandbox_output):
        fragment = match.group(1).strip()[:80]
        for token in _API_TOKEN.findall(fragment):
            cleaned = token.replace("cadquery.", "").strip(".")
            if cleaned and cleaned not in terms:
                terms.append(cleaned)

    if not terms and user_prompt:
        terms.extend(user_prompt.split()[:6])

    if not terms:
        return "cadquery exporters export step"

    deduped = list(dict.fromkeys(terms))
    return "cadquery " + " ".join(deduped[:8])


def fetch_cadquery_docs(query: str, *, limit: int = 5) -> str:
    try:
        return search_cadquery_docs(
            query,
            db_url=settings.docs_db_url,
            collection_name=settings.docs_collection_name,
            embedding_model=settings.docs_embedding_model,
            chunks_cache=settings.resolve(settings.docs_chunks_cache),
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        return f"(CadQuery docs unavailable: {exc})"


def build_sandbox_recovery_message(
    sandbox_output: str,
    *,
    user_prompt: str = "",
) -> tuple[str, str, str, str]:
    """Return (short_nudge, full_message_for_model, docs_query, docs_text)."""
    query = docs_query_from_failure(sandbox_output, user_prompt=user_prompt)
    docs = fetch_cadquery_docs(query)
    debug_skill = load_skill("cad-debug")

    short = f"sandbox failed — searched docs for: {query!r}"

    full = (
        "Sandbox failed. You MUST fix the code using the CadQuery documentation below.\n"
        "Do NOT guess API names. Read the doc snippets, then use write_file or search_replace, "
        "then run_python_sandbox again.\n"
        "Respond with tool calls only — no prose.\n\n"
        f"## Doc search query\n{query}\n\n"
        f"## CadQuery documentation\n{docs}\n\n"
        f"## cad-debug skill\n{debug_skill}\n\n"
        f"## Sandbox output\n{sandbox_output}"
    )
    return short, full, query, docs
