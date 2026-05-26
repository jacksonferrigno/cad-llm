"""Helpers for building doc search queries after failures."""

from __future__ import annotations

import re

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
        "stl",
        "attributeerror",
        "syntaxerror",
        "typeerror",
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
