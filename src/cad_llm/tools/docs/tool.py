"""LangChain tool wrapper for CadQuery documentation search."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cad_llm.config import settings
from cad_llm.tools.docs.search import search_cadquery_docs


class SearchCadQueryDocsInput(BaseModel):
    query: str = Field(
        description="CadQuery API or feature to look up, e.g. Workplane extrude cutThruAll"
    )


def search_cadquery_docs_tool(query: str) -> str:
    return search_cadquery_docs(
        query,
        db_url=settings.docs_db_url,
        collection_name=settings.docs_collection_name,
        embedding_model=settings.docs_embedding_model,
        chunks_cache=settings.resolve(settings.docs_chunks_cache),
    )


def get_search_cadquery_docs_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=search_cadquery_docs_tool,
        name="search_cadquery_docs",
        description="Search local CadQuery documentation for API usage and examples.",
        args_schema=SearchCadQueryDocsInput,
    )
