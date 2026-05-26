"""Retrieve CadQuery documentation snippets for agent tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from cad_llm.docs.loader import load_chunks_cache
from cad_llm.docs.store import build_hybrid_retriever, build_vectorstore


@dataclass(frozen=True)
class DocHit:
    content: str
    metadata: dict[str, object]


def _format_documents(documents: list[Document]) -> str:
    if not documents:
        return "No CadQuery documentation matches found."

    parts: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        symbols = doc.metadata.get("symbols", [])
        symbol_line = ""
        if isinstance(symbols, list) and symbols:
            symbol_line = f"symbols: {', '.join(str(s) for s in symbols)}\n"
        snippet = doc.page_content[:1200]
        source = doc.metadata.get("source", "cadquery")
        parts.append(f"[{idx}] source={source}\n{symbol_line}{snippet}")
    return "\n\n".join(parts)


def search_cadquery_docs(
    query: str,
    *,
    db_url: str,
    collection_name: str,
    embedding_model: str,
    chunks_cache: str | Path,
    limit: int = 5,
) -> str:
    chunks = load_chunks_cache(Path(chunks_cache))
    vectorstore = build_vectorstore(
        db_url=db_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        pre_delete_collection=False,
    )
    retriever = build_hybrid_retriever(vectorstore, chunks, k=limit)
    documents = retriever.invoke(query)
    return _format_documents(documents)
