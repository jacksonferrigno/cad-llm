"""Retrieve CadQuery documentation snippets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from cad_llm.tools.docs.loader import load_chunks_cache
from cad_llm.tools.docs.store import build_hybrid_retriever, build_vectorstore

_DEFAULT_RESULT_LIMIT = 2
_SNIPPET_CHARS = 600


def _format_documents(documents: list[Document]) -> str:
    if not documents:
        return "No CadQuery documentation matches found."

    parts: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        symbols = doc.metadata.get("symbols", [])
        symbol_line = ""
        if isinstance(symbols, list) and symbols:
            symbol_line = f"symbols: {', '.join(str(s) for s in symbols)}\n"
        snippet = doc.page_content[:_SNIPPET_CHARS]
        source = Path(str(doc.metadata.get("source", "cadquery"))).name
        parts.append(f"[{idx}] source={source}\n{symbol_line}{snippet}")
    return "\n\n".join(parts)


@dataclass
class _RetrieverCache:
    key: tuple[str, str, str, str, float]
    retriever: BaseRetriever


_retriever_cache: _RetrieverCache | None = None
_MAX_RETRIEVER_K = 5


def _cache_key(
    *,
    db_url: str,
    collection_name: str,
    embedding_model: str,
    chunks_cache: Path,
) -> tuple[str, str, str, str, float]:
    mtime = chunks_cache.stat().st_mtime if chunks_cache.exists() else 0.0
    return (
        db_url,
        collection_name,
        embedding_model,
        str(chunks_cache.resolve()),
        mtime,
    )


def _get_retriever(
    *,
    db_url: str,
    collection_name: str,
    embedding_model: str,
    chunks_cache: Path,
) -> BaseRetriever:
    global _retriever_cache

    key = _cache_key(
        db_url=db_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunks_cache=chunks_cache,
    )
    if _retriever_cache is not None and _retriever_cache.key == key:
        return _retriever_cache.retriever

    chunks = load_chunks_cache(chunks_cache)
    vectorstore = build_vectorstore(
        db_url=db_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        pre_delete_collection=False,
    )
    retriever = build_hybrid_retriever(vectorstore, chunks, k=_MAX_RETRIEVER_K)
    _retriever_cache = _RetrieverCache(key=key, retriever=retriever)
    return retriever


def search_cadquery_docs(
    query: str,
    *,
    db_url: str,
    collection_name: str,
    embedding_model: str,
    chunks_cache: str | Path,
    limit: int = _DEFAULT_RESULT_LIMIT,
) -> str:
    chunks_path = Path(chunks_cache)
    retriever = _get_retriever(
        db_url=db_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunks_cache=chunks_path,
    )
    documents = retriever.invoke(query)[:limit]
    return _format_documents(documents)
