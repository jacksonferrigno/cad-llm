"""Index CadQuery HTML into LangChain PGVector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cad_llm.tools.docs.loader import load_and_split_html, save_chunks_cache
from cad_llm.tools.docs.store import build_vectorstore


@dataclass(frozen=True)
class IndexSummary:
    html_path: Path
    chunks: int
    db_url: str
    collection_name: str
    chunks_cache: Path


def index_docs(
    html_path: Path,
    db_url: str,
    *,
    collection_name: str,
    embedding_model: str,
    chunks_cache: Path,
    reset: bool = True,
) -> IndexSummary:
    chunks = load_and_split_html(html_path)
    if not chunks:
        msg = f"No chunks extracted from {html_path}"
        raise ValueError(msg)

    vectorstore = build_vectorstore(
        db_url=db_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        pre_delete_collection=reset,
    )
    vectorstore.add_documents(chunks)
    save_chunks_cache(chunks, chunks_cache)

    return IndexSummary(
        html_path=html_path,
        chunks=len(chunks),
        db_url=db_url,
        collection_name=collection_name,
        chunks_cache=chunks_cache,
    )
