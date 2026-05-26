"""LangChain PGVector store + hybrid retrieval."""

from __future__ import annotations

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres.vectorstores import PGVector
from langchain_classic.retrievers import EnsembleRetriever

from cad_llm.docs.loader import langchain_db_url


def build_embeddings(model_name: str) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=model_name)


def build_vectorstore(
    *,
    db_url: str,
    collection_name: str,
    embedding_model: str,
    pre_delete_collection: bool = False,
) -> PGVector:
    return PGVector(
        embeddings=build_embeddings(embedding_model),
        collection_name=collection_name,
        connection=langchain_db_url(db_url),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def build_hybrid_retriever(
    vectorstore: PGVector,
    chunks: list[Document],
    *,
    k: int = 5,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> BaseRetriever:
    """Vector search + BM25 with keyword/symbol boosting via ensemble."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    vector = vectorstore.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(
        retrievers=[vector, bm25],
        weights=[vector_weight, keyword_weight],
    )
