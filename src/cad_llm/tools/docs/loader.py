"""LangChain helpers for CadQuery doc RAG."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SYMBOL = re.compile(
    r"\b(?:cadquery(?:\.\w+)?|Workplane(?:\.\w+)?|Sketch(?:\.\w+)?)\b"
    r"|\b(?:extrude|cut|union|intersect|fillet|chamfer|circle|rect|box|polygon|"
    r"cutThruAll|faces|edges|vertices|translate|rotate|mirror|loft|sweep|revolve)\b",
    re.IGNORECASE,
)

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150


def langchain_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql+psycopg://"):
        return db_url
    return db_url.replace("postgresql://", "postgresql+psycopg://", 1)


def extract_symbols(text: str) -> list[str]:
    return sorted({match.lower() for match in _SYMBOL.findall(text)})


def enrich_metadata(document: Document) -> Document:
    symbols = extract_symbols(document.page_content)
    document.metadata.setdefault("source", "cadquery-latest")
    if symbols:
        document.metadata["symbols"] = symbols
    return document


def load_and_split_html(
    html_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    loader = BSHTMLLoader(
        str(html_path),
        open_encoding="utf-8",
        bs_kwargs={"features": "html.parser"},
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return [enrich_metadata(chunk) for chunk in chunks if chunk.page_content.strip()]


def save_chunks_cache(chunks: list[Document], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(chunks))


def load_chunks_cache(cache_path: Path) -> list[Document]:
    if not cache_path.exists():
        msg = f"Missing chunk cache at {cache_path}. Run `cad-llm docs index` first."
        raise FileNotFoundError(msg)
    return pickle.loads(cache_path.read_bytes())  # noqa: S301
