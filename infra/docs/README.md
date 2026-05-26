# CadQuery docs RAG (LangChain + pgvector)

## 1. Start Postgres

```bash
docker compose -f infra/docker-compose.yml up -d
```

DB: `postgresql://cadllm:cadllm@127.0.0.1:5433/cadllm`

## 2. Index HTML docs

Docs live at `data/cadquery-latest/index.html` (bundled Sphinx export).

```bash
uv sync
uv run cad-llm docs index
```

This uses LangChain:

- **Load:** `BSHTMLLoader`
- **Chunk:** `RecursiveCharacterTextSplitter` (1200 chars, 150 overlap)
- **Store:** `langchain-postgres` `PGVector`
- **Keyword sidecar:** BM25 chunk cache at `artifacts/docs/cadquery_chunks.pkl`

## 3. Test retrieval

```bash
uv run cad-llm docs search "Workplane circle extrude cut through hole"
```

Hybrid retrieval = **PGVector similarity + BM25** via LangChain `EnsembleRetriever`.

## Next (not built yet)

- MCP HTTP server exposing `search_cadquery_docs`
- Agent loop in bench (`--with-docs`) that calls the tool before codegen
