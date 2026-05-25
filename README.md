# Local CAD LLM

Fine-tune a 4B coder model on Apple Silicon to generate CadQuery/build123d from plain English — fully local, no cloud GPU.

See [overview.md](overview.md) for the full roadmap.

## Requirements

- Apple Silicon Mac (MLX training/inference)
- macOS 13+
- [uv](https://docs.astral.sh/uv/) installed

## Setup

```bash
# Install all deps (core + MLX + CAD + web + dev tools)
uv sync

# Create local data/ and artifacts/ directories
uv run cad-llm ensure-dirs

# Sanity check
make verify
```

## Project layout

```
cad-llm/
├── src/cad_llm/          # Python package
│   ├── cad/              # CadQuery execution & validation
│   ├── data/             # scraping & dataset pipeline
│   ├── eval/             # metrics & benchmarks
│   ├── inference/        # MLX inference wrappers
│   └── training/         # SFT & GRPO
├── scripts/              # one-off utilities
├── tests/
├── data/                 # datasets (gitignored)
└── artifacts/            # models & checkpoints (gitignored)
```

## Common commands

```bash
make sync
uv run cad-llm ensure-dirs

# Training data
uv run cad-llm data download
uv run cad-llm data prepare

# Benchmark (Text2CAD-Bench)
uv run cad-llm bench download
uv run cad-llm bench run
```

## Dependency groups

Core dependencies install with `uv sync`. Optional extras are wired into the `dev` group:


| Extra | Packages                     | Purpose                 |
| ----- | ---------------------------- | ----------------------- |
| `mlx` | mlx-lm, mlx-tune             | inference & fine-tuning |
| `cad` | cadquery, build123d, trimesh | geometry kernel         |
| `web` | fastapi, uvicorn             | demo API (Step 8)       |


Install a subset without dev tools:

```bash
uv sync --no-default-groups --extra mlx --extra cad
```

## Configuration

Copy `.env.example` to `.env` to override defaults (model ID, server host/port, paths).