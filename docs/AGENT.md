# CAD Agent

A local agent that turns plain-English CAD requests into runnable CadQuery projects. Everything runs on your Mac. No cloud API calls. No per-token billing.

## Model

The agent uses a 4-billion-parameter Qwen3.5 model for CAD generation. Inference runs locally on Apple Silicon. The model loads once per session and stays in memory for follow-up prompts.

Default model path: `Qwen/Qwen3.5-4B`.

## What it does

You describe a part in natural language. The agent researches CadQuery APIs, writes Python source, executes it in a sandbox, fixes failures, exports STEP or STL to `outputs/`, and shows the result in the desktop preview.

For vague or brainstorming prompts, the agent replies with options and questions only. It does not write code until you ask for a build.

## Architecture

Single agent with a tool-calling loop:

```
User prompt
    ↓
Bootstrap (skills + doc snippets)
    ↓
Agent loop: write → sandbox → fix → review → export
```

The agent loads `cad-generation` and `brainstorming` skills plus CadQuery doc snippets on the first turn. For build requests it writes `src/main.py`, runs the sandbox automatically after edits, fixes failures, and verifies the result before finishing. For vague prompts it replies in plain text without tools.

## Self-correction loop

The agent runs up to 15 steps per turn:

```
generate → tool call → observe result → generate again
```

Gates enforce progress:

- Cannot finish until sandbox returns `exit_code=0` after the last src edit.
- Syntax errors on write trigger an immediate rewrite nudge.
- No-op patches trigger a nudge to make a material change.

When sandbox fails, the error goes back into the conversation. The model edits and retries until the script runs or steps are exhausted.

## Tools

| Tool | Purpose |
|------|---------|
| `search_cadquery_docs` | Hybrid vector + keyword search over indexed CadQuery HTML docs |
| `write_file` | Create or overwrite project files |
| `search_replace` | Patch existing files |
| `read_file` | Read source for review or debugging |
| `grep` | Search project files |
| `run_python_sandbox` | Execute `src/main.py` with CadQuery in an isolated project cwd |
| `load_skill` | Load markdown skills for brainstorming and CAD conventions |

Doc search uses a local pgvector database and a cached chunk index. No external retrieval API.

## Skills

Markdown instruction files loaded into context:

- `brainstorming` — discuss first, build second
- `cad-generation` — layout, export rules, review checklist
- `cad-debug` — how to recover from sandbox failures

## Project workspace

Each project lives under `workspace/projects/{id}/`:

```
project/
  src/main.py          # entry point the sandbox runs
  src/parts/           # optional part modules
  outputs/             # exported STEP, STL
  chats/{chat_id}/
    transcript.jsonl   # full agent log
    run_*.json         # per-turn summary
```

Projects are gitignored. The agent never writes outside the project root.

## Desktop app

Native CustomTkinter shell:

- Left: project list
- Center: terminal-style transcript
- Right: 3D mesh preview from newest export

Launch:

```bash
uv sync --extra app
uv run cad-llm desktop
```

The model preloads in the background when the app opens.

## CLI

```bash
uv run cad-llm project create "bracket" --id my-bracket
uv run cad-llm project chat my-bracket
uv run cad-llm project run my-bracket "Build a 50mm cube with a center hole"
```

Interactive chat keeps the model loaded across turns. Skills and doc context cache after the first message in a session.

## End-to-end flow

```
User prompt
    ↓
Bootstrap skills + doc snippets
    ↓
Agent: write_file → auto sandbox → fix loop → export
    ↓
Desktop preview loads outputs/*.step
    ↓
Agent summary to user
```

## Dependencies

| Component | Local stack |
|-----------|-------------|
| Inference | MLX + mlx-lm |
| CAD kernel | CadQuery, build123d |
| Mesh preview | trimesh, cascadio, matplotlib |
| Doc index | PostgreSQL + pgvector, fastembed, rank-bm25 |
| UI | CustomTkinter |

All optional groups install via `uv sync`. Doc search requires `docker compose up` for the local Postgres container and `cad-llm docs index` to build the index once.

## Configuration

Environment variables and defaults live in `src/cad_llm/config.py`. Override model path, docs database URL, and project directories through `.env`.
