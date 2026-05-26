# CAD LLM

**Vibe-code CAD on your Mac.** Describe a part in plain English, get runnable CadQuery code, a live 3D preview, and an exported STEP file — all locally, no cloud API.

## Demo

Prompt: *generate a 80×80×40mm block with 20mm radius hemispherical pocket on top face*

![Demo](media/demo.gif)

## What it does

CAD LLM is a local desktop agent for parametric CAD. You chat like you would in a coding assistant, but the output is real geometry:

- **Natural language in** — flanges, brackets, pockets, holes, exports
- **CadQuery code out** — written to `src/main.py` in a project workspace
- **Sandbox + self-correction** — runs the script, reads errors, fixes and retries
- **3D preview** — see the model in the app as soon as it exports
- **STEP download** — files land in `outputs/` for your CAD workflow

Brainstorming prompts get options and questions first. Build prompts go straight to code.

Everything runs on-device with **[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** via MLX on Apple Silicon. No API keys, no token billing, no sending your designs to a server.

## Quick start

**Requirements:** Apple Silicon Mac, macOS 13+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/jacksonferrigno/cad-llm.git
cd cad-llm
uv sync --extra app
uv run cad-llm desktop
```

Or from the CLI:

```bash
uv run cad-llm project create "my-part"
uv run cad-llm project run my-part "Build a 50mm cube with a 20mm center hole"
```

## How it works

1. You describe the part
2. The agent loads CAD skills + CadQuery doc context
3. It writes Python, runs it in a sandbox, and fixes failures
4. On success it exports STEP and shows the result in the preview pane

See [docs/AGENT.md](docs/AGENT.md) for agent architecture and tools.

## Docs

| Doc | What's in it |
| --- | --- |
| [docs/AGENT.md](docs/AGENT.md) | Agent loop, tools, desktop app, CLI |
| [overview.md](overview.md) | Training roadmap, benchmarks, fine-tuning |

## Dev setup

```bash
uv sync
uv run cad-llm ensure-dirs
make verify
```

Copy `.env.example` to `.env` to override the model (`MLX_MODEL_ID`, default `Qwen/Qwen3.5-4B`), docs DB, and workspace directories.
