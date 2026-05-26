import json
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cad_llm.agent.display import AgentConsole, print_paths
from cad_llm.agent.runner import AgentRunResult, run_agent
from cad_llm.agent.session import AgentSession
from cad_llm.agent.steps import AgentStep
from cad_llm.config import settings
from cad_llm.data import bench as bench_data
from cad_llm.data import text2cadquery
from cad_llm.eval.bench import default_output_path, run_bench, save_report
from cad_llm.inference.generate import CadGenerator, resolve_model_id
from cad_llm.tools.docs.index import index_docs
from cad_llm.tools.docs.search import search_cadquery_docs
from cad_llm.tools.workspace import create_chat, create_project, list_projects, load_project
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout
from cad_llm.training.sft import DEFAULT_DATA, DEFAULT_OUTPUT, run_sft

app = typer.Typer(
    name="cad-llm",
    help="Local CAD LLM tooling — text to CadQuery on Apple Silicon.",
    no_args_is_help=True,
)
console = Console()

data_app = typer.Typer(help="Download and prepare training data.")
bench_app = typer.Typer(help="Text2CAD-Bench evaluation.")
train_app = typer.Typer(help="Model training.")
docs_app = typer.Typer(help="CadQuery documentation RAG (LangChain + pgvector).")
project_app = typer.Typer(help="CAD workspace projects and agent runs.")
app.add_typer(data_app, name="data")
app.add_typer(bench_app, name="bench")
app.add_typer(train_app, name="train")
app.add_typer(docs_app, name="docs")
app.add_typer(project_app, name="project")

DEFAULT_BENCH_PROMPTS = bench_data.default_prompts_path(
    settings.resolve(settings.text2cad_bench_dir)
)
projects_root = settings.resolve(settings.projects_dir)


def _agent_callbacks(
    console: Console,
    *,
    quiet: bool,
    no_stream: bool,
) -> tuple[Callable[[AgentStep], None] | None, Callable[..., str] | None]:
    agent_console = AgentConsole(console, stream=not no_stream and not quiet, show_reasoning=False)

    def on_step(step: AgentStep) -> None:
        if not quiet:
            agent_console.print_step(step)

    def wrapped_generate(model, tokenizer, prompt, max_tokens):  # noqa: ANN001
        from cad_llm.agent.generate import generate_completion

        return generate_completion(model, tokenizer, prompt, max_tokens)

    return (on_step if not quiet else None, wrapped_generate if not quiet else None)


def _run_project_turn(
    project: ProjectLayout,
    chat: ChatLayout,
    prompt: str,
    *,
    console: Console,
    session: AgentSession | None,
    model_path: str | None,
    max_steps: int,
    max_tokens: int,
    no_stream: bool,
    quiet: bool,
) -> AgentRunResult:
    on_step, generate_fn = _agent_callbacks(console, quiet=quiet, no_stream=no_stream)

    if session is not None:
        return session.run_turn(prompt, on_step=on_step, generate_fn=generate_fn)

    return run_agent(
        project,
        chat,
        prompt,
        model_id=model_path,
        max_steps=max_steps,
        max_tokens=max_tokens,
        on_step=on_step,
        generate_fn=generate_fn,
    )


@app.command()
def desktop() -> None:
    """Launch native desktop shell (terminal chat + CAD preview)."""
    try:
        from cad_llm.app.window import run
    except ImportError as exc:
        raise typer.BadParameter(
            "Desktop UI dependencies missing. Run: uv sync --extra app"
        ) from exc
    run()


@app.command()
def info() -> None:
    """Print project paths and default model configuration."""
    console.print("[bold]cad-llm[/bold]")
    console.print(f"  project root:  {settings.project_root}")
    console.print(f"  data dir:      {settings.resolve(settings.data_dir)}")
    console.print(f"  artifacts dir: {settings.resolve(settings.artifacts_dir)}")
    console.print(f"  workspace dir: {settings.resolve(settings.workspace_dir)}")
    console.print(f"  projects dir:  {settings.resolve(settings.projects_dir)}")
    console.print(f"  vertex model:  {settings.vertex_base_model}")
    console.print(f"  hf model:      {settings.hf_model_id}")
    console.print(f"  mlx model:     {settings.mlx_model_id}")


@app.command()
def ensure_dirs() -> None:
    """Create local data and artifact directories."""
    for path in (
        settings.data_dir,
        settings.artifacts_dir,
        settings.models_dir,
        settings.checkpoints_dir,
        settings.workspace_dir,
        settings.projects_dir,
        settings.text2cadquery_dir,
        settings.text2cad_bench_dir,
    ):
        resolved = settings.resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]ok[/green]  {resolved}")


@data_app.command("download")
def data_download() -> None:
    """Download Text-to-CadQuery (CadQuery.zip + captions CSV)."""
    base = settings.resolve(settings.text2cadquery_dir)
    console.print("[bold]Downloading Text-to-CadQuery[/bold]")
    zip_path, csv_path = text2cadquery.download(base)
    zip_mb = zip_path.stat().st_size // 1_000_000
    csv_mb = csv_path.stat().st_size // 1_000_000
    console.print(f"  [green]ok[/green]  {zip_path.name} ({zip_mb} MB)")
    console.print(f"  [green]ok[/green]  {csv_path.name} ({csv_mb} MB)")


@data_app.command("prepare")
def data_prepare(
    sft_size: int = typer.Option(10_000, "--sft-size", min=1),
    grpo_size: int = typer.Option(60_000, "--grpo-size", min=1),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Join prompts + code and write SFT/GRPO JSONL splits."""
    base = settings.resolve(settings.text2cadquery_dir)
    console.print("[bold]Preparing training splits[/bold]")
    summary = text2cadquery.prepare(base, sft_size=sft_size, grpo_size=grpo_size, seed=seed)
    console.print(f"  matched scripts: {summary.total_scripts}")
    console.print(
        f"  sft train: {summary.sft_scripts} scripts → {summary.sft_rows} rows → {summary.sft_path}"
    )
    console.print(
        f"  sft val:   {summary.sft_val_scripts} scripts → {summary.sft_val_rows} rows → {summary.sft_val_path}"
    )
    console.print(
        f"  grpo: {summary.grpo_scripts} scripts → {summary.grpo_rows} rows → {summary.grpo_path}"
    )


@bench_app.command("download")
def bench_download() -> None:
    """Download Text2CAD-Bench prompts."""
    base = settings.resolve(settings.text2cad_bench_dir)
    console.print("[bold]Downloading Text2CAD-Bench[/bold]")
    path = bench_data.download(base)
    prompts = bench_data.load_prompts(path)
    console.print(f"  [green]ok[/green]  {path.name} ({len(prompts)} prompts)")


@bench_app.command("run")
def bench_run(
    prompts_path: Path = typer.Option(
        DEFAULT_BENCH_PROMPTS,
        "--prompts",
        help="Text2CAD-Bench release.csv",
    ),
    model_path: str | None = typer.Option(
        None,
        "--model",
        help="MLX model path or Hugging Face repo id (default: settings.mlx_model_id)",
    ),
    output_path: Path | None = typer.Option(None, "--output"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    max_tokens: int = typer.Option(512, "--max-tokens", min=16),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run zero-shot eval on Text2CAD-Bench prompts."""
    prompts = bench_data.load_prompts(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    console.print(f"[bold]Text2CAD-Bench[/bold] — {len(prompts)} prompts")
    if model_path:
        resolved_model = resolve_model_id(model_path)
        console.print(f"  model:  {resolved_model}")
        generator = CadGenerator(model_id=resolved_model)
    else:
        generator = None
    report = run_bench(prompts, generator=generator, max_tokens=max_tokens, verbose=verbose)
    destination = output_path or default_output_path()
    save_report(report, destination)

    summary = report.summary
    table = Table(title="Text2CAD-Bench summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total prompts", str(summary.total))
    table.add_row("Compile rate", f"{summary.compile_rate:.1%}")
    table.add_row("Geometry rate", f"{summary.geometry_rate:.1%}")
    table.add_row("Watertight rate", f"{summary.watertight_rate:.1%}")
    console.print(table)
    console.print(f"\nReport saved to [cyan]{destination}[/cyan]")


@bench_app.command("summarize")
def bench_summarize(report_path: Path) -> None:
    """Print summary from a saved bench report."""
    data = json.loads(report_path.read_text())
    summary = data["summary"]
    table = Table(title=f"{data['benchmark']} — {data['model_id']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Run at", data["run_at"])
    table.add_row("Total prompts", str(summary["total"]))
    table.add_row("Compile rate", f"{summary['compile_rate']:.1%}")
    table.add_row("Geometry rate", f"{summary['geometry_rate']:.1%}")
    table.add_row("Watertight rate", f"{summary['watertight_rate']:.1%}")
    console.print(table)


@train_app.command("sft")
def train_sft(
    data_path: Path = typer.Option(DEFAULT_DATA, "--data"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--output"),
    max_samples: int | None = typer.Option(None, "--max-samples", min=1),
) -> None:
    """LoRA SFT warm-up on prepared JSONL."""
    console.print("[bold]SFT training[/bold]")
    console.print(f"  data:   {data_path}")
    console.print(f"  output: {output_dir}")
    if max_samples:
        console.print(f"  limit:  {max_samples} samples")

    adapter_path = run_sft(data_path, output_dir, max_samples=max_samples)
    console.print(f"\n[green]Done.[/green] Adapters saved to [cyan]{adapter_path}[/cyan]")


@docs_app.command("index")
def docs_index(
    html_path: Path | None = typer.Option(None, "--html"),
    reset: bool = typer.Option(True, "--reset/--no-reset"),
) -> None:
    """Chunk CadQuery HTML with LangChain and index into pgvector."""
    resolved_html = html_path or settings.resolve(settings.cadquery_docs_dir) / "index.html"
    summary = index_docs(
        resolved_html,
        settings.docs_db_url,
        collection_name=settings.docs_collection_name,
        embedding_model=settings.docs_embedding_model,
        chunks_cache=settings.resolve(settings.docs_chunks_cache),
        reset=reset,
    )
    console.print("[bold]CadQuery docs indexed[/bold]")
    console.print(f"  html:       {summary.html_path}")
    console.print(f"  chunks:     {summary.chunks}")
    console.print(f"  collection: {summary.collection_name}")
    console.print(f"  cache:      {summary.chunks_cache}")


@docs_app.command("search")
def docs_search(
    query: str,
    limit: int = typer.Option(5, "--limit", min=1),
) -> None:
    """Hybrid search (PGVector + BM25) over indexed CadQuery docs."""
    result = search_cadquery_docs(
        query,
        db_url=settings.docs_db_url,
        collection_name=settings.docs_collection_name,
        embedding_model=settings.docs_embedding_model,
        chunks_cache=settings.resolve(settings.docs_chunks_cache),
        limit=limit,
    )
    console.print(result)


@project_app.command("create")
def project_create(
    name: str,
    project_id: str | None = typer.Option(None, "--id", help="Optional stable project id."),
) -> None:
    """Create a new CAD workspace project."""
    projects_root.mkdir(parents=True, exist_ok=True)
    project = create_project(projects_root, name=name, project_id=project_id)
    console.print(f"[green]Created[/green] project [bold]{project.project_id}[/bold] — {name}")
    print_paths(
        console,
        project_root=str(project.root),
        src_dir=str(project.src_dir),
        outputs_dir=str(project.outputs_dir),
        transcript=str(project.chats_dir / "<chat_id>" / "transcript.jsonl"),
    )


@project_app.command("list")
def project_list() -> None:
    """List workspace projects."""
    projects = list_projects(projects_root)
    if not projects:
        console.print(f"No projects under [cyan]{projects_root}[/cyan]")
        return

    table = Table(title="Workspace projects")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Root")
    for project in projects:
        meta = json.loads((project.meta_dir / "project.json").read_text(encoding="utf-8"))
        table.add_row(project.project_id, meta.get("name", ""), str(project.root))
    console.print(table)


@project_app.command("show")
def project_show(project_id: str) -> None:
    """Show paths and layout for one project."""
    project = load_project(projects_root, project_id)
    meta = json.loads((project.meta_dir / "project.json").read_text(encoding="utf-8"))
    console.print(f"[bold]{meta.get('name', project_id)}[/bold] ({project.project_id})")
    print_paths(
        console,
        project_root=str(project.root),
        src_dir=str(project.src_dir),
        outputs_dir=str(project.outputs_dir),
        transcript=str(project.chats_dir / "<chat_id>" / "transcript.jsonl"),
    )
    chats = sorted(project.chats_dir.glob("*/meta.json")) if project.chats_dir.is_dir() else []
    if chats:
        console.print(f"\nChats: {len(chats)}")
        for meta_path in chats:
            chat_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            console.print(f"  • {chat_meta.get('id')} — {chat_meta.get('title', '')}")


@project_app.command("run")
def project_run(
    project_id: str,
    prompt: str,
    chat_id: str | None = typer.Option(None, "--chat-id", help="Reuse an existing chat id."),
    chat_title: str = typer.Option(
        "CLI run",
        "--chat-title",
        help="Title when creating a new chat.",
    ),
    model_path: str | None = typer.Option(
        None,
        "--model",
        help="MLX model path or Hugging Face repo id (default: settings.mlx_model_id).",
    ),
    max_steps: int = typer.Option(15, "--max-steps", min=1),
    max_tokens: int = typer.Option(2048, "--max-tokens", min=64),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable token streaming."),
    quiet: bool = typer.Option(False, "--quiet", help="No live output (still writes transcript)."),
) -> None:
    """Run the CAD agent on a single prompt."""
    project = load_project(projects_root, project_id)
    if chat_id:
        from cad_llm.tools.workspace.project import chat_layout

        chat = chat_layout(project, chat_id)
        if not chat.root.is_dir():
            raise typer.BadParameter(f"Unknown chat id for project: {chat_id}")
    else:
        chat = create_chat(project, title=chat_title)

    preview = prompt if len(prompt) <= 60 else prompt[:60] + "…"
    console.print(f"[bold]agent[/bold]  {project_id}  ·  {preview}")
    print_paths(
        console,
        project_root=str(project.root),
        src_dir=str(project.src_dir),
        outputs_dir=str(project.outputs_dir),
        transcript=str(chat.transcript_path),
    )
    console.print()

    result = _run_project_turn(
        project,
        chat,
        prompt,
        console=console,
        session=None,
        model_path=model_path,
        max_steps=max_steps,
        max_tokens=max_tokens,
        no_stream=no_stream,
        quiet=quiet,
    )

    console.print()
    console.print(f"[green]Done.[/green] Transcript: [cyan]{result.transcript_path}[/cyan]")
    if result.run_summary_path is not None:
        console.print(f"Run summary:     [cyan]{result.run_summary_path}[/cyan]")
    if result.final_response:
        console.print(f"\n{result.final_response}")


@project_app.command("chat")
def project_chat(
    project_id: str,
    chat_id: str | None = typer.Option(None, "--chat-id", help="Resume an existing chat."),
    chat_title: str = typer.Option("Interactive session", "--chat-title"),
    model_path: str | None = typer.Option(None, "--model"),
    max_steps: int = typer.Option(15, "--max-steps", min=1),
    max_tokens: int = typer.Option(2048, "--max-tokens", min=64),
    no_stream: bool = typer.Option(False, "--no-stream"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Interactive chat session in a project (type exit to leave)."""
    project = load_project(projects_root, project_id)
    if chat_id:
        from cad_llm.tools.workspace.project import chat_layout

        chat = chat_layout(project, chat_id)
        if not chat.root.is_dir():
            raise typer.BadParameter(f"Unknown chat id for project: {chat_id}")
    else:
        chat = create_chat(project, title=chat_title)

    console.print(f"[bold]chat[/bold]  {project_id}  ·  {chat.chat_id}")
    print_paths(
        console,
        project_root=str(project.root),
        src_dir=str(project.src_dir),
        outputs_dir=str(project.outputs_dir),
        transcript=str(chat.transcript_path),
    )
    console.print("[dim]Enter prompts. Type exit or quit to leave.[/dim]\n")

    if not quiet:
        console.print("[dim]Loading model…[/dim]")
    session = AgentSession.create(
        project,
        chat,
        model_id=model_path,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )
    if not quiet:
        console.print("[dim]Ready.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold]you[/bold] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            break

        console.print()
        _run_project_turn(
            project,
            chat,
            user_input,
            console=console,
            session=session,
            model_path=model_path,
            max_steps=max_steps,
            max_tokens=max_tokens,
            no_stream=no_stream,
            quiet=quiet,
        )
        console.print()

    console.print(f"[green]Session ended.[/green] Transcript: [cyan]{chat.transcript_path}[/cyan]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
