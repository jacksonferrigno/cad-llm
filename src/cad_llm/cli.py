import typer
from rich.console import Console

from cad_llm.config import settings

app = typer.Typer(
    name="cad-llm",
    help="Local CAD LLM tooling — text to CadQuery on Apple Silicon.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def info() -> None:
    """Print project paths and default model configuration."""
    console.print("[bold]cad-llm[/bold]")
    console.print(f"  project root:  {settings.project_root}")
    console.print(f"  data dir:      {settings.resolve(settings.data_dir)}")
    console.print(f"  artifacts dir: {settings.resolve(settings.artifacts_dir)}")
    console.print(f"  model id:      {settings.mlx_model_id}")


@app.command()
def ensure_dirs() -> None:
    """Create local data and artifact directories."""
    for path in (
        settings.data_dir,
        settings.artifacts_dir,
        settings.models_dir,
        settings.checkpoints_dir,
    ):
        resolved = settings.resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]ok[/green]  {resolved}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
