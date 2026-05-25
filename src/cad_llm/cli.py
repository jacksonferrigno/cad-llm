import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cad_llm.config import settings
from cad_llm.data import bench as bench_data
from cad_llm.data import text2cadquery
from cad_llm.eval.bench import default_output_path, run_bench, save_report

app = typer.Typer(
    name="cad-llm",
    help="Local CAD LLM tooling — text to CadQuery on Apple Silicon.",
    no_args_is_help=True,
)
console = Console()

data_app = typer.Typer(help="Download and prepare training data.")
bench_app = typer.Typer(help="Text2CAD-Bench evaluation.")
app.add_typer(data_app, name="data")
app.add_typer(bench_app, name="bench")

DEFAULT_BENCH_PROMPTS = bench_data.default_prompts_path(
    settings.resolve(settings.text2cad_bench_dir)
)


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
        f"  sft:  {summary.sft_scripts} scripts → {summary.sft_rows} rows → {summary.sft_path}"
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
    report = run_bench(prompts, max_tokens=max_tokens, verbose=verbose)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
