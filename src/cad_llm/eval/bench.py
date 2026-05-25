"""Run model eval against Text2CAD-Bench prompts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from cad_llm.cad.execute import ExecutionResult, execute_cad_code
from cad_llm.config import settings
from cad_llm.data.bench import BenchPrompt
from cad_llm.inference.generate import CadGenerator, GenerationResult


@dataclass
class BenchResult:
    id: str
    prompt: str
    style: str
    raw_response: str
    code: str
    execution: ExecutionResult


@dataclass
class BenchSummary:
    total: int
    compile_rate: float
    geometry_rate: float
    watertight_rate: float


@dataclass
class BenchReport:
    benchmark: str
    model_id: str
    run_at: str
    summary: BenchSummary
    results: list[BenchResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "model_id": self.model_id,
            "run_at": self.run_at,
            "summary": asdict(self.summary),
            "results": [
                {
                    "id": r.id,
                    "prompt": r.prompt,
                    "style": r.style,
                    "raw_response": r.raw_response,
                    "code": r.code,
                    "execution": asdict(r.execution),
                }
                for r in self.results
            ],
        }


def summarize(results: list[BenchResult]) -> BenchSummary:
    total = len(results)
    if total == 0:
        return BenchSummary(0, 0.0, 0.0, 0.0)

    compiled = sum(1 for r in results if r.execution.success)
    with_geometry = sum(1 for r in results if r.execution.has_geometry)
    watertight = sum(1 for r in results if r.execution.is_watertight)

    return BenchSummary(
        total=total,
        compile_rate=compiled / total,
        geometry_rate=with_geometry / total,
        watertight_rate=watertight / total,
    )


def run_bench(
    prompts: list[BenchPrompt],
    *,
    generator: CadGenerator | None = None,
    max_tokens: int = 512,
    verbose: bool = False,
) -> BenchReport:
    gen = generator or CadGenerator()
    gen.load()

    results: list[BenchResult] = []
    for item in prompts:
        generation: GenerationResult = gen.generate(
            item.prompt,
            max_tokens=max_tokens,
            verbose=verbose,
        )
        execution = execute_cad_code(generation.code)
        results.append(
            BenchResult(
                id=item.id,
                prompt=item.prompt,
                style=item.style,
                raw_response=generation.raw_response,
                code=generation.code,
                execution=execution,
            )
        )

    return BenchReport(
        benchmark="Text2CAD-Bench",
        model_id=gen.model_id,
        run_at=datetime.now(tz=UTC).isoformat(),
        summary=summarize(results),
        results=results,
    )


def default_output_path() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return settings.resolve(settings.artifacts_dir) / "bench" / f"run_{stamp}.json"


def save_report(report: BenchReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
