from cad_llm.cad.execute import ExecutionResult
from cad_llm.eval.bench import BenchResult, summarize


def test_summarize_bench_results() -> None:
    results = [
        BenchResult(
            id="L1_1",
            prompt="p",
            style="geo",
            raw_response="r",
            code="c",
            execution=ExecutionResult(True, None, True, True, "cadquery"),
        ),
        BenchResult(
            id="L1_2",
            prompt="p",
            style="geo",
            raw_response="r",
            code="c",
            execution=ExecutionResult(False, "err", False, False, None),
        ),
    ]
    summary = summarize(results)
    assert summary.total == 2
    assert summary.compile_rate == 0.5
