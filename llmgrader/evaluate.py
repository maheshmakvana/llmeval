"""Bulk evaluation engine — evaluate() and assert_test()."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .metrics.base import BaseMetric, MetricResult
from .test_case import ConversationalTestCase, LLMTestCase


@dataclass
class TestResult:
    """Result for a single test case across all metrics."""

    test_case: Union[LLMTestCase, ConversationalTestCase]
    metrics_data: List[MetricResult]
    success: bool
    total_latency_ms: float = 0.0

    @property
    def failed_metrics(self) -> List[MetricResult]:
        return [m for m in self.metrics_data if not m.passed]

    @property
    def passed_metrics(self) -> List[MetricResult]:
        return [m for m in self.metrics_data if m.passed]


@dataclass
class EvaluationResult:
    """Aggregated results from an evaluate() call."""

    test_results: List[TestResult]
    overall_score: float
    pass_rate: float
    metric_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_evaluation_time_ms: float = 0.0

    def print_summary(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\n[bold]Evaluation Summary[/bold]")
        console.print(f"Tests: {len(self.test_results)} | Pass rate: {self.pass_rate:.1%} | Score: {self.overall_score:.3f}\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Avg Score", justify="right")
        table.add_column("Pass Rate", justify="right")

        for metric_name, summary in self.metric_summaries.items():
            table.add_row(
                metric_name,
                f"{summary['avg_score']:.3f}",
                f"{summary['pass_rate']:.1%}",
            )
        console.print(table)


def evaluate(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    metrics: List[BaseMetric],
    max_concurrent: int = 4,
    verbose: bool = True,
    ignore_errors: bool = False,
) -> EvaluationResult:
    """
    Run all metrics against all test cases and return aggregated results.

    Args:
        test_cases: List of LLMTestCase or ConversationalTestCase objects.
        metrics: List of metric instances to evaluate.
        max_concurrent: Number of concurrent evaluation threads.
        verbose: Print progress and results.
        ignore_errors: If True, catch metric errors and mark as failed instead of raising.

    Returns:
        EvaluationResult with per-test and aggregate data.
    """
    start_time = time.time()
    test_results: List[TestResult] = []

    def _evaluate_single(tc) -> TestResult:
        metric_results = []
        tc_start = time.time()
        for metric in metrics:
            try:
                result = metric.measure(tc)
            except Exception as e:
                if ignore_errors:
                    result = MetricResult(
                        score=0.0,
                        passed=False,
                        reason=f"Error: {e}",
                        metric_name=metric.name,
                        threshold=metric.threshold,
                        strict_mode=metric.strict_mode,
                    )
                else:
                    raise
            metric_results.append(result)

        tc_latency = (time.time() - tc_start) * 1000
        success = all(r.passed for r in metric_results)
        return TestResult(
            test_case=tc,
            metrics_data=metric_results,
            success=success,
            total_latency_ms=tc_latency,
        )

    if max_concurrent > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(_evaluate_single, tc): tc for tc in test_cases}
            for future in concurrent.futures.as_completed(futures):
                test_results.append(future.result())
    else:
        for tc in test_cases:
            test_results.append(_evaluate_single(tc))

    # Aggregate
    total_time = (time.time() - start_time) * 1000
    pass_count = sum(1 for r in test_results if r.success)
    pass_rate = pass_count / len(test_results) if test_results else 0.0

    # Per-metric summaries
    metric_names = [m.name for m in metrics]
    metric_summaries: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        scores = [
            mr.score
            for tr in test_results
            for mr in tr.metrics_data
            if mr.metric_name == name
        ]
        passes = [
            mr.passed
            for tr in test_results
            for mr in tr.metrics_data
            if mr.metric_name == name
        ]
        metric_summaries[name] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "pass_rate": sum(passes) / len(passes) if passes else 0.0,
        }

    overall_score = (
        sum(v["avg_score"] for v in metric_summaries.values()) / len(metric_summaries)
        if metric_summaries else 0.0
    )

    result = EvaluationResult(
        test_results=test_results,
        overall_score=overall_score,
        pass_rate=pass_rate,
        metric_summaries=metric_summaries,
        total_evaluation_time_ms=total_time,
    )

    if verbose:
        _print_verbose_results(test_results, result)

    return result


def assert_test(
    test_case: Union[LLMTestCase, ConversationalTestCase],
    metrics: List[BaseMetric],
) -> None:
    """
    Assert-style single test evaluation — raises AssertionError on failure.
    Designed for use inside pytest test functions.

    Example:
        def test_my_llm():
            tc = LLMTestCase(input="What is 2+2?", actual_output="4")
            assert_test(tc, metrics=[GEvalMetric(...)])
    """
    failed = []
    for metric in metrics:
        result = metric.measure(test_case)
        if not result.passed:
            failed.append(f"{result.metric_name}: score={result.score:.3f} < threshold={result.threshold} — {result.reason}")

    if failed:
        raise AssertionError(
            "LLM evaluation failed:\n" + "\n".join(f"  - {f}" for f in failed)
        )


def _print_verbose_results(test_results: List[TestResult], summary: EvaluationResult) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="llmgrader Evaluation Results", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Input", max_width=40)
        table.add_column("Metrics", max_width=60)
        table.add_column("Pass", justify="center")

        for i, tr in enumerate(test_results):
            tc = tr.test_case
            input_str = getattr(tc, "input", "N/A")[:40]
            metrics_str = " | ".join(
                f"{r.metric_name}={r.score:.2f}({'✓' if r.passed else '✗'})"
                for r in tr.metrics_data
            )
            pass_str = "[green]✓[/green]" if tr.success else "[red]✗[/red]"
            table.add_row(str(i + 1), input_str, metrics_str, pass_str)

        console.print(table)
        console.print(f"\nPass rate: [bold]{summary.pass_rate:.1%}[/bold] | Overall score: [bold]{summary.overall_score:.3f}[/bold]\n")
    except ImportError:
        for i, tr in enumerate(test_results):
            status = "PASS" if tr.success else "FAIL"
            print(f"[{i+1}] {status}")
