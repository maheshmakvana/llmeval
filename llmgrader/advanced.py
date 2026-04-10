"""
llmgrader.advanced — Advanced evaluation utilities.

New in 1.2.0:
- RegressionTracker: Track metric scores across runs and detect regressions
- ScoreTrend: Rolling-window trend analysis for metric time-series
- EvaluationReport: Export results to JSON / CSV / Markdown
- CustomBenchmarkBuilder: Build domain-specific benchmarks from your own data
- AsyncEvaluator: Fully async evaluate() for high-throughput pipelines
- EvaluationFilter: Filter test cases by tag, metadata, or score threshold
- MetricWeightedScorer: Weighted aggregate scorer across multiple metrics
- evaluate_async: Drop-in async replacement for evaluate()
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from .evaluate import EvaluationResult, TestResult, evaluate
from .metrics.base import BaseMetric, MetricResult
from .test_case import LLMTestCase, ConversationalTestCase


# ---------------------------------------------------------------------------
# RegressionTracker
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """A single evaluation run stored by RegressionTracker."""
    run_id: str
    timestamp: float
    metric_scores: Dict[str, float]  # metric_name -> avg_score
    pass_rates: Dict[str, float]     # metric_name -> pass_rate
    overall_score: float
    n_tests: int
    tags: Dict[str, str] = field(default_factory=dict)


class RegressionTracker:
    """
    Track metric scores across evaluation runs and detect regressions.

    A regression is triggered when a metric's average score drops by more
    than *threshold* compared to the previous run.

    Example
    -------
    >>> tracker = RegressionTracker(threshold=0.05)
    >>> tracker.record(result, run_id="v1.0")
    >>> regressions = tracker.check_regressions()
    >>> tracker.trend("answer_relevancy", last_n=5)
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self._threshold = threshold
        self._runs: List[RunRecord] = []

    def record(
        self,
        result: EvaluationResult,
        run_id: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> RunRecord:
        """Record a completed EvaluationResult."""
        record = RunRecord(
            run_id=run_id,
            timestamp=time.time(),
            metric_scores={k: v["avg_score"] for k, v in result.metric_summaries.items()},
            pass_rates={k: v["pass_rate"] for k, v in result.metric_summaries.items()},
            overall_score=result.overall_score,
            n_tests=len(result.test_results),
            tags=tags or {},
        )
        self._runs.append(record)
        return record

    def check_regressions(self) -> List[Dict[str, Any]]:
        """
        Compare the last two runs and return a list of regression dicts.

        Each dict has keys: metric, before, after, delta, run_id_before, run_id_after
        """
        if len(self._runs) < 2:
            return []

        prev, curr = self._runs[-2], self._runs[-1]
        regressions = []
        for metric, curr_score in curr.metric_scores.items():
            prev_score = prev.metric_scores.get(metric)
            if prev_score is None:
                continue
            delta = curr_score - prev_score
            if delta < -self._threshold:
                regressions.append({
                    "metric": metric,
                    "before": round(prev_score, 4),
                    "after": round(curr_score, 4),
                    "delta": round(delta, 4),
                    "run_id_before": prev.run_id,
                    "run_id_after": curr.run_id,
                })
        return regressions

    def trend(self, metric: str, last_n: int = 10) -> List[Dict[str, Any]]:
        """Return the score history for a given metric across the last N runs."""
        history = []
        for run in self._runs[-last_n:]:
            if metric in run.metric_scores:
                history.append({
                    "run_id": run.run_id,
                    "timestamp": run.timestamp,
                    "score": run.metric_scores[metric],
                    "pass_rate": run.pass_rates.get(metric),
                })
        return history

    def best_run(self, metric: Optional[str] = None) -> Optional[RunRecord]:
        """Return the run with the highest score for *metric* (or overall_score)."""
        if not self._runs:
            return None
        if metric:
            return max(self._runs, key=lambda r: r.metric_scores.get(metric, -1))
        return max(self._runs, key=lambda r: r.overall_score)

    def summary(self) -> Dict[str, Any]:
        if not self._runs:
            return {"runs": 0}
        latest = self._runs[-1]
        return {
            "runs": len(self._runs),
            "latest_run_id": latest.run_id,
            "latest_overall_score": latest.overall_score,
            "regressions": self.check_regressions(),
        }

    def to_json(self) -> str:
        data = [
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "overall_score": r.overall_score,
                "n_tests": r.n_tests,
                "metric_scores": r.metric_scores,
                "pass_rates": r.pass_rates,
                "tags": r.tags,
            }
            for r in self._runs
        ]
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# ScoreTrend
# ---------------------------------------------------------------------------

class ScoreTrend:
    """
    Rolling-window trend analysis on metric score time series.

    Example
    -------
    >>> trend = ScoreTrend(window=5)
    >>> trend.add(0.72)
    >>> trend.add(0.75)
    >>> trend.add(0.68)
    >>> trend.direction  # 'declining'
    >>> trend.moving_average  # 0.717
    """

    def __init__(self, window: int = 5) -> None:
        self._window = window
        self._scores: List[float] = []

    def add(self, score: float) -> None:
        self._scores.append(score)

    @property
    def moving_average(self) -> Optional[float]:
        recent = self._scores[-self._window:]
        if not recent:
            return None
        return round(sum(recent) / len(recent), 4)

    @property
    def direction(self) -> str:
        """'improving', 'declining', 'stable', or 'insufficient_data'"""
        if len(self._scores) < 2:
            return "insufficient_data"
        recent = self._scores[-self._window:]
        if len(recent) < 2:
            return "insufficient_data"
        delta = recent[-1] - recent[0]
        if delta > 0.02:
            return "improving"
        if delta < -0.02:
            return "declining"
        return "stable"

    @property
    def volatility(self) -> Optional[float]:
        """Standard deviation of the last N scores."""
        recent = self._scores[-self._window:]
        if len(recent) < 2:
            return None
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return round(math.sqrt(variance), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self._scores),
            "moving_average": self.moving_average,
            "direction": self.direction,
            "volatility": self.volatility,
            "latest": self._scores[-1] if self._scores else None,
        }


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

class EvaluationReport:
    """
    Export EvaluationResult to JSON, CSV, or Markdown.

    Example
    -------
    >>> report = EvaluationReport(result)
    >>> report.to_markdown()
    >>> report.to_json("report.json")
    >>> report.to_csv("report.csv")
    """

    def __init__(self, result: EvaluationResult, run_id: str = "evaluation") -> None:
        self._result = result
        self._run_id = run_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self._run_id,
            "overall_score": self._result.overall_score,
            "pass_rate": self._result.pass_rate,
            "total_tests": len(self._result.test_results),
            "total_time_ms": self._result.total_evaluation_time_ms,
            "metric_summaries": self._result.metric_summaries,
            "test_results": [
                {
                    "input": getattr(tr.test_case, "input", "N/A"),
                    "success": tr.success,
                    "latency_ms": tr.total_latency_ms,
                    "metrics": [
                        {
                            "name": mr.metric_name,
                            "score": mr.score,
                            "passed": mr.passed,
                            "reason": mr.reason,
                        }
                        for mr in tr.metrics_data
                    ],
                }
                for tr in self._result.test_results
            ],
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        data = self.to_dict()
        text = json.dumps(data, indent=indent)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_csv(self, filepath: Optional[str] = None) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        # header
        metric_names = list(self._result.metric_summaries.keys())
        header = ["test_id", "input", "success", "latency_ms"] + metric_names
        writer.writerow(header)
        for i, tr in enumerate(self._result.test_results):
            row = [
                i + 1,
                getattr(tr.test_case, "input", "N/A")[:80],
                tr.success,
                round(tr.total_latency_ms, 2),
            ]
            scores_by_name = {mr.metric_name: mr.score for mr in tr.metrics_data}
            for name in metric_names:
                row.append(round(scores_by_name.get(name, 0.0), 4))
            writer.writerow(row)
        text = buf.getvalue()
        if filepath:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        return text

    def to_markdown(self, filepath: Optional[str] = None) -> str:
        r = self._result
        lines = [
            f"# Evaluation Report — {self._run_id}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Score | {r.overall_score:.4f} |",
            f"| Pass Rate | {r.pass_rate:.1%} |",
            f"| Total Tests | {len(r.test_results)} |",
            f"| Evaluation Time | {r.total_evaluation_time_ms:.0f} ms |",
            "",
            "## Metric Summaries",
            "",
            "| Metric | Avg Score | Pass Rate |",
            "|--------|-----------|-----------|",
        ]
        for name, summary in r.metric_summaries.items():
            lines.append(
                f"| {name} | {summary['avg_score']:.4f} | {summary['pass_rate']:.1%} |"
            )
        lines += [
            "",
            "## Test Results",
            "",
            "| # | Input (truncated) | Pass | Latency ms |",
            "|---|-------------------|------|------------|",
        ]
        for i, tr in enumerate(r.test_results):
            inp = getattr(tr.test_case, "input", "N/A")[:50]
            lines.append(
                f"| {i+1} | {inp} | {'✓' if tr.success else '✗'} | {tr.total_latency_ms:.1f} |"
            )
        text = "\n".join(lines) + "\n"
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
        return text


# ---------------------------------------------------------------------------
# CustomBenchmarkBuilder
# ---------------------------------------------------------------------------

class CustomBenchmarkBuilder:
    """
    Build a domain-specific benchmark from your own golden dataset.

    Example
    -------
    >>> builder = CustomBenchmarkBuilder("customer-support-bench")
    >>> builder.add(input="How do I reset my password?",
    ...             expected_output="Go to settings > security > reset password.",
    ...             context=["password policy doc"], tags=["auth"])
    >>> benchmark = builder.build()
    >>> result = benchmark.run(metrics=[AnswerRelevancyMetric()])
    """

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._cases: List[Dict[str, Any]] = []

    def add(
        self,
        input: str,
        expected_output: Optional[str] = None,
        actual_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CustomBenchmarkBuilder":
        self._cases.append({
            "input": input,
            "expected_output": expected_output,
            "actual_output": actual_output or "",
            "context": context,
            "retrieval_context": retrieval_context,
            "tags": tags or [],
            "metadata": metadata or {},
        })
        return self

    def add_from_dicts(self, records: List[Dict[str, Any]]) -> "CustomBenchmarkBuilder":
        for r in records:
            self.add(**r)
        return self

    def add_from_json(self, filepath: str) -> "CustomBenchmarkBuilder":
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)
        return self.add_from_dicts(records)

    def add_from_csv(self, filepath: str) -> "CustomBenchmarkBuilder":
        with open(filepath, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.add(
                    input=row.get("input", ""),
                    expected_output=row.get("expected_output"),
                    actual_output=row.get("actual_output", ""),
                    context=row.get("context", "").split("|") if row.get("context") else None,
                )
        return self

    def build(self) -> "BuiltBenchmark":
        test_cases = [
            LLMTestCase(
                input=c["input"],
                actual_output=c["actual_output"],
                expected_output=c["expected_output"],
                context=c["context"],
                retrieval_context=c["retrieval_context"],
            )
            for c in self._cases
        ]
        return BuiltBenchmark(
            name=self._name,
            description=self._description,
            test_cases=test_cases,
            raw_cases=self._cases,
        )

    def filter_by_tag(self, tag: str) -> "CustomBenchmarkBuilder":
        """Return a new builder with only cases carrying the given tag."""
        b = CustomBenchmarkBuilder(self._name, self._description)
        b._cases = [c for c in self._cases if tag in c.get("tags", [])]
        return b

    def __len__(self) -> int:
        return len(self._cases)


@dataclass
class BuiltBenchmark:
    name: str
    description: str
    test_cases: List[LLMTestCase]
    raw_cases: List[Dict[str, Any]]

    def run(
        self,
        metrics: List[BaseMetric],
        max_concurrent: int = 4,
        verbose: bool = True,
    ) -> EvaluationResult:
        """Run all built test cases against the provided metrics."""
        result = evaluate(
            test_cases=self.test_cases,
            metrics=metrics,
            max_concurrent=max_concurrent,
            verbose=verbose,
        )
        return result

    def to_json(self, filepath: Optional[str] = None) -> str:
        text = json.dumps({"name": self.name, "cases": self.raw_cases}, indent=2)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
        return text


# ---------------------------------------------------------------------------
# EvaluationFilter
# ---------------------------------------------------------------------------

class EvaluationFilter:
    """
    Filter test cases from an EvaluationResult by success, score, or custom predicate.

    Example
    -------
    >>> f = EvaluationFilter(result)
    >>> failures = f.failures()
    >>> low_score = f.below_score("answer_relevancy", threshold=0.5)
    """

    def __init__(self, result: EvaluationResult) -> None:
        self._result = result

    def failures(self) -> List[TestResult]:
        return [tr for tr in self._result.test_results if not tr.success]

    def successes(self) -> List[TestResult]:
        return [tr for tr in self._result.test_results if tr.success]

    def below_score(self, metric_name: str, threshold: float) -> List[TestResult]:
        out = []
        for tr in self._result.test_results:
            for mr in tr.metrics_data:
                if mr.metric_name == metric_name and mr.score < threshold:
                    out.append(tr)
                    break
        return out

    def above_score(self, metric_name: str, threshold: float) -> List[TestResult]:
        out = []
        for tr in self._result.test_results:
            for mr in tr.metrics_data:
                if mr.metric_name == metric_name and mr.score >= threshold:
                    out.append(tr)
                    break
        return out

    def where(self, predicate: Callable[[TestResult], bool]) -> List[TestResult]:
        return [tr for tr in self._result.test_results if predicate(tr)]

    def slowest(self, n: int = 10) -> List[TestResult]:
        return sorted(self._result.test_results, key=lambda t: t.total_latency_ms, reverse=True)[:n]


# ---------------------------------------------------------------------------
# MetricWeightedScorer
# ---------------------------------------------------------------------------

class MetricWeightedScorer:
    """
    Compute a single weighted aggregate score across multiple metrics.

    Example
    -------
    >>> scorer = MetricWeightedScorer()
    >>> scorer.add("answer_relevancy", weight=0.5)
    >>> scorer.add("faithfulness", weight=0.3)
    >>> scorer.add("toxicity", weight=0.2, inverse=True)
    >>> score = scorer.compute(result)
    """

    def __init__(self) -> None:
        self._weights: List[Tuple[str, float, bool]] = []  # (name, weight, inverse)

    def add(self, metric_name: str, weight: float = 1.0, inverse: bool = False) -> "MetricWeightedScorer":
        """
        Parameters
        ----------
        metric_name : str
        weight : float
        inverse : bool
            If True, (1 - score) is used. Useful for safety metrics where 0 is good.
        """
        self._weights.append((metric_name, weight, inverse))
        return self

    def compute(self, result: EvaluationResult) -> float:
        total_weight = sum(w for _, w, _ in self._weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for metric_name, weight, inverse in self._weights:
            score = result.metric_summaries.get(metric_name, {}).get("avg_score", 0.0)
            if inverse:
                score = 1.0 - score
            weighted_sum += weight * score

        return round(weighted_sum / total_weight, 4)


# ---------------------------------------------------------------------------
# evaluate_async — async drop-in for evaluate()
# ---------------------------------------------------------------------------

async def evaluate_async(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    metrics: List[BaseMetric],
    max_concurrent: int = 8,
    ignore_errors: bool = False,
) -> EvaluationResult:
    """
    Fully async evaluation runner. Uses asyncio to evaluate test cases concurrently.

    Example
    -------
    >>> result = await evaluate_async(test_cases, metrics, max_concurrent=16)
    """
    import time as _time

    start = _time.time()
    sem = asyncio.Semaphore(max_concurrent)

    async def _evaluate_single(tc) -> TestResult:
        async with sem:
            metric_results = []
            tc_start = _time.time()
            for metric in metrics:
                try:
                    # Run synchronous metric.measure in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, metric.measure, tc)
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
            tc_latency = (_time.time() - tc_start) * 1000
            success = all(r.passed for r in metric_results)
            return TestResult(
                test_case=tc,
                metrics_data=metric_results,
                success=success,
                total_latency_ms=tc_latency,
            )

    tasks = [_evaluate_single(tc) for tc in test_cases]
    test_results = await asyncio.gather(*tasks)
    test_results = list(test_results)

    total_time = (_time.time() - start) * 1000
    pass_count = sum(1 for r in test_results if r.success)
    pass_rate = pass_count / len(test_results) if test_results else 0.0

    metric_names = [m.name for m in metrics]
    metric_summaries: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        scores = [mr.score for tr in test_results for mr in tr.metrics_data if mr.metric_name == name]
        passes = [mr.passed for tr in test_results for mr in tr.metrics_data if mr.metric_name == name]
        metric_summaries[name] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "pass_rate": sum(passes) / len(passes) if passes else 0.0,
        }

    overall_score = (
        sum(v["avg_score"] for v in metric_summaries.values()) / len(metric_summaries)
        if metric_summaries else 0.0
    )

    return EvaluationResult(
        test_results=test_results,
        overall_score=overall_score,
        pass_rate=pass_rate,
        metric_summaries=metric_summaries,
        total_evaluation_time_ms=total_time,
    )
