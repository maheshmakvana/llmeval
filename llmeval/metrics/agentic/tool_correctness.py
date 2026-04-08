"""ToolCorrectnessMetric — evaluates whether the agent called the right tools."""

from __future__ import annotations

from typing import Set

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase


class ToolCorrectnessMetric(BaseMetric):
    """
    Measures whether the agent called the expected tools.

    Uses exact set comparison (Jaccard similarity) between
    tools_called and expected_tools.

    Requires: tools_called, expected_tools
    """

    def __init__(self, threshold: float = 0.5, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not test_case.expected_tools:
            raise ValueError("ToolCorrectnessMetric requires expected_tools.")

        called: Set[str] = {t.name for t in (test_case.tools_called or [])}
        expected: Set[str] = set(test_case.expected_tools)

        correct = called & expected
        missing = expected - called
        extra = called - expected

        if not expected:
            score = 1.0
        else:
            score = len(correct) / len(expected)

        passed = self._pass_fail(score)
        steps = []
        if correct:
            steps.append(f"Correctly called: {', '.join(sorted(correct))}")
        if missing:
            steps.append(f"Missing tools: {', '.join(sorted(missing))}")
        if extra:
            steps.append(f"Unexpected tools: {', '.join(sorted(extra))}")

        reason = (
            f"{len(correct)}/{len(expected)} expected tools were called. "
            + (f"Missing: {sorted(missing)}. " if missing else "")
            + (f"Extra: {sorted(extra)}." if extra else "")
        )

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
