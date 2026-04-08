"""ExactMatchMetric — deterministic exact string matching."""

from __future__ import annotations

from typing import Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase


class ExactMatchMetric(BaseMetric):
    """
    Deterministic exact string matching metric.

    Compares actual_output against expected_output using exact string equality.
    Supports optional case normalization and whitespace normalization.

    No LLM is used — this is a purely deterministic metric.

    Requires: actual_output, expected_output

    Args:
        ignore_case: If True, comparison is case-insensitive. Default False.
        ignore_whitespace: If True, normalizes whitespace before comparing. Default False.
        strip: If True, strips leading/trailing whitespace. Default True.
    """

    def __init__(
        self,
        ignore_case: bool = False,
        ignore_whitespace: bool = False,
        strip: bool = True,
        threshold: float = 1.0,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.strip = strip

    def _normalize(self, text: str) -> str:
        if self.strip:
            text = text.strip()
        if self.ignore_whitespace:
            import re
            text = re.sub(r"\s+", " ", text)
        if self.ignore_case:
            text = text.lower()
        return text

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if test_case.expected_output is None:
            raise ValueError("ExactMatchMetric requires expected_output to be set.")

        actual = self._normalize(test_case.actual_output)
        expected = self._normalize(test_case.expected_output)

        matched = actual == expected
        score = 1.0 if matched else 0.0
        passed = self._pass_fail(score)

        steps = [
            f"Actual (normalized): '{actual[:100]}{'...' if len(actual) > 100 else ''}'",
            f"Expected (normalized): '{expected[:100]}{'...' if len(expected) > 100 else ''}'",
            f"Match: {matched}",
        ]

        if matched:
            reason = "Actual output exactly matches the expected output."
        else:
            # Find first difference position
            for i, (a, e) in enumerate(zip(actual, expected)):
                if a != e:
                    reason = (
                        f"Outputs differ at position {i}: "
                        f"got '{actual[max(0,i-10):i+10]}', "
                        f"expected '{expected[max(0,i-10):i+10]}'"
                    )
                    break
            else:
                reason = (
                    f"Outputs differ in length: actual={len(actual)}, expected={len(expected)}"
                )

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score,
            passed=passed,
            reason=reason,
            metric_name=self.name,
            threshold=self.threshold,
            strict_mode=self.strict_mode,
            evaluation_steps=steps,
        )
        return self._result
