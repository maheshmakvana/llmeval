"""PatternMatchMetric — regex-based pattern matching evaluation."""

from __future__ import annotations

import re
from typing import List, Optional, Union

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase


class PatternMatchMetric(BaseMetric):
    """
    Regex-based pattern matching metric.

    Checks whether actual_output matches one or more regex patterns.
    Supports both "all must match" and "any must match" modes.

    No LLM is used — this is a purely deterministic metric.

    Requires: actual_output

    Args:
        patterns: A single regex pattern string or a list of patterns.
        match_all: If True, ALL patterns must match. If False, ANY pattern
                   must match. Default True.
        flags: Regex flags (e.g., re.IGNORECASE). Default 0.
        search_mode: If True, uses re.search (pattern anywhere in text).
                     If False, uses re.fullmatch (entire text must match).
                     Default True.

    Example:
        # Check that output contains an email address
        metric = PatternMatchMetric(
            patterns=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
        )

        # Check that output contains both a date and a number
        metric = PatternMatchMetric(
            patterns=[r"\\d{4}-\\d{2}-\\d{2}", r"\\$\\d+"],
            match_all=True,
        )
    """

    def __init__(
        self,
        patterns: Union[str, List[str]],
        match_all: bool = True,
        flags: int = 0,
        search_mode: bool = True,
        threshold: float = 1.0,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self.patterns: List[str] = [patterns] if isinstance(patterns, str) else patterns
        self.match_all = match_all
        self.flags = flags
        self.search_mode = search_mode

    def _check_pattern(self, text: str, pattern: str) -> bool:
        try:
            if self.search_mode:
                return re.search(pattern, text, self.flags) is not None
            else:
                return re.fullmatch(pattern, text, self.flags) is not None
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        text = test_case.actual_output
        results = [(p, self._check_pattern(text, p)) for p in self.patterns]

        matched_patterns = [p for p, r in results if r]
        unmatched_patterns = [p for p, r in results if not r]

        if self.match_all:
            score = len(matched_patterns) / len(self.patterns) if self.patterns else 1.0
            success = len(unmatched_patterns) == 0
        else:
            # Any-match: score 1.0 if at least one matches
            score = 1.0 if matched_patterns else 0.0
            success = bool(matched_patterns)

        passed = self._pass_fail(score)
        mode_str = "ALL" if self.match_all else "ANY"
        search_str = "search" if self.search_mode else "fullmatch"

        eval_steps = [f"Mode: {mode_str} patterns must match ({search_str})"]
        for p, r in results:
            status = "MATCHED" if r else "NOT MATCHED"
            eval_steps.append(f"Pattern '{p}': {status}")

        if success:
            reason = (
                f"{'All' if self.match_all else 'At least one'} pattern(s) matched. "
                f"({len(matched_patterns)}/{len(self.patterns)} matched)"
            )
        else:
            reason = (
                f"Pattern match failed. "
                f"{len(matched_patterns)}/{len(self.patterns)} patterns matched. "
                f"Unmatched: {unmatched_patterns}"
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
            evaluation_steps=eval_steps,
        )
        return self._result
