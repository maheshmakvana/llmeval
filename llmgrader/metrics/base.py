"""Base metric class and MetricResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..test_case import ConversationalTestCase, LLMTestCase


@dataclass
class MetricResult:
    """Result from a metric measurement."""

    score: float
    passed: bool
    reason: str
    metric_name: str
    threshold: float
    strict_mode: bool = False
    evaluation_steps: List[str] = field(default_factory=list)
    evaluation_cost: Optional[float] = None
    verbose_logs: Optional[str] = None

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"MetricResult({self.metric_name}: {self.score:.3f} [{status}] reason='{self.reason[:80]}...')"


class BaseMetric(ABC):
    """
    Abstract base for all llmgrader metrics.

    Subclass and implement `measure(test_case)` and `is_successful()`.

    Example:
        class MyMetric(BaseMetric):
            def measure(self, test_case):
                ...
                return MetricResult(score=0.9, passed=True, reason="...", ...)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self._score: Optional[float] = None
        self._reason: Optional[str] = None
        self._result: Optional[MetricResult] = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def score(self) -> Optional[float]:
        return self._score

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @abstractmethod
    def measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> MetricResult:
        """Run the metric and return a MetricResult."""

    def is_successful(self) -> bool:
        if self._result is None:
            raise RuntimeError("Call measure() before is_successful().")
        return self._result.passed

    def _pass_fail(self, score: float) -> bool:
        if self.strict_mode:
            return score == 1.0
        return score >= self.threshold

    def __repr__(self) -> str:
        return f"{self.name}(threshold={self.threshold})"

    # ------------------------------------------------------------------
    # Shared LLM-judge helpers used by multiple metrics
    # ------------------------------------------------------------------

    def _get_default_provider(self):
        """Return a default OpenAI provider if none is set on the metric."""
        from ..providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model="gpt-4o", temperature=0.0)

    def _llm_judge(self, prompt: str, provider=None) -> str:
        """Call the judge LLM and return raw text response."""
        p = provider or self._get_default_provider()
        return p.generate(prompt)

    def _parse_score_from_response(self, response: str, scale: int = 10) -> float:
        """Extract a numeric score from an LLM response like 'Score: 7/10'."""
        import re
        patterns = [
            rf"(?i)score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*{scale}",
            rf"(?i)(\d+(?:\.\d+)?)\s*/\s*{scale}",
            r"(?i)score\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                raw = float(match.group(1))
                return min(raw / scale, 1.0) if raw > 1.0 else raw
        return 0.0

    def _parse_verdict_list(self, response: str) -> List[str]:
        """Extract a JSON list of verdicts from LLM response."""
        import json, re
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    def _parse_json_field(self, response: str, field: str) -> Any:
        """Extract a named field from a JSON block in LLM response."""
        import json, re
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                return obj.get(field)
            except json.JSONDecodeError:
                pass
        return None
