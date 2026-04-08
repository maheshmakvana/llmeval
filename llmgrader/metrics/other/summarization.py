"""SummarizationMetric — evaluates quality of LLM-generated summaries."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_SUMMARY_PROMPT = """Evaluate the quality of the following summary of the provided source text.

Criteria to evaluate:
1. Coverage: Does the summary capture all key points?
2. Conciseness: Is it appropriately brief without being vague?
3. Accuracy: Are all statements in the summary true to the source?
4. No hallucination: Does the summary avoid adding information not in the source?

Score from 0 to 10 (10 = perfect summary, 0 = completely wrong/incomplete).
Return as JSON: {{"score": <0-10>, "missing_points": ["..."], "hallucinations": ["..."], "reason": "..."}}

Source text:
{source}

Summary to evaluate:
{summary}

Evaluation:"""


class SummarizationMetric(BaseMetric):
    """
    Evaluates the quality of a summary.

    Uses input as the source text and actual_output as the summary.

    Requires: input (source), actual_output (summary)
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        raw = self._llm_judge(
            _SUMMARY_PROMPT.format(
                source=test_case.input,
                summary=test_case.actual_output,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        missing = self._parse_json_field(raw, "missing_points") or []
        hallucinations = self._parse_json_field(raw, "hallucinations") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)
        steps = []
        for m in missing:
            steps.append(f"Missing: {m}")
        for h in hallucinations:
            steps.append(f"Hallucination: {h}")
        if not steps:
            steps.append("Summary quality is good.")

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
