"""HallucinationMetric — detects factual hallucinations against provided context."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_VERDICT_PROMPT = """Given the provided context, does the following statement contain information that CONTRADICTS or is NOT PRESENT in the context (i.e., hallucinated)?
Return ONLY a JSON object: {{"verdict": "yes"}} (hallucinated) or {{"verdict": "no"}} (not hallucinated), plus a brief reason.
Format: {{"verdict": "yes"/"no", "reason": "..."}}

Context:
{context}

Statement: {statement}

Verdict:"""

_SPLIT_PROMPT = """Split the following text into individual factual statements.
Return ONLY a JSON array of strings.

Text:
{text}

Statements (JSON array):"""


class HallucinationMetric(BaseMetric):
    """
    Detects hallucinations in the actual_output compared to the provided context.

    score = 1 - (hallucinated_statements / total_statements)
    A LOW score means HIGH hallucination.

    Requires: actual_output, context
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not test_case.context:
            raise ValueError("HallucinationMetric requires context.")

        provider = self._model or self._get_default_provider()
        context_str = "\n".join(test_case.context)

        stmts_raw = self._llm_judge(
            _SPLIT_PROMPT.format(text=test_case.actual_output),
            provider=provider,
        )
        statements = self._parse_verdict_list(stmts_raw) or [test_case.actual_output]

        steps = []
        hallucinated = []
        for stmt in statements:
            raw = self._llm_judge(
                _VERDICT_PROMPT.format(context=context_str, statement=stmt),
                provider=provider,
            )
            verdict = self._parse_json_field(raw, "verdict") or "no"
            reason = self._parse_json_field(raw, "reason") or ""
            is_hallucination = verdict.lower() == "yes"
            hallucinated.append(is_hallucination)
            status = "HALLUCINATED" if is_hallucination else "grounded"
            steps.append(f"[{status}] '{stmt[:80]}' — {reason[:80]}")

        hallucination_rate = sum(hallucinated) / len(hallucinated) if hallucinated else 0.0
        score = 1.0 - hallucination_rate
        passed = self._pass_fail(score)
        reason_str = (
            f"{sum(hallucinated)}/{len(hallucinated)} statements appear hallucinated. "
            f"Hallucination rate: {hallucination_rate:.1%}"
        )

        self._score = score
        self._reason = reason_str
        self._result = MetricResult(
            score=score, passed=passed, reason=reason_str,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
