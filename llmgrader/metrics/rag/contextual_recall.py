"""ContextualRecallMetric — measures how well retrieval_context covers the expected_output."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_CLAIMS_PROMPT = """Break the following expected answer into independent factual claims.
Return ONLY a JSON array of strings. Do not include explanations.

Answer:
{answer}

Claims (JSON array):"""

_VERDICT_PROMPT = """Is the following factual claim supported by the provided context?
Return ONLY a JSON object: {{"verdict": "yes"}} or {{"verdict": "no"}}

Context:
{context}

Claim: {claim}

Verdict:"""


class ContextualRecallMetric(BaseMetric):
    """
    Measures whether expected_output claims are covered by retrieval_context.

    Algorithm:
    1. Extract claims from expected_output.
    2. For each claim, check if retrieval_context supports it.
    3. score = covered_claims / total_claims

    Requires: expected_output, retrieval_context
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not test_case.retrieval_context:
            raise ValueError("ContextualRecallMetric requires retrieval_context.")
        if not test_case.expected_output:
            raise ValueError("ContextualRecallMetric requires expected_output.")

        provider = self._model or self._get_default_provider()
        context_str = "\n".join(test_case.retrieval_context)

        claims_raw = self._llm_judge(
            _CLAIMS_PROMPT.format(answer=test_case.expected_output),
            provider=provider,
        )
        claims = self._parse_verdict_list(claims_raw) or [test_case.expected_output]

        steps = []
        covered = []
        for claim in claims:
            raw = self._llm_judge(
                _VERDICT_PROMPT.format(context=context_str, claim=claim),
                provider=provider,
            )
            verdict = self._parse_json_field(raw, "verdict") or "no"
            ok = verdict.lower() == "yes"
            covered.append(ok)
            steps.append(f"Claim '{claim[:80]}': {'covered' if ok else 'NOT covered'}")

        score = sum(covered) / len(covered) if covered else 0.0
        passed = self._pass_fail(score)
        reason = f"{sum(covered)}/{len(covered)} expected output claims are covered by retrieval context."

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
