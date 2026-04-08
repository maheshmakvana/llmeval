"""FaithfulnessMetric — detects hallucinations against retrieval_context."""

from __future__ import annotations

from typing import Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_CLAIMS_PROMPT = """Break the following answer into independent factual claims.
Return ONLY a JSON array of strings (each claim). Do not include explanations.

Answer:
{answer}

Claims (JSON array):"""

_VERDICT_PROMPT = """Given the following context and a factual claim, determine whether the claim is directly supported by the context.
Return ONLY a JSON object: {{"verdict": "yes"}} or {{"verdict": "no"}}

Context:
{context}

Claim: {claim}

Verdict:"""


class FaithfulnessMetric(BaseMetric):
    """
    Measures whether all factual claims in actual_output are grounded in retrieval_context.

    Algorithm:
    1. Extract atomic claims from actual_output.
    2. For each claim, ask judge: is it supported by retrieval_context?
    3. score = supported_claims / total_claims

    Requires: actual_output, retrieval_context
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        include_reason: bool = True,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model
        self.include_reason = include_reason

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not isinstance(test_case, LLMTestCase):
            raise TypeError("FaithfulnessMetric requires an LLMTestCase.")
        if not test_case.retrieval_context:
            raise ValueError("FaithfulnessMetric requires retrieval_context.")

        provider = self._model or self._get_default_provider()
        context_str = "\n".join(test_case.retrieval_context)

        # Step 1: extract claims
        claims_raw = self._llm_judge(
            _CLAIMS_PROMPT.format(answer=test_case.actual_output),
            provider=provider,
        )
        claims = self._parse_verdict_list(claims_raw)
        if not claims:
            claims = [test_case.actual_output]

        # Step 2: verdict per claim
        steps = []
        supported = []
        for claim in claims:
            raw = self._llm_judge(
                _VERDICT_PROMPT.format(context=context_str, claim=claim),
                provider=provider,
            )
            verdict = self._parse_json_field(raw, "verdict") or "no"
            ok = verdict.lower() == "yes"
            supported.append(ok)
            steps.append(f"Claim: '{claim[:80]}' → {'supported' if ok else 'NOT supported'}")

        score = sum(supported) / len(supported) if supported else 0.0
        passed = self._pass_fail(score)
        reason = f"{sum(supported)}/{len(supported)} claims are supported by the retrieval context."

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
