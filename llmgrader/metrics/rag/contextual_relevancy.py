"""ContextualRelevancyMetric — measures how relevant each retrieved chunk is to the input."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_VERDICT_PROMPT = """Given the following question and a retrieved context chunk, determine whether the chunk is relevant to answering the question.
Return ONLY a JSON object: {{"verdict": "yes"}} or {{"verdict": "no"}}

Question: {question}
Context chunk: {chunk}

Verdict:"""


class ContextualRelevancyMetric(BaseMetric):
    """
    Measures whether each chunk in retrieval_context is relevant to the input question.

    Algorithm:
    1. For each chunk in retrieval_context, ask: relevant to input?
    2. score = relevant_chunks / total_chunks

    Requires: input, retrieval_context
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not test_case.retrieval_context:
            raise ValueError("ContextualRelevancyMetric requires retrieval_context.")

        provider = self._model or self._get_default_provider()
        steps = []
        verdicts = []

        for i, chunk in enumerate(test_case.retrieval_context):
            raw = self._llm_judge(
                _VERDICT_PROMPT.format(question=test_case.input, chunk=chunk),
                provider=provider,
            )
            verdict = self._parse_json_field(raw, "verdict") or "no"
            ok = verdict.lower() == "yes"
            verdicts.append(ok)
            steps.append(f"Chunk {i+1}: {'relevant' if ok else 'irrelevant'}")

        score = sum(verdicts) / len(verdicts) if verdicts else 0.0
        passed = self._pass_fail(score)
        reason = f"{sum(verdicts)}/{len(verdicts)} retrieved chunks are relevant to the input."

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
