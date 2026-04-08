"""ContextualPrecisionMetric — measures ranking quality of retrieved chunks."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_VERDICT_PROMPT = """Given the question and expected output, is the following retrieved context chunk relevant/useful for generating the expected output?
Return ONLY a JSON object: {{"verdict": "yes"}} or {{"verdict": "no"}}

Question: {question}
Expected output: {expected_output}
Context chunk: {chunk}

Verdict:"""


class ContextualPrecisionMetric(BaseMetric):
    """
    Measures the precision of the retriever — do the retrieved chunks actually help answer the question?
    Rewards relevant chunks appearing earlier (weighted precision).

    Requires: input, expected_output, retrieval_context
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not test_case.retrieval_context:
            raise ValueError("ContextualPrecisionMetric requires retrieval_context.")
        if not test_case.expected_output:
            raise ValueError("ContextualPrecisionMetric requires expected_output.")

        provider = self._model or self._get_default_provider()
        steps = []
        verdicts = []

        for i, chunk in enumerate(test_case.retrieval_context):
            raw = self._llm_judge(
                _VERDICT_PROMPT.format(
                    question=test_case.input,
                    expected_output=test_case.expected_output,
                    chunk=chunk,
                ),
                provider=provider,
            )
            verdict = self._parse_json_field(raw, "verdict") or "no"
            ok = verdict.lower() == "yes"
            verdicts.append(ok)
            steps.append(f"Chunk {i+1}: {'useful' if ok else 'not useful'}")

        # Weighted precision: penalise relevant chunks that rank after irrelevant ones
        score = self._weighted_precision(verdicts)
        passed = self._pass_fail(score)
        reason = (
            f"Weighted contextual precision score: {score:.3f}. "
            f"{sum(verdicts)}/{len(verdicts)} chunks are relevant."
        )

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result

    @staticmethod
    def _weighted_precision(verdicts):
        """Average precision (AP) — standard IR metric."""
        if not any(verdicts):
            return 0.0
        running, total, n_relevant = 0.0, 0, 0
        for i, v in enumerate(verdicts):
            if v:
                n_relevant += 1
                precision_at_k = n_relevant / (i + 1)
                running += precision_at_k
                total += 1
        return running / total if total else 0.0
