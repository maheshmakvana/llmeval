"""AnswerRelevancyMetric — measures whether the LLM output is relevant to the input."""

from __future__ import annotations

from typing import Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_EXTRACT_QUESTIONS_PROMPT = """Given the following answer, generate a list of questions that the answer is responding to.
Return ONLY a JSON array of strings (the questions). Do not include any explanation.

Answer:
{answer}

Questions (JSON array):"""

_VERDICT_PROMPT = """Given the original question and a candidate question extracted from an answer, determine whether the candidate question is relevant to the original question.
Return ONLY a JSON object: {{"verdict": "yes"}} or {{"verdict": "no"}}

Original question: {original}
Candidate question: {candidate}

Verdict:"""


class AnswerRelevancyMetric(BaseMetric):
    """
    Measures whether the LLM's actual_output is relevant to the input question.

    Algorithm (QAG):
    1. Extract implied questions from actual_output using an LLM.
    2. For each extracted question, ask an LLM judge: is it relevant to input?
    3. score = relevant_count / total_extracted_questions

    Requires: input, actual_output
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
            raise TypeError("AnswerRelevancyMetric requires an LLMTestCase.")

        provider = self._model or self._get_default_provider()

        # Step 1: extract questions implied by the answer
        questions_raw = self._llm_judge(
            _EXTRACT_QUESTIONS_PROMPT.format(answer=test_case.actual_output),
            provider=provider,
        )
        questions = self._parse_verdict_list(questions_raw)
        if not questions:
            questions = [test_case.input]  # fallback

        # Step 2: verdict per question
        verdicts = []
        steps = []
        for q in questions:
            raw = self._llm_judge(
                _VERDICT_PROMPT.format(original=test_case.input, candidate=q),
                provider=provider,
            )
            verdict = self._parse_json_field(raw, "verdict") or "no"
            verdicts.append(verdict.lower() == "yes")
            steps.append(f"Q: '{q}' → {'relevant' if verdict.lower() == 'yes' else 'irrelevant'}")

        relevant_count = sum(verdicts)
        score = relevant_count / len(verdicts) if verdicts else 0.0
        passed = self._pass_fail(score)

        reason = (
            f"{relevant_count}/{len(verdicts)} extracted questions were relevant to the input."
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
