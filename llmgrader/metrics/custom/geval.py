"""GEvalMetric — LLM-as-a-judge with chain-of-thought for ANY custom criteria."""

from __future__ import annotations

from typing import List, Optional, Union

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase, LLMTestCaseParams

_COT_PROMPT = """You are an evaluation assistant. Your task is to evaluate an LLM output based on specific criteria.

Evaluation criteria: {criteria}

Evaluation steps to follow:
{steps}

Test case details:
{test_case_details}

Think step by step, then provide a final score from 0 to 10 (10 = perfect, 0 = completely fails criteria).

Output your response as:
Reasoning: <your chain-of-thought reasoning>
Score: <integer from 0 to 10>
"""

_GENERATE_STEPS_PROMPT = """Generate a numbered list of evaluation steps for the following criteria.
Each step should be a concrete, measurable check.

Criteria: {criteria}

Evaluation steps (numbered list):"""


class GEvalMetric(BaseMetric):
    """
    GEval: LLM-as-a-judge with chain-of-thought reasoning.

    Supports ANY custom evaluation criteria. The judge LLM generates
    evaluation steps from the criteria (if not provided), then scores
    the test case using chain-of-thought reasoning.

    Example:
        metric = GEvalMetric(
            name="Correctness",
            criteria="The output should be factually correct and complete.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        )
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: List[LLMTestCaseParams],
        evaluation_steps: Optional[List[str]] = None,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._name = name
        self.criteria = criteria
        self.evaluation_params = evaluation_params
        self._eval_steps = evaluation_steps
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    def _get_eval_steps(self, provider) -> List[str]:
        if self._eval_steps:
            return self._eval_steps
        raw = self._llm_judge(
            _GENERATE_STEPS_PROMPT.format(criteria=self.criteria),
            provider=provider,
        )
        import re
        lines = re.findall(r"\d+\.\s+(.+)", raw)
        return lines or [self.criteria]

    def _build_test_case_details(self, test_case: LLMTestCase) -> str:
        parts = []
        param_map = {
            LLMTestCaseParams.INPUT: ("Input", test_case.input),
            LLMTestCaseParams.ACTUAL_OUTPUT: ("Actual Output", test_case.actual_output),
            LLMTestCaseParams.EXPECTED_OUTPUT: ("Expected Output", test_case.expected_output),
            LLMTestCaseParams.CONTEXT: ("Context", "\n".join(test_case.context or [])),
            LLMTestCaseParams.RETRIEVAL_CONTEXT: ("Retrieval Context", "\n".join(test_case.retrieval_context or [])),
        }
        for param in self.evaluation_params:
            label, value = param_map.get(param, (param.value, None))
            if value:
                parts.append(f"{label}:\n{value}")
        return "\n\n".join(parts)

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()
        steps = self._get_eval_steps(provider)
        details = self._build_test_case_details(test_case)

        prompt = _COT_PROMPT.format(
            criteria=self.criteria,
            steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
            test_case_details=details,
        )
        response = self._llm_judge(prompt, provider=provider)

        # Parse score
        score = self._parse_score_from_response(response, scale=10)

        # Parse reasoning
        import re
        reason_match = re.search(r"(?i)reasoning\s*[:\-]\s*(.+?)(?=score\s*[:\-]|\Z)", response, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else response[:200]

        passed = self._pass_fail(score)

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
            verbose_logs=response if self.verbose_mode else None,
        )
        return self._result
