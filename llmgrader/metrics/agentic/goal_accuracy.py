"""GoalAccuracyMetric — evaluates if the agent achieved its intended goal accurately."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_GOAL_ACCURACY_PROMPT = """You are evaluating whether an AI agent achieved its intended goal accurately.

Intended Goal (input): {goal}
Agent's Final Output: {output}
Expected Outcome (if provided): {expected}
Tools Used: {tools}

Evaluate the ACCURACY of what was achieved versus what was intended. This is distinct from mere task
completion — focus on whether the result is correct, precise, and aligned with the goal's intent.

Consider:
1. Does the output directly address the intended goal?
2. Is the result accurate — are facts, calculations, or decisions correct?
3. How closely does the achieved outcome match the intended outcome?
4. Are there any gaps, errors, or misalignments between intention and result?
5. If an expected outcome is provided, how well does the actual output match it?

Score from 0 to 10:
- 10: Output perfectly achieves the intended goal with full accuracy
- 7-9: Output largely achieves the goal with minor inaccuracies
- 4-6: Output partially achieves the goal or has moderate inaccuracies
- 1-3: Output misses key aspects of the goal or has significant errors
- 0: Output fails to achieve the intended goal or is completely wrong

Return as JSON: {{"score": <0-10>, "goal_achieved": true/false, "accuracy_gaps": ["..."], "reason": "..."}}

Evaluation:"""


class GoalAccuracyMetric(BaseMetric):
    """
    Evaluates whether an LLM agent achieved its intended goal accurately.

    Unlike TaskCompletionMetric (which checks IF the task was done), this metric
    checks HOW ACCURATELY the goal was achieved — correctness, precision, and
    alignment with the intended outcome.

    Requires: input (the goal), actual_output
    Optional: expected_output (for ground-truth comparison), tools_called
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        tools_str = ", ".join(
            [t.name for t in (test_case.tools_called or [])]
        ) or "None"

        expected_str = test_case.expected_output or "Not provided"

        raw = self._llm_judge(
            _GOAL_ACCURACY_PROMPT.format(
                goal=test_case.input,
                output=test_case.actual_output,
                expected=expected_str,
                tools=tools_str,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        gaps = self._parse_json_field(raw, "accuracy_gaps") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]
        goal_achieved = self._parse_json_field(raw, "goal_achieved")

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        steps = []
        if goal_achieved is not None:
            steps.append(f"Goal achieved: {goal_achieved}")
        steps += [f"Accuracy gap: {g}" for g in gaps] or ["No accuracy gaps identified."]

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
            verbose_logs=raw if self.verbose_mode else None,
        )
        return self._result
