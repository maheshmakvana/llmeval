"""TaskCompletionMetric — evaluates whether an agent accomplished its goal."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_TASK_PROMPT = """You are evaluating an AI agent's task completion.

Task/Goal (input): {task}
Agent's final response: {response}
Tools used: {tools}

Evaluate whether the agent FULLY completed the task. Consider:
1. Did the agent address all parts of the task?
2. Is the output accurate and actionable?
3. Did the agent use appropriate tools?
4. Is the response complete (no unresolved steps)?

Score from 0 to 10 (10 = task fully completed, 0 = task not attempted or failed completely).
Return as JSON: {{"score": <0-10>, "completion_aspects": ["..."], "reason": "..."}}

Evaluation:"""


class TaskCompletionMetric(BaseMetric):
    """
    Evaluates whether an LLM agent successfully completed the given task.

    Requires: input, actual_output
    Optional: tools_called
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()
        tools_str = ", ".join(
            [t.name for t in (test_case.tools_called or [])]
        ) or "None"

        raw = self._llm_judge(
            _TASK_PROMPT.format(
                task=test_case.input,
                response=test_case.actual_output,
                tools=tools_str,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        aspects = self._parse_json_field(raw, "completion_aspects") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)
        steps = [f"Aspect: {a}" for a in aspects] or ["Task evaluation complete."]

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
