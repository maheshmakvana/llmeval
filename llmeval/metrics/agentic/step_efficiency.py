"""StepEfficiencyMetric — evaluates whether the agent solved the task efficiently."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_EFFICIENCY_PROMPT = """Evaluate the efficiency of an AI agent's solution process.

Task: {task}
Tools called (in order): {tools}
Final response: {response}

Consider:
1. Were any redundant or unnecessary tool calls made?
2. Could the task have been solved with fewer steps?
3. Was there any circular or wasteful reasoning?
4. Is the tool call sequence logical and direct?

Score from 0 to 10 (10 = maximally efficient, 0 = extremely wasteful).
Return as JSON: {{"score": <0-10>, "inefficiencies": ["..."], "reason": "..."}}

Evaluation:"""


class StepEfficiencyMetric(BaseMetric):
    """
    Measures how efficiently the agent solved the task (minimal unnecessary steps).

    Requires: input, actual_output
    Optional: tools_called (richer evaluation with tool history)
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        tools_str = "\n".join(
            f"{i+1}. {t.name}({t.input_parameters})"
            for i, t in enumerate(test_case.tools_called or [])
        ) or "None"

        raw = self._llm_judge(
            _EFFICIENCY_PROMPT.format(
                task=test_case.input,
                tools=tools_str,
                response=test_case.actual_output,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        inefficiencies = self._parse_json_field(raw, "inefficiencies") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)
        steps = [f"Inefficiency: {i}" for i in inefficiencies] or ["Solution appears efficient."]

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
