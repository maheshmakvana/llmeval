"""ArgumentCorrectnessMetric — checks tool call arguments are correct."""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase, ToolCall

_ARG_PROMPT = """Given the tool name and the task context, evaluate whether the tool was called with correct arguments.

Task: {task}
Tool: {tool_name}
Arguments provided: {args}
Expected arguments (if known): {expected_args}

Are the arguments correct, appropriate, and complete for the given task?
Score from 0 to 10 (10 = perfect arguments, 0 = completely wrong).
Return as JSON: {{"score": <0-10>, "issues": ["..."], "reason": "..."}}

Evaluation:"""


class ArgumentCorrectnessMetric(BaseMetric):
    """
    Checks whether tools were called with correct arguments.

    If expected tool arguments are not provided, uses LLM to judge correctness
    based on task context.

    Requires: input, tools_called
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model=None,
        expected_tool_args: Optional[Dict[str, Dict[str, Any]]] = None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model
        self.expected_tool_args = expected_tool_args or {}

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        if not test_case.tools_called:
            raise ValueError("ArgumentCorrectnessMetric requires tools_called.")

        provider = self._model or self._get_default_provider()
        scores = []
        steps = []

        for tool in test_case.tools_called:
            expected = self.expected_tool_args.get(tool.name, {})
            raw = self._llm_judge(
                _ARG_PROMPT.format(
                    task=test_case.input,
                    tool_name=tool.name,
                    args=tool.input_parameters,
                    expected_args=expected or "Not specified",
                ),
                provider=provider,
            )
            s_raw = self._parse_json_field(raw, "score") or 0
            issues = self._parse_json_field(raw, "issues") or []
            t_score = float(s_raw) / 10.0
            scores.append(t_score)
            for issue in issues:
                steps.append(f"{tool.name}: {issue}")
            if not issues:
                steps.append(f"{tool.name}: arguments are correct ({t_score:.2f})")

        score = sum(scores) / len(scores) if scores else 0.0
        passed = self._pass_fail(score)
        reason = f"Average argument correctness across {len(scores)} tool calls: {score:.3f}"

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result


from typing import Optional
