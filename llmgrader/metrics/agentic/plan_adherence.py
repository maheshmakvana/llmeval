"""PlanAdherenceMetric — checks if the agent followed a predefined plan."""

from __future__ import annotations

from typing import List, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_PLAN_ADHERENCE_PROMPT = """You are evaluating whether an AI agent adhered to a predefined plan.

Original Goal: {goal}
Predefined Plan Steps:
{plan_steps}

Agent's Actual Steps / Output:
{output}

Tools Used (in order): {tools}

Evaluate how closely the agent followed the predefined plan. Consider:
1. Were all plan steps attempted in the correct order?
2. Were any plan steps skipped or omitted?
3. Were any steps performed that deviated significantly from the plan?
4. Did deviations (if any) improve or harm the outcome?
5. Was the overall execution consistent with the plan's intent?

Score from 0 to 10:
- 10: Agent followed the plan exactly as specified
- 7-9: Agent followed the plan with minor, justified deviations
- 4-6: Agent partially followed the plan with some notable deviations
- 1-3: Agent largely ignored the plan and took a different approach
- 0: Agent completely disregarded the predefined plan

Return as JSON: {{
  "score": <0-10>,
  "steps_followed": ["..."],
  "steps_skipped": ["..."],
  "unplanned_steps": ["..."],
  "reason": "..."
}}

Evaluation:"""


class PlanAdherenceMetric(BaseMetric):
    """
    Checks whether an LLM agent followed a predefined plan.

    The plan is a list of steps the agent was expected to execute. This metric
    evaluates adherence to both the content and order of those steps.

    Requires: input (goal), actual_output
    Required: plan (list of expected steps)
    Optional: tools_called
    """

    def __init__(
        self,
        plan: Optional[List[str]] = None,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model
        self.plan = plan or []

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        # Use plan from metric or from test_case additional_metadata
        plan = self.plan or test_case.additional_metadata.get("plan", [])
        if not plan:
            score, reason = 1.0, "No predefined plan provided; adherence cannot be evaluated."
            self._score = score
            self._reason = reason
            self._result = MetricResult(
                score=score, passed=True, reason=reason,
                metric_name=self.name, threshold=self.threshold,
                strict_mode=self.strict_mode,
            )
            return self._result

        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        tools_str = ", ".join(
            [t.name for t in (test_case.tools_called or [])]
        ) or "None"

        raw = self._llm_judge(
            _PLAN_ADHERENCE_PROMPT.format(
                goal=test_case.input,
                plan_steps=plan_str,
                output=test_case.actual_output,
                tools=tools_str,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        steps_followed = self._parse_json_field(raw, "steps_followed") or []
        steps_skipped = self._parse_json_field(raw, "steps_skipped") or []
        unplanned = self._parse_json_field(raw, "unplanned_steps") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if steps_followed:
            eval_steps.append(f"Followed: {', '.join(steps_followed)}")
        if steps_skipped:
            eval_steps.append(f"Skipped: {', '.join(steps_skipped)}")
        if unplanned:
            eval_steps.append(f"Unplanned steps taken: {', '.join(unplanned)}")
        if not eval_steps:
            eval_steps = ["Plan adherence evaluated."]

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score,
            passed=passed,
            reason=reason,
            metric_name=self.name,
            threshold=self.threshold,
            strict_mode=self.strict_mode,
            evaluation_steps=eval_steps,
            verbose_logs=raw if self.verbose_mode else None,
        )
        return self._result
