"""PromptAlignmentMetric — checks if output follows instructions/constraints in the prompt."""

from __future__ import annotations

from typing import List, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_PROMPT_ALIGNMENT_PROMPT = """You are evaluating whether an LLM's output aligns with the instructions and constraints in its prompt.

System/User Prompt (instructions):
{prompt}

LLM Output:
{output}

Constraints to check (if explicitly listed):
{constraints}

Evaluate how well the output follows ALL instructions and constraints in the prompt. Consider:
1. Does the output follow the specified format (e.g., JSON, bullet points, specific length)?
2. Does the output respect all explicit constraints (e.g., "don't include X", "always include Y")?
3. Does the output match the requested tone, style, or persona?
4. Does the output stay within any topic or scope restrictions?
5. Are there any instructions that were ignored or violated?

Score from 0 to 10:
- 10: Output perfectly follows all instructions and constraints
- 7-9: Output follows most instructions with minor deviations
- 4-6: Output partially follows instructions; several deviations
- 1-3: Output largely ignores key instructions
- 0: Output completely disregards the prompt instructions

Return as JSON: {{
  "score": <0-10>,
  "followed_instructions": ["..."],
  "violated_instructions": ["..."],
  "reason": "..."
}}

Evaluation:"""


class PromptAlignmentMetric(BaseMetric):
    """
    Checks whether LLM output follows the instructions and constraints specified in the prompt.

    This metric is useful for testing instruction-following capability — whether the model
    respects formatting requirements, content constraints, style instructions, etc.

    Requires: input (the prompt/instructions), actual_output
    Optional: constraints (explicit list of constraints to check)
    """

    def __init__(
        self,
        constraints: Optional[List[str]] = None,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model
        self.constraints = constraints or []

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        # Build constraints string
        if self.constraints:
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
        else:
            constraints_str = "None explicitly listed — infer from the prompt."

        raw = self._llm_judge(
            _PROMPT_ALIGNMENT_PROMPT.format(
                prompt=test_case.input,
                output=test_case.actual_output,
                constraints=constraints_str,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        followed = self._parse_json_field(raw, "followed_instructions") or []
        violated = self._parse_json_field(raw, "violated_instructions") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if followed:
            eval_steps.append(f"Followed: {', '.join(followed)}")
        if violated:
            eval_steps.append(f"Violated: {', '.join(violated)}")
        if not eval_steps:
            eval_steps = ["Prompt alignment evaluated."]

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
