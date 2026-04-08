"""ImageEditingMetric — evaluates image editing quality."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_IMAGE_EDITING_PROMPT = """You are evaluating the quality of an AI image editing operation.

Editing instruction: {instruction}
Original image description: {original_description}
Edited image description: {edited_description}

Evaluate the quality of the image edit based on these dimensions:
1. Instruction adherence: Was the editing instruction correctly applied?
2. Preservation: Were the parts of the image NOT mentioned in the instruction preserved?
3. Naturalness: Does the edited image look natural (no artifacts described)?
4. Completeness: Was the edit fully applied (not partial)?
5. Quality: Is the overall edited image high quality?

Score from 0 to 10:
- 10: Edit perfectly follows instruction, preserves everything else, looks natural
- 7-9: Edit is mostly correct with minor issues
- 4-6: Edit partially applied or some unwanted changes occurred
- 1-3: Edit largely failed or caused significant unintended changes
- 0: Edit completely failed or made the image worse

Return as JSON: {{
  "score": <0-10>,
  "instruction_applied": true/false,
  "preservation_score": <0-10>,
  "unintended_changes": ["..."],
  "reason": "..."
}}

Evaluation:"""


class ImageEditingMetric(BaseMetric):
    """
    Evaluates the quality of AI image editing operations.

    Assesses whether the editing instruction was correctly applied, whether
    unedited parts were preserved, and the overall quality of the edit.

    Since LLMs cannot directly process images, this metric uses textual
    descriptions of the original and edited images.

    Requires:
        - input: The editing instruction (e.g., "Remove the background")
        - actual_output: Description of the edited image
        - additional_metadata["original_description"]: Description of the original image
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
        original_desc = test_case.additional_metadata.get(
            "original_description",
            test_case.expected_output or "Not provided",
        )

        raw = self._llm_judge(
            _IMAGE_EDITING_PROMPT.format(
                instruction=test_case.input,
                original_description=original_desc,
                edited_description=test_case.actual_output,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        instruction_applied = self._parse_json_field(raw, "instruction_applied")
        preservation_score = self._parse_json_field(raw, "preservation_score")
        unintended = self._parse_json_field(raw, "unintended_changes") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if instruction_applied is not None:
            eval_steps.append(f"Instruction applied: {instruction_applied}")
        if preservation_score is not None:
            eval_steps.append(f"Preservation score: {float(preservation_score)/10:.2f}")
        if unintended:
            eval_steps.append(f"Unintended changes: {', '.join(unintended[:3])}")
        if not eval_steps:
            eval_steps = ["Image editing quality evaluated."]

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
