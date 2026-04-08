"""ImageHelpfulnessMetric — evaluates whether an image is helpful for the user's need."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_IMAGE_HELPFULNESS_PROMPT = """You are evaluating whether an AI-generated image is helpful for the user's stated need.

User's need / request: {user_need}
Image description:
{image_description}

Evaluate how helpful the described image is in addressing the user's need. Consider:
1. Relevance: Does the image address what the user asked for?
2. Utility: Would this image actually be useful for the stated purpose?
3. Information density: Does the image convey sufficient relevant information?
4. Clarity: Is the image clear enough to be useful (based on its description)?
5. Completeness: Does the image fully address the user's need or only partially?
6. Actionability: Can the user use this image for their intended purpose?

Score from 0 to 10:
- 10: Image is extremely helpful — perfectly addresses the user's need
- 7-9: Image is quite helpful with minor limitations
- 4-6: Image is partially helpful but missing key elements for the user's purpose
- 1-3: Image is minimally helpful — barely addresses the user's need
- 0: Image is not helpful at all for the stated need

Return as JSON: {{
  "score": <0-10>,
  "helpful_aspects": ["..."],
  "unhelpful_aspects": ["..."],
  "would_serve_purpose": true/false,
  "reason": "..."
}}

Evaluation:"""


class ImageHelpfulnessMetric(BaseMetric):
    """
    Evaluates whether an AI-generated or retrieved image is helpful for the user's need.

    Assesses relevance, utility, information density, clarity, and completeness
    relative to what the user requested.

    Requires:
        - input: The user's need or request
        - actual_output: Textual description of the image (from vision model or human)
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

        raw = self._llm_judge(
            _IMAGE_HELPFULNESS_PROMPT.format(
                user_need=test_case.input,
                image_description=test_case.actual_output,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        helpful = self._parse_json_field(raw, "helpful_aspects") or []
        unhelpful = self._parse_json_field(raw, "unhelpful_aspects") or []
        serves_purpose = self._parse_json_field(raw, "would_serve_purpose")
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if serves_purpose is not None:
            eval_steps.append(f"Would serve purpose: {serves_purpose}")
        if helpful:
            eval_steps.append(f"Helpful aspects: {', '.join(helpful[:3])}")
        if unhelpful:
            eval_steps.append(f"Unhelpful aspects: {', '.join(unhelpful[:3])}")
        if not eval_steps:
            eval_steps = ["Image helpfulness evaluated."]

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
