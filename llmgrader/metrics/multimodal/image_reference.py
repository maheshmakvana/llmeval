"""ImageReferenceMetric — checks an image against a reference image."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_IMAGE_REFERENCE_PROMPT = """You are evaluating how well a generated image matches a reference image.

Reference image description:
{reference_description}

Generated/Edited image description:
{generated_description}

Comparison context: {context}

Evaluate the similarity between the generated image and the reference image. Consider:
1. Subject similarity: Are the main subjects/objects the same or equivalent?
2. Compositional similarity: Is the layout and composition similar?
3. Style similarity: Are the artistic style, mood, and tone similar?
4. Color similarity: Are the color palette and tones similar?
5. Detail fidelity: Are important details from the reference preserved?
6. Overall visual likeness: How similar are the two images overall?

Score from 0 to 10:
- 10: Generated image is virtually identical to the reference
- 7-9: Images are very similar with minor differences
- 4-6: Images share the main subject but differ in significant details
- 1-3: Images are superficially related but substantially different
- 0: Images bear no resemblance

Return as JSON: {{
  "score": <0-10>,
  "matching_aspects": ["..."],
  "differing_aspects": ["..."],
  "similarity_level": "identical"/"very_similar"/"somewhat_similar"/"different"/"unrelated",
  "reason": "..."
}}

Evaluation:"""


class ImageReferenceMetric(BaseMetric):
    """
    Compares a generated/edited image against a reference image.

    Useful for style transfer validation, image-to-image generation quality,
    or checking that edits preserve the reference image's key characteristics.

    Requires:
        - actual_output: Description of the generated/edited image
        - expected_output OR additional_metadata["reference_description"]:
          Description of the reference image
        - input: Optional context about the transformation applied
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

        reference_desc = (
            test_case.additional_metadata.get("reference_description")
            or test_case.expected_output
        )
        if not reference_desc:
            raise ValueError(
                "ImageReferenceMetric requires a reference image description. "
                "Set expected_output or additional_metadata['reference_description']."
            )

        context = test_case.input or "No transformation context provided."

        raw = self._llm_judge(
            _IMAGE_REFERENCE_PROMPT.format(
                reference_description=reference_desc,
                generated_description=test_case.actual_output,
                context=context,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        matching = self._parse_json_field(raw, "matching_aspects") or []
        differing = self._parse_json_field(raw, "differing_aspects") or []
        similarity_level = self._parse_json_field(raw, "similarity_level") or "unknown"
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = [f"Similarity level: {similarity_level}"]
        if matching:
            eval_steps.append(f"Matching: {', '.join(matching[:3])}")
        if differing:
            eval_steps.append(f"Differing: {', '.join(differing[:3])}")

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
