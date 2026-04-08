"""TextToImageMetric — evaluates text-to-image generation quality."""

from __future__ import annotations

from typing import Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_TEXT_TO_IMAGE_PROMPT = """You are evaluating the quality of a text-to-image generation.

Text Prompt: {prompt}
Image Description / Caption (provided by vision model or human): {image_description}
Generation Model: {model_name}

Since we cannot directly analyze the image pixels, evaluate based on the image description
provided. Assess how well the generated image (as described) aligns with the text prompt.

Evaluate on these dimensions:
1. Prompt fidelity: Does the image match all elements mentioned in the prompt?
2. Subject accuracy: Are the main subjects correctly depicted?
3. Style alignment: Does the style/mood/tone of the image match the prompt?
4. Detail completeness: Are specific details from the prompt represented?
5. Coherence: Is the image internally coherent (no contradictory elements)?

Score from 0 to 10:
- 10: Image perfectly represents all aspects of the prompt
- 7-9: Image captures the main elements with minor omissions
- 4-6: Image partially matches the prompt; some key elements missing
- 1-3: Image significantly misses the prompt intent
- 0: Image bears no resemblance to the prompt

Return as JSON: {{
  "score": <0-10>,
  "matched_elements": ["..."],
  "missing_elements": ["..."],
  "style_match": true/false,
  "reason": "..."
}}

Evaluation:"""


class TextToImageMetric(BaseMetric):
    """
    Evaluates text-to-image generation quality by assessing alignment between
    the text prompt and the generated image.

    Since LLMs cannot directly process images, this metric uses an image
    description (from a vision model or human annotator) and evaluates
    whether the described image faithfully represents the text prompt.

    Requires:
        - input: The text prompt used for generation
        - actual_output: A textual description of the generated image
          (e.g., from a vision LLM like GPT-4V, or human description)

    Optional:
        - additional_metadata["model_name"]: Name of the image generation model
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
        model_name = test_case.additional_metadata.get("model_name", "Unknown")

        raw = self._llm_judge(
            _TEXT_TO_IMAGE_PROMPT.format(
                prompt=test_case.input,
                image_description=test_case.actual_output,
                model_name=model_name,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        matched = self._parse_json_field(raw, "matched_elements") or []
        missing = self._parse_json_field(raw, "missing_elements") or []
        style_match = self._parse_json_field(raw, "style_match")
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if style_match is not None:
            eval_steps.append(f"Style match: {style_match}")
        if matched:
            eval_steps.append(f"Matched elements: {', '.join(matched[:3])}")
        if missing:
            eval_steps.append(f"Missing elements: {', '.join(missing[:3])}")
        if not eval_steps:
            eval_steps = ["Text-to-image alignment evaluated."]

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
