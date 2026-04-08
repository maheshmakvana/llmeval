"""ImageCoherenceMetric — evaluates internal coherence and consistency of an image."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_IMAGE_COHERENCE_PROMPT = """You are evaluating the internal coherence and consistency of an AI-generated or edited image.

Image description:
{image_description}

Generation/Edit context: {context}

Evaluate whether the described image is internally coherent and consistent. Consider:
1. Visual consistency: Do all elements in the image belong together (consistent style, lighting, perspective)?
2. Physical plausibility: Are there impossible or physically implausible elements?
3. Semantic coherence: Do the described objects/elements make sense together?
4. Style consistency: Is the artistic style uniform throughout?
5. Color/lighting coherence: Are colors and lighting consistent across the image?
6. Absence of artifacts: Are there any described artifacts, distortions, or corrupted regions?

Score from 0 to 10:
- 10: Perfectly coherent — all elements are consistent and make visual sense
- 7-9: Mostly coherent with minor inconsistencies
- 4-6: Some noticeable inconsistencies or artifacts
- 1-3: Significant coherence issues
- 0: Image is incoherent (e.g., multiple conflicting styles, impossible elements)

Return as JSON: {{
  "score": <0-10>,
  "coherent_aspects": ["..."],
  "incoherent_aspects": ["..."],
  "artifacts_detected": ["..."],
  "reason": "..."
}}

Evaluation:"""


class ImageCoherenceMetric(BaseMetric):
    """
    Evaluates the internal coherence and consistency of AI-generated or edited images.

    Checks for visual consistency, physical plausibility, semantic coherence,
    style uniformity, and absence of artifacts.

    Requires:
        - actual_output: Textual description of the image (from vision model or human)
        - input: Optional context (prompt used to generate, or "no context")
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
        context = test_case.input or "No specific generation context provided."

        raw = self._llm_judge(
            _IMAGE_COHERENCE_PROMPT.format(
                image_description=test_case.actual_output,
                context=context,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        coherent = self._parse_json_field(raw, "coherent_aspects") or []
        incoherent = self._parse_json_field(raw, "incoherent_aspects") or []
        artifacts = self._parse_json_field(raw, "artifacts_detected") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if coherent:
            eval_steps.append(f"Coherent aspects: {', '.join(coherent[:3])}")
        if incoherent:
            eval_steps.append(f"Incoherent aspects: {', '.join(incoherent[:3])}")
        if artifacts:
            eval_steps.append(f"Artifacts: {', '.join(artifacts[:3])}")
        if not eval_steps:
            eval_steps = ["Image coherence evaluated."]

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
