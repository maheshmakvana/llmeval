"""TopicAdherenceMetric — checks if the output stays on the intended topic."""

from __future__ import annotations

from typing import List, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_TOPIC_ADHERENCE_PROMPT = """You are evaluating whether an LLM's output adheres to the intended topic(s).

Input / Question: {input}
Intended topic(s): {topics}
LLM Output:
{output}

Evaluate how well the output stays on the specified topic(s). Consider:
1. Does the output primarily address the intended topic(s)?
2. Are there significant off-topic digressions or tangents?
3. If multiple topics are specified, does the output address all of them?
4. Is the response focused, or does it wander into unrelated areas?
5. Is any off-topic content disruptive to the overall response quality?

Score from 0 to 10:
- 10: Output stays perfectly on-topic throughout
- 7-9: Output is mostly on-topic with minor relevant tangents
- 4-6: Output partially addresses the topic but has notable off-topic content
- 1-3: Output is mostly off-topic
- 0: Output is entirely unrelated to the intended topic(s)

Return as JSON: {{
  "score": <0-10>,
  "on_topic_aspects": ["..."],
  "off_topic_aspects": ["..."],
  "topic_coverage": <0-1 float>,
  "reason": "..."
}}

Evaluation:"""


class TopicAdherenceMetric(BaseMetric):
    """
    Checks whether an LLM's output stays on the intended topic(s).

    Useful for content moderation, focused assistant testing, and ensuring
    chatbots don't stray from their designated domain.

    Requires: input, actual_output
    Optional: topics (list of topic strings to enforce)
              If no topics provided, topics are inferred from the input.
    """

    def __init__(
        self,
        topics: Optional[List[str]] = None,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model
        self.topics = topics or []

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        if self.topics:
            topics_str = ", ".join(f"'{t}'" for t in self.topics)
        else:
            topics_str = "Infer the intended topic from the input question"

        raw = self._llm_judge(
            _TOPIC_ADHERENCE_PROMPT.format(
                input=test_case.input,
                topics=topics_str,
                output=test_case.actual_output,
            ),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        on_topic = self._parse_json_field(raw, "on_topic_aspects") or []
        off_topic = self._parse_json_field(raw, "off_topic_aspects") or []
        coverage = self._parse_json_field(raw, "topic_coverage")
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)

        eval_steps = []
        if coverage is not None:
            eval_steps.append(f"Topic coverage: {float(coverage):.0%}")
        if on_topic:
            eval_steps.append(f"On-topic: {', '.join(on_topic[:3])}")
        if off_topic:
            eval_steps.append(f"Off-topic: {', '.join(off_topic[:3])}")
        if not eval_steps:
            eval_steps = ["Topic adherence evaluated."]

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
