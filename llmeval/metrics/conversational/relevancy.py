"""ConversationalRelevancyMetric — measures dialogue coherence across turns."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase

_RELEVANCY_PROMPT = """Review the following conversation and evaluate whether each assistant response is relevant to the preceding user message and overall conversation context.

Conversation:
{conversation}

For each assistant turn, determine if it is relevant (yes/no).
Return as JSON: {{"verdicts": [{{"turn": 1, "relevant": true/false, "reason": "..."}}], "overall_reason": "..."}}

Evaluation:"""


class ConversationalRelevancyMetric(BaseMetric):
    """
    Measures whether each assistant turn is relevant to the conversation context.

    Requires: ConversationalTestCase
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("ConversationalRelevancyMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        conv_str = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in test_case.messages
        )

        raw = self._llm_judge(
            _RELEVANCY_PROMPT.format(conversation=conv_str),
            provider=provider,
        )

        verdicts = self._parse_json_field(raw, "verdicts") or []
        overall_reason = self._parse_json_field(raw, "overall_reason") or raw[:200]

        if verdicts:
            relevant_count = sum(1 for v in verdicts if v.get("relevant", False))
            score = relevant_count / len(verdicts)
            steps = [f"Turn {v['turn']}: {'relevant' if v.get('relevant') else 'irrelevant'} — {v.get('reason','')}" for v in verdicts]
        else:
            score = 0.5
            steps = ["Could not parse individual turn verdicts."]

        passed = self._pass_fail(score)

        self._score = score
        self._reason = overall_reason
        self._result = MetricResult(
            score=score, passed=passed, reason=overall_reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
