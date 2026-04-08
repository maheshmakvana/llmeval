"""ConversationCompletenessMetric — checks all user intents are addressed."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase

_COMPLETENESS_PROMPT = """Analyze the following conversation and determine whether all user intents/requests were fully addressed by the assistant.

Conversation:
{conversation}

List all user intents/requests and whether each was fully addressed.
Score from 0 to 10 (10 = all intents addressed, 0 = none addressed).
Return as JSON: {{"score": <0-10>, "unaddressed_intents": ["..."], "reason": "..."}}

Analysis:"""


class ConversationCompletenessMetric(BaseMetric):
    """
    Measures whether the assistant addressed all user intents in the conversation.

    Requires: ConversationalTestCase
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("ConversationCompletenessMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        conv_str = "\n".join(f"{m.role.upper()}: {m.content}" for m in test_case.messages)

        raw = self._llm_judge(
            _COMPLETENESS_PROMPT.format(conversation=conv_str),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        unaddressed = self._parse_json_field(raw, "unaddressed_intents") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)
        steps = [f"Unaddressed: {u}" for u in unaddressed] or ["All intents addressed."]

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
