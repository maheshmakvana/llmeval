"""KnowledgeRetentionMetric — checks if the chatbot remembers key facts across turns."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase

_RETENTION_PROMPT = """Analyze whether the assistant correctly retained and used information provided earlier in the conversation.

Conversation:
{conversation}

Identify key facts shared by the user and check if the assistant referenced them correctly in later turns.
Score from 0 to 10 (10 = perfect retention, 0 = completely ignores prior context).
Return as JSON: {{"score": <0-10>, "forgotten_facts": ["..."], "reason": "..."}}

Analysis:"""


class KnowledgeRetentionMetric(BaseMetric):
    """
    Measures whether the chatbot retains and correctly uses information across conversation turns.

    Requires: ConversationalTestCase (multi-turn)
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("KnowledgeRetentionMetric requires a ConversationalTestCase.")
        if test_case.turns < 3:
            # Need at least 3 turns to have meaningful retention
            score, reason = 1.0, "Insufficient turns to evaluate knowledge retention (need >= 3 turns)."
            self._score = score
            self._reason = reason
            self._result = MetricResult(
                score=score, passed=True, reason=reason,
                metric_name=self.name, threshold=self.threshold, strict_mode=self.strict_mode,
            )
            return self._result

        provider = self._model or self._get_default_provider()
        conv_str = "\n".join(f"{m.role.upper()}: {m.content}" for m in test_case.messages)

        raw = self._llm_judge(
            _RETENTION_PROMPT.format(conversation=conv_str),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        forgotten = self._parse_json_field(raw, "forgotten_facts") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)
        steps = [f"Forgotten: {f}" for f in forgotten] or ["All key facts retained."]

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
