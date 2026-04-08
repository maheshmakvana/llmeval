"""RoleAdherenceMetric — checks chatbot stays in its defined persona/role."""

from __future__ import annotations

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase

_ROLE_PROMPT = """A chatbot was given the following role/persona: "{role}"

Review the conversation and determine whether the chatbot consistently adhered to this role.
Score from 0 to 10 (10 = perfectly in-role, 0 = completely off-role).
Return as JSON: {{"score": <0-10>, "violations": ["..."], "reason": "..."}}

Conversation:
{conversation}

Evaluation:"""


class RoleAdherenceMetric(BaseMetric):
    """
    Evaluates whether a chatbot consistently adheres to its defined role/persona.

    Requires: ConversationalTestCase with chatbot_role set
    """

    def __init__(self, threshold: float = 0.5, model=None, strict_mode: bool = False, verbose_mode: bool = False) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._model = model

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("RoleAdherenceMetric requires a ConversationalTestCase.")
        if not test_case.chatbot_role:
            raise ValueError("RoleAdherenceMetric requires ConversationalTestCase.chatbot_role to be set.")

        provider = self._model or self._get_default_provider()
        conv_str = "\n".join(f"{m.role.upper()}: {m.content}" for m in test_case.messages)

        raw = self._llm_judge(
            _ROLE_PROMPT.format(role=test_case.chatbot_role, conversation=conv_str),
            provider=provider,
        )

        score_raw = self._parse_json_field(raw, "score") or 0
        violations = self._parse_json_field(raw, "violations") or []
        reason = self._parse_json_field(raw, "reason") or raw[:200]

        score = float(score_raw) / 10.0
        passed = self._pass_fail(score)
        steps = [f"Violation: {v}" for v in violations] or ["No role violations detected."]

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result
