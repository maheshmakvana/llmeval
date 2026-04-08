"""TurnRelevancyMetric — per-turn relevancy evaluation in multi-turn conversations."""

from __future__ import annotations

from typing import List

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase, Message

_TURN_RELEVANCY_PROMPT = """You are evaluating whether an assistant's response is relevant to the user's message in a conversation.

Conversation context (previous turns):
{context}

User message (current turn): {user_message}
Assistant response (current turn): {assistant_response}

Evaluate whether the assistant's response is directly relevant to the user's current message.
Consider:
1. Does the response address what the user specifically asked or said?
2. Does it stay on topic relative to the user's message?
3. Does it make use of appropriate context from earlier turns?
4. Are there any off-topic or irrelevant elements in the response?

Score from 0 to 10 (10 = perfectly relevant, 0 = completely off-topic).
Return as JSON: {{"score": <0-10>, "relevant_aspects": ["..."], "irrelevant_aspects": ["..."], "reason": "..."}}

Evaluation:"""


class TurnRelevancyMetric(BaseMetric):
    """
    Evaluates relevancy on a per-turn basis in multi-turn conversations.

    Unlike ConversationalRelevancyMetric (which looks at overall conversation),
    this metric evaluates each individual assistant turn's relevancy to the
    corresponding user message, then aggregates across turns.

    Requires: ConversationalTestCase with interleaved user/assistant messages
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

    def _format_context(self, messages: List[Message], up_to_index: int) -> str:
        if up_to_index == 0:
            return "None (first turn)"
        return "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in messages[:up_to_index]
        )

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("TurnRelevancyMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        messages = test_case.messages

        # Find all user-assistant pairs
        turn_scores = []
        eval_steps = []

        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                context_str = self._format_context(messages, i)
                raw = self._llm_judge(
                    _TURN_RELEVANCY_PROMPT.format(
                        context=context_str,
                        user_message=messages[i].content,
                        assistant_response=messages[i + 1].content,
                    ),
                    provider=provider,
                )
                score_raw = self._parse_json_field(raw, "score") or 0
                reason = self._parse_json_field(raw, "reason") or raw[:100]
                turn_score = float(score_raw) / 10.0
                turn_scores.append(turn_score)
                eval_steps.append(f"Turn {i//2 + 1} relevancy: {turn_score:.2f} — {reason}")
                i += 2
            else:
                i += 1

        if not turn_scores:
            score = 1.0
            reason = "No user-assistant turn pairs found to evaluate."
        else:
            score = sum(turn_scores) / len(turn_scores)
            reason = f"Average per-turn relevancy across {len(turn_scores)} turns: {score:.2f}"

        passed = self._pass_fail(score)
        if not eval_steps:
            eval_steps = ["No evaluable turns found."]

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
            verbose_logs="\n".join(eval_steps) if self.verbose_mode else None,
        )
        return self._result
