"""TurnFaithfulnessMetric — per-turn faithfulness to context in multi-turn conversations."""

from __future__ import annotations

from typing import List

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase, Message

_TURN_FAITHFULNESS_PROMPT = """You are evaluating whether an assistant's response is faithful to the provided context.

User message: {user_message}
Assistant response: {assistant_response}
Context / Retrieved documents for this turn:
{context}

Evaluate whether the assistant's response is grounded in and faithful to the provided context.
Faithfulness means:
1. Claims in the response are supported by the context
2. The response does not contradict information in the context
3. The response does not introduce facts not present in the context (hallucination)
4. Information from context is accurately represented (no distortion)

Score from 0 to 10:
- 10: Every claim in the response is fully supported by the context
- 7-9: Most claims are supported; minor additions that don't contradict context
- 4-6: Mix of grounded and ungrounded claims
- 1-3: Many claims contradict or are unsupported by the context
- 0: Response is completely unfaithful to the context

Return as JSON: {{
  "score": <0-10>,
  "faithful_claims": ["..."],
  "unfaithful_claims": ["..."],
  "reason": "..."
}}

Evaluation:"""


class TurnFaithfulnessMetric(BaseMetric):
    """
    Evaluates faithfulness to context on a per-turn basis in multi-turn conversations.

    Each assistant turn is evaluated against the retrieval context associated with
    that turn's LLMTestCase (if available) or the global context. Scores are
    aggregated across all evaluated turns.

    Requires: ConversationalTestCase
    Best used when messages have associated llm_test_case with retrieval_context.
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

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("TurnFaithfulnessMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        messages = test_case.messages

        turn_scores = []
        eval_steps = []

        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                assistant_msg = messages[i + 1]
                user_msg = messages[i]

                # Get context: prefer per-turn llm_test_case context
                context_docs: List[str] = []
                if assistant_msg.llm_test_case and assistant_msg.llm_test_case.retrieval_context:
                    context_docs = assistant_msg.llm_test_case.retrieval_context
                elif user_msg.llm_test_case and user_msg.llm_test_case.retrieval_context:
                    context_docs = user_msg.llm_test_case.retrieval_context

                if not context_docs:
                    # Skip turns with no context — faithfulness requires context
                    i += 2
                    continue

                context_str = "\n---\n".join(context_docs)
                raw = self._llm_judge(
                    _TURN_FAITHFULNESS_PROMPT.format(
                        user_message=user_msg.content,
                        assistant_response=assistant_msg.content,
                        context=context_str,
                    ),
                    provider=provider,
                )

                score_raw = self._parse_json_field(raw, "score") or 0
                unfaithful = self._parse_json_field(raw, "unfaithful_claims") or []
                reason = self._parse_json_field(raw, "reason") or raw[:100]
                turn_score = float(score_raw) / 10.0
                turn_scores.append(turn_score)

                step_detail = f"Turn {i//2 + 1} faithfulness: {turn_score:.2f}"
                if unfaithful:
                    step_detail += f" — Unfaithful claims: {', '.join(unfaithful[:2])}"
                eval_steps.append(step_detail)
                i += 2
            else:
                i += 1

        if not turn_scores:
            score = 1.0
            reason = "No turns with retrieval context found; faithfulness not applicable."
        else:
            score = sum(turn_scores) / len(turn_scores)
            reason = f"Average per-turn faithfulness across {len(turn_scores)} turns: {score:.2f}"

        passed = self._pass_fail(score)
        if not eval_steps:
            eval_steps = ["No evaluable turns with context found."]

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
