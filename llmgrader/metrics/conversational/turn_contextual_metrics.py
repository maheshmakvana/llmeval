"""Per-turn contextual metrics: Precision, Recall, and Relevancy for multi-turn conversations."""

from __future__ import annotations

from typing import List

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase, Message

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _get_turn_context(user_msg: Message, assistant_msg: Message) -> List[str]:
    """Extract retrieval context from a turn's associated LLMTestCase."""
    if assistant_msg.llm_test_case and assistant_msg.llm_test_case.retrieval_context:
        return assistant_msg.llm_test_case.retrieval_context
    if user_msg.llm_test_case and user_msg.llm_test_case.retrieval_context:
        return user_msg.llm_test_case.retrieval_context
    return []


def _get_turn_expected(user_msg: Message, assistant_msg: Message) -> str:
    """Extract expected output from a turn's associated LLMTestCase."""
    if assistant_msg.llm_test_case and assistant_msg.llm_test_case.expected_output:
        return assistant_msg.llm_test_case.expected_output
    if user_msg.llm_test_case and user_msg.llm_test_case.expected_output:
        return user_msg.llm_test_case.expected_output
    return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PRECISION_PROMPT = """You are evaluating contextual precision for a conversation turn.

User question: {question}
Assistant response: {response}
Retrieved context chunks (ordered by retrieval rank):
{context}

Contextual Precision evaluates whether the MOST RELEVANT context chunks are ranked higher.
Higher-ranked (earlier) chunks should be more relevant to generating the response than lower-ranked ones.

For each context chunk, determine if it was actually useful/needed to generate the response:
- Relevant: The chunk contributed to the response
- Irrelevant: The chunk was not needed or used

Score from 0 to 10 based on how well the ranking prioritizes relevant chunks.
- 10: All relevant chunks appear before irrelevant ones (perfect precision)
- 0: Irrelevant chunks appear before relevant ones (worst precision)

Return as JSON: {{
  "score": <0-10>,
  "chunk_verdicts": ["relevant"/"irrelevant", ...],
  "reason": "..."
}}

Evaluation:"""

_RECALL_PROMPT = """You are evaluating contextual recall for a conversation turn.

User question: {question}
Expected ideal response: {expected}
Assistant actual response: {response}
Retrieved context:
{context}

Contextual Recall evaluates whether the retrieved context contains all the information
needed to produce the expected (ideal) response.

Identify each piece of information in the expected response and check if it's supported
by the retrieved context.

Score from 0 to 10:
- 10: All information needed for the expected response is present in the context
- 0: None of the needed information is in the context

Return as JSON: {{
  "score": <0-10>,
  "covered_aspects": ["..."],
  "missing_aspects": ["..."],
  "reason": "..."
}}

Evaluation:"""

_RELEVANCY_PROMPT = """You are evaluating contextual relevancy for a conversation turn.

User question: {question}
Retrieved context chunks:
{context}

Contextual Relevancy evaluates what proportion of the retrieved context is actually
relevant to answering the user's question (signal-to-noise ratio).

For each chunk, decide: is this chunk relevant to the question or not?

Score from 0 to 10 based on the proportion of relevant chunks:
- 10: All retrieved chunks are highly relevant to the question
- 5: About half the chunks are relevant
- 0: No retrieved chunks are relevant

Return as JSON: {{
  "score": <0-10>,
  "relevant_chunks": <count>,
  "total_chunks": <count>,
  "reason": "..."
}}

Evaluation:"""


# ---------------------------------------------------------------------------
# TurnContextualPrecisionMetric
# ---------------------------------------------------------------------------

class TurnContextualPrecisionMetric(BaseMetric):
    """
    Evaluates contextual precision on a per-turn basis in multi-turn conversations.

    Checks whether the most relevant context chunks are ranked higher in the
    retrieval results for each conversation turn.

    Requires: ConversationalTestCase with per-turn retrieval_context in llm_test_case.
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
            raise TypeError("TurnContextualPrecisionMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        messages = test_case.messages
        turn_scores = []
        eval_steps = []

        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                context_docs = _get_turn_context(messages[i], messages[i + 1])
                if not context_docs:
                    i += 2
                    continue

                context_str = "\n".join(
                    f"[{j+1}] {doc}" for j, doc in enumerate(context_docs)
                )
                raw = self._llm_judge(
                    _PRECISION_PROMPT.format(
                        question=messages[i].content,
                        response=messages[i + 1].content,
                        context=context_str,
                    ),
                    provider=provider,
                )
                score_raw = self._parse_json_field(raw, "score") or 0
                reason = self._parse_json_field(raw, "reason") or raw[:100]
                turn_score = float(score_raw) / 10.0
                turn_scores.append(turn_score)
                eval_steps.append(f"Turn {i//2 + 1} contextual precision: {turn_score:.2f} — {reason}")
                i += 2
            else:
                i += 1

        if not turn_scores:
            score, reason = 1.0, "No turns with retrieval context found."
        else:
            score = sum(turn_scores) / len(turn_scores)
            reason = f"Average contextual precision across {len(turn_scores)} turns: {score:.2f}"

        passed = self._pass_fail(score)
        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode,
            evaluation_steps=eval_steps or ["No evaluable turns found."],
            verbose_logs="\n".join(eval_steps) if self.verbose_mode else None,
        )
        return self._result


# ---------------------------------------------------------------------------
# TurnContextualRecallMetric
# ---------------------------------------------------------------------------

class TurnContextualRecallMetric(BaseMetric):
    """
    Evaluates contextual recall on a per-turn basis in multi-turn conversations.

    Checks whether the retrieved context contains all information needed to
    produce the expected response for each turn.

    Requires: ConversationalTestCase with per-turn retrieval_context AND
              expected_output in llm_test_case.
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
            raise TypeError("TurnContextualRecallMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        messages = test_case.messages
        turn_scores = []
        eval_steps = []

        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                context_docs = _get_turn_context(messages[i], messages[i + 1])
                expected = _get_turn_expected(messages[i], messages[i + 1])

                if not context_docs or not expected:
                    i += 2
                    continue

                context_str = "\n---\n".join(context_docs)
                raw = self._llm_judge(
                    _RECALL_PROMPT.format(
                        question=messages[i].content,
                        expected=expected,
                        response=messages[i + 1].content,
                        context=context_str,
                    ),
                    provider=provider,
                )
                score_raw = self._parse_json_field(raw, "score") or 0
                missing = self._parse_json_field(raw, "missing_aspects") or []
                reason = self._parse_json_field(raw, "reason") or raw[:100]
                turn_score = float(score_raw) / 10.0
                turn_scores.append(turn_score)
                step = f"Turn {i//2 + 1} contextual recall: {turn_score:.2f}"
                if missing:
                    step += f" — Missing: {', '.join(missing[:2])}"
                eval_steps.append(step)
                i += 2
            else:
                i += 1

        if not turn_scores:
            score, reason = 1.0, "No turns with retrieval context and expected output found."
        else:
            score = sum(turn_scores) / len(turn_scores)
            reason = f"Average contextual recall across {len(turn_scores)} turns: {score:.2f}"

        passed = self._pass_fail(score)
        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode,
            evaluation_steps=eval_steps or ["No evaluable turns found."],
            verbose_logs="\n".join(eval_steps) if self.verbose_mode else None,
        )
        return self._result


# ---------------------------------------------------------------------------
# TurnContextualRelevancyMetric
# ---------------------------------------------------------------------------

class TurnContextualRelevancyMetric(BaseMetric):
    """
    Evaluates contextual relevancy on a per-turn basis in multi-turn conversations.

    Measures what proportion of the retrieved context chunks are actually relevant
    to the user's question for each turn (signal-to-noise ratio).

    Requires: ConversationalTestCase with per-turn retrieval_context in llm_test_case.
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
            raise TypeError("TurnContextualRelevancyMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        messages = test_case.messages
        turn_scores = []
        eval_steps = []

        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                context_docs = _get_turn_context(messages[i], messages[i + 1])
                if not context_docs:
                    i += 2
                    continue

                context_str = "\n".join(
                    f"[{j+1}] {doc}" for j, doc in enumerate(context_docs)
                )
                raw = self._llm_judge(
                    _RELEVANCY_PROMPT.format(
                        question=messages[i].content,
                        context=context_str,
                    ),
                    provider=provider,
                )
                score_raw = self._parse_json_field(raw, "score") or 0
                relevant_count = self._parse_json_field(raw, "relevant_chunks") or 0
                total_count = self._parse_json_field(raw, "total_chunks") or len(context_docs)
                reason = self._parse_json_field(raw, "reason") or raw[:100]
                turn_score = float(score_raw) / 10.0
                turn_scores.append(turn_score)
                eval_steps.append(
                    f"Turn {i//2 + 1} contextual relevancy: {turn_score:.2f} "
                    f"({relevant_count}/{total_count} chunks relevant)"
                )
                i += 2
            else:
                i += 1

        if not turn_scores:
            score, reason = 1.0, "No turns with retrieval context found."
        else:
            score = sum(turn_scores) / len(turn_scores)
            reason = f"Average contextual relevancy across {len(turn_scores)} turns: {score:.2f}"

        passed = self._pass_fail(score)
        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode,
            evaluation_steps=eval_steps or ["No evaluable turns found."],
            verbose_logs="\n".join(eval_steps) if self.verbose_mode else None,
        )
        return self._result
