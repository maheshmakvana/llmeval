"""ConversationalGEvalMetric — G-Eval adapted for multi-turn conversations."""

from __future__ import annotations

import re
from typing import List, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase

_COT_CONV_PROMPT = """You are an evaluation assistant. Your task is to evaluate a multi-turn conversation based on specific criteria.

Evaluation criteria: {criteria}

Evaluation steps to follow:
{steps}

Multi-turn conversation:
{conversation}

Additional context:
- Chatbot role/persona: {chatbot_role}
- Total turns: {turns}

Think step by step through the conversation, evaluating each aspect of the criteria. Then provide a final score.

Score from 0 to 10 (10 = conversation perfectly satisfies the criteria, 0 = completely fails).

Output your response as:
Reasoning: <your chain-of-thought reasoning across all turns>
Score: <integer from 0 to 10>
"""

_GENERATE_STEPS_PROMPT = """Generate a numbered list of evaluation steps for assessing a multi-turn conversation based on the following criteria.
Each step should be a concrete, measurable check applicable to a full conversation.

Criteria: {criteria}

Evaluation steps (numbered list):"""


class ConversationalGEvalMetric(BaseMetric):
    """
    G-Eval adapted for multi-turn conversational evaluation.

    Uses chain-of-thought reasoning to evaluate a full conversation against
    any custom criteria. The judge LLM can auto-generate evaluation steps
    from the criteria if not provided.

    Example:
        metric = ConversationalGEvalMetric(
            name="EngagementQuality",
            criteria="The assistant maintains engaging, helpful, and contextually aware responses throughout the conversation.",
        )
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_steps: Optional[List[str]] = None,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._name = name
        self.criteria = criteria
        self._eval_steps = evaluation_steps
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    def _get_eval_steps(self, provider) -> List[str]:
        if self._eval_steps:
            return self._eval_steps
        raw = self._llm_judge(
            _GENERATE_STEPS_PROMPT.format(criteria=self.criteria),
            provider=provider,
        )
        lines = re.findall(r"\d+\.\s+(.+)", raw)
        return lines or [self.criteria]

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("ConversationalGEvalMetric requires a ConversationalTestCase.")

        provider = self._model or self._get_default_provider()
        steps = self._get_eval_steps(provider)

        # Format conversation
        conv_str = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in test_case.messages
        )
        chatbot_role = test_case.chatbot_role or "Not specified"

        prompt = _COT_CONV_PROMPT.format(
            criteria=self.criteria,
            steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
            conversation=conv_str,
            chatbot_role=chatbot_role,
            turns=test_case.turns,
        )
        response = self._llm_judge(prompt, provider=provider)

        score = self._parse_score_from_response(response, scale=10)

        reason_match = re.search(
            r"(?i)reasoning\s*[:\-]\s*(.+?)(?=score\s*[:\-]|\Z)",
            response,
            re.DOTALL,
        )
        reason = reason_match.group(1).strip() if reason_match else response[:200]

        passed = self._pass_fail(score)
        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score,
            passed=passed,
            reason=reason,
            metric_name=self.name,
            threshold=self.threshold,
            strict_mode=self.strict_mode,
            evaluation_steps=steps,
            verbose_logs=response if self.verbose_mode else None,
        )
        return self._result
