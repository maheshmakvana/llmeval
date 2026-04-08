"""ArenaGEvalMetric — comparative A/B evaluation between two LLM outputs."""

from __future__ import annotations

import re
from typing import List, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase

_ARENA_PROMPT = """You are an impartial judge evaluating two AI assistant responses to the same input.

Input / Question:
{input}

Response A:
{response_a}

Response B:
{response_b}

Evaluation criteria: {criteria}

Evaluation steps:
{steps}

Compare the two responses carefully based on the criteria. Consider:
1. Which response better addresses the input?
2. Which response is more accurate, helpful, and relevant?
3. Which response demonstrates better quality on the specified criteria?
4. Are there specific strengths or weaknesses of each response?

Provide a comparative score for each response from 0 to 10.
Then determine the winner: "A", "B", or "tie".

Output your response as:
Reasoning: <your chain-of-thought comparison>
Score A: <integer from 0 to 10>
Score B: <integer from 0 to 10>
Winner: <"A", "B", or "tie">
"""

_GENERATE_STEPS_PROMPT = """Generate a numbered list of evaluation steps for comparing two AI responses based on the following criteria.

Criteria: {criteria}

Evaluation steps (numbered list):"""


class ArenaGEvalMetric(BaseMetric):
    """
    Comparative G-Eval for A/B testing between two LLM outputs (Arena-style).

    Evaluates two responses to the same input against custom criteria and
    determines which is better (or if they're equal). This is useful for
    model comparison, red-teaming, and preference learning.

    The score represents how much better Response A is relative to Response B:
    - score > 0.5: Response A wins
    - score == 0.5: Tie
    - score < 0.5: Response B wins

    Requires: input, actual_output (Response A)
    Requires: response_b — set via additional_metadata["response_b"] or constructor.

    Example:
        metric = ArenaGEvalMetric(
            criteria="Which response is more helpful, accurate, and concise?",
            response_b="The other model's response here.",
        )
        tc = LLMTestCase(
            input="What is quantum entanglement?",
            actual_output="Response from Model A...",
        )
        result = metric.measure(tc)
        # result.score > 0.5 means A won
    """

    def __init__(
        self,
        criteria: str = "Which response is more helpful, accurate, and relevant to the input?",
        response_b: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        threshold: float = 0.5,
        model=None,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self.criteria = criteria
        self.response_b = response_b
        self._eval_steps = evaluation_steps
        self._model = model

    def _get_eval_steps(self, provider) -> List[str]:
        if self._eval_steps:
            return self._eval_steps
        raw = self._llm_judge(
            _GENERATE_STEPS_PROMPT.format(criteria=self.criteria),
            provider=provider,
        )
        lines = re.findall(r"\d+\.\s+(.+)", raw)
        return lines or [self.criteria]

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        provider = self._model or self._get_default_provider()

        # Get response_b from constructor or metadata
        response_b = (
            self.response_b
            or test_case.additional_metadata.get("response_b")
            or test_case.expected_output
        )
        if not response_b:
            raise ValueError(
                "ArenaGEvalMetric requires a second response. "
                "Set response_b in the constructor or test_case.additional_metadata['response_b']."
            )

        steps = self._get_eval_steps(provider)
        response = self._llm_judge(
            _ARENA_PROMPT.format(
                input=test_case.input,
                response_a=test_case.actual_output,
                response_b=response_b,
                criteria=self.criteria,
                steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
            ),
            provider=provider,
        )

        # Parse scores
        score_a_match = re.search(r"(?i)score\s+a\s*[:\-]\s*(\d+(?:\.\d+)?)", response)
        score_b_match = re.search(r"(?i)score\s+b\s*[:\-]\s*(\d+(?:\.\d+)?)", response)
        winner_match = re.search(r"(?i)winner\s*[:\-]\s*(A|B|tie)", response)

        score_a = float(score_a_match.group(1)) / 10.0 if score_a_match else 0.5
        score_b = float(score_b_match.group(1)) / 10.0 if score_b_match else 0.5
        winner = winner_match.group(1).upper() if winner_match else "tie"

        # Overall score: relative advantage of A over B (0=B wins, 0.5=tie, 1=A wins)
        if winner == "tie":
            score = 0.5
        elif winner == "A":
            score = min(1.0, 0.5 + (score_a - score_b) / 2)
        else:  # B wins
            score = max(0.0, 0.5 - (score_b - score_a) / 2)

        # Parse reasoning
        reason_match = re.search(
            r"(?i)reasoning\s*[:\-]\s*(.+?)(?=score\s+[ab]\s*[:\-]|\Z)",
            response,
            re.DOTALL,
        )
        reason = reason_match.group(1).strip() if reason_match else response[:200]
        reason = f"Winner: {winner}. Score A: {score_a:.2f}, Score B: {score_b:.2f}. {reason}"

        passed = self._pass_fail(score)
        eval_steps = steps + [
            f"Response A score: {score_a:.2f}",
            f"Response B score: {score_b:.2f}",
            f"Winner: {winner}",
        ]

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
            verbose_logs=response if self.verbose_mode else None,
        )
        return self._result
