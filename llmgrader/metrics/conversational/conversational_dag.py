"""ConversationalDAGMetric — DAG-based deterministic evaluation for multi-turn conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import ConversationalTestCase


@dataclass
class ConversationalDAGNode:
    """A node in a conversation evaluation decision tree."""

    condition: Callable[[ConversationalTestCase], bool]
    score_if_true: Optional[float] = None
    score_if_false: Optional[float] = None
    next_if_true: Optional["ConversationalDAGNode"] = None
    next_if_false: Optional["ConversationalDAGNode"] = None
    label: str = ""

    def evaluate(self, test_case: ConversationalTestCase) -> float:
        result = self.condition(test_case)
        if result:
            if self.next_if_true is not None:
                return self.next_if_true.evaluate(test_case)
            return self.score_if_true if self.score_if_true is not None else 1.0
        else:
            if self.next_if_false is not None:
                return self.next_if_false.evaluate(test_case)
            return self.score_if_false if self.score_if_false is not None else 0.0


class ConversationalDAGMetric(BaseMetric):
    """
    Deterministic, decision-tree-based evaluation for multi-turn conversations.

    Like DAGMetric but accepts a ConversationalTestCase. Build a tree of
    ConversationalDAGNode conditions that deterministically score a conversation.

    Example:
        dag = ConversationalDAGNode(
            condition=lambda tc: tc.turns >= 2,
            score_if_false=0.0,
            next_if_true=ConversationalDAGNode(
                condition=lambda tc: any(
                    m.role == "assistant" for m in tc.messages
                ),
                score_if_true=1.0,
                score_if_false=0.1,
            )
        )
        metric = ConversationalDAGMetric(
            name="BasicConversationQuality",
            root=dag,
            threshold=0.5,
        )
    """

    def __init__(
        self,
        name: str,
        root: ConversationalDAGNode,
        threshold: float = 0.5,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self._name = name
        self.root = root

    @property
    def name(self) -> str:
        return self._name

    def measure(self, test_case: ConversationalTestCase) -> MetricResult:
        if not isinstance(test_case, ConversationalTestCase):
            raise TypeError("ConversationalDAGMetric requires a ConversationalTestCase.")

        score = self.root.evaluate(test_case)
        passed = self._pass_fail(score)
        reason = (
            f"Conversational DAG evaluation produced score {score:.3f} "
            f"over {test_case.turns} turns."
        )

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score,
            passed=passed,
            reason=reason,
            metric_name=self.name,
            threshold=self.threshold,
            strict_mode=self.strict_mode,
        )
        return self._result
