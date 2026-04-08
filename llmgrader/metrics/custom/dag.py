"""DAGMetric — deterministic decision-tree evaluation (Deep Acyclic Graph)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase


@dataclass
class DAGNode:
    """A node in the evaluation decision tree."""

    condition: Callable[[LLMTestCase], bool]
    score_if_true: Optional[float] = None
    score_if_false: Optional[float] = None
    next_if_true: Optional["DAGNode"] = None
    next_if_false: Optional["DAGNode"] = None
    label: str = ""

    def evaluate(self, test_case: LLMTestCase) -> float:
        result = self.condition(test_case)
        if result:
            if self.next_if_true is not None:
                return self.next_if_true.evaluate(test_case)
            return self.score_if_true if self.score_if_true is not None else 1.0
        else:
            if self.next_if_false is not None:
                return self.next_if_false.evaluate(test_case)
            return self.score_if_false if self.score_if_false is not None else 0.0


class DAGMetric(BaseMetric):
    """
    Deterministic, decision-tree-based evaluation metric.

    Build a tree of DAGNode conditions that deterministically score a test case.
    Useful when you have clear pass/fail business rules.

    Example:
        dag = DAGNode(
            condition=lambda tc: len(tc.actual_output) > 0,
            score_if_false=0.0,
            next_if_true=DAGNode(
                condition=lambda tc: "error" not in tc.actual_output.lower(),
                score_if_true=1.0,
                score_if_false=0.2,
            )
        )
        metric = DAGMetric(name="ResponseQuality", root=dag, threshold=0.5)
    """

    def __init__(
        self,
        name: str,
        root: DAGNode,
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

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        score = self.root.evaluate(test_case)
        passed = self._pass_fail(score)
        reason = f"DAG evaluation produced score {score:.3f}."

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
