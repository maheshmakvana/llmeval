"""LlamaIndex integration — evaluate query engines and RAG pipelines."""

from __future__ import annotations

from typing import Any, List, Optional

from ..test_case import LLMTestCase


def evaluate_query_engine(
    query_engine: Any,
    questions: List[str],
    metrics: List[Any],
    include_source_nodes: bool = True,
) -> Any:
    """
    Evaluate a LlamaIndex query engine against a list of questions.

    Automatically captures the query string, response text, and
    source node texts as retrieval_context.

    Args:
        query_engine: A LlamaIndex query engine (BaseQueryEngine).
        questions: List of question strings.
        metrics: List of llmeval metrics.
        include_source_nodes: Include retrieved nodes as retrieval_context.

    Returns:
        EvaluationResult
    """
    from ..evaluate import evaluate

    test_cases = []
    for q in questions:
        try:
            response = query_engine.query(q)
            actual_output = str(response)
            retrieval_context = None
            if include_source_nodes and hasattr(response, "source_nodes"):
                retrieval_context = [
                    node.get_content() for node in response.source_nodes
                ]
        except Exception as e:
            actual_output = f"ERROR: {e}"
            retrieval_context = None

        test_cases.append(LLMTestCase(
            input=q,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        ))

    return evaluate(test_cases, metrics)


class LlamaIndexObserver:
    """
    Context manager that instruments a LlamaIndex query engine
    and collects trace data via llmeval's tracing system.

    Usage:
        with LlamaIndexObserver() as observer:
            response = query_engine.query("What is AI?")
        print(observer.get_test_cases())
    """

    def __init__(self) -> None:
        self._test_cases: List[LLMTestCase] = []
        self._last_query: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def record(self, query: str, response: str, retrieval_context: Optional[List[str]] = None) -> None:
        self._test_cases.append(LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=retrieval_context,
        ))

    def get_test_cases(self) -> List[LLMTestCase]:
        return self._test_cases
