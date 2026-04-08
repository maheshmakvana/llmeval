"""CrewAI integration — evaluate agent task outputs."""

from __future__ import annotations

from typing import Any, List, Optional

from ..test_case import LLMTestCase, ToolCall


def evaluate_crew(
    crew: Any,
    inputs: List[dict],
    metrics: List[Any],
) -> Any:
    """
    Evaluate a CrewAI crew by running it on a list of inputs.

    Args:
        crew: A CrewAI Crew instance.
        inputs: List of input dicts to pass to crew.kickoff().
        metrics: List of llmgrader metrics.

    Returns:
        EvaluationResult
    """
    from ..evaluate import evaluate

    test_cases = []
    for inp in inputs:
        try:
            result = crew.kickoff(inputs=inp)
            actual_output = str(result)
        except Exception as e:
            actual_output = f"ERROR: {e}"

        input_str = " | ".join(f"{k}={v}" for k, v in inp.items())
        test_cases.append(LLMTestCase(input=input_str, actual_output=actual_output))

    return evaluate(test_cases, metrics)


class CrewAITaskCapture:
    """
    Wraps a CrewAI task function to capture inputs and outputs
    as LLMTestCase objects for evaluation.

    Usage:
        capture = CrewAITaskCapture()

        @capture.track
        def my_agent_task(input):
            return agent.run(input)
    """

    def __init__(self) -> None:
        self._test_cases: List[LLMTestCase] = []

    def track(self, func):
        import functools

        @functools.wraps(func)
        def wrapper(input_str: str, *args, **kwargs):
            output = func(input_str, *args, **kwargs)
            self._test_cases.append(LLMTestCase(
                input=str(input_str),
                actual_output=str(output),
            ))
            return output
        return wrapper

    def get_test_cases(self) -> List[LLMTestCase]:
        return self._test_cases
