"""LangChain integration — callbacks and chain evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..test_case import LLMTestCase, ToolCall


class LangChainCallbackHandler:
    """
    LangChain callback handler that automatically captures
    LLM inputs/outputs into LLMTestCase-compatible format.

    Usage:
        from langchain_openai import ChatOpenAI
        from llmgrader.integrations.langchain import LangChainCallbackHandler

        handler = LangChainCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler])
        llm.invoke("What is 2+2?")
        test_cases = handler.get_test_cases()
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._current_input: Optional[str] = None
        self._tool_calls: List[ToolCall] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self._current_input = prompts[0] if prompts else ""

    def on_llm_end(self, response: Any, **kwargs) -> None:
        try:
            output = response.generations[0][0].text
        except (AttributeError, IndexError):
            output = str(response)

        self._records.append({
            "input": self._current_input or "",
            "actual_output": output,
            "tools_called": list(self._tool_calls),
        })
        self._tool_calls.clear()

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        name = serialized.get("name", "unknown_tool")
        self._tool_calls.append(ToolCall(name=name, input_parameters={"input": input_str}))

    def on_tool_end(self, output: str, **kwargs) -> None:
        if self._tool_calls:
            self._tool_calls[-1].output = output

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self._records.append({
            "input": self._current_input or "",
            "actual_output": f"ERROR: {error}",
        })

    def get_test_cases(self) -> List[LLMTestCase]:
        return [
            LLMTestCase(
                input=r["input"],
                actual_output=r["actual_output"],
                tools_called=r.get("tools_called"),
            )
            for r in self._records
        ]

    def clear(self) -> None:
        self._records.clear()


def evaluate_chain(chain, inputs: List[str], metrics: List[Any]) -> Any:
    """
    Convenience function to evaluate a LangChain chain.

    Args:
        chain: Any LangChain runnable (chain, agent, etc.)
        inputs: List of input strings to test.
        metrics: List of llmgrader metrics.

    Returns:
        EvaluationResult from evaluate().
    """
    from ..evaluate import evaluate

    handler = LangChainCallbackHandler()
    test_cases = []

    for inp in inputs:
        try:
            output = chain.invoke(inp, config={"callbacks": [handler]})
            actual = output if isinstance(output, str) else str(output)
        except Exception as e:
            actual = f"ERROR: {e}"
        test_cases.append(LLMTestCase(input=inp, actual_output=actual))

    return evaluate(test_cases, metrics)
