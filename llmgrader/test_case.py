"""Core test case classes for LLM evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMTestCaseParams(str, Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    REASONING = "reasoning"
    LATENCY = "latency"
    COST = "cost"


@dataclass
class ToolCall:
    """Represents a single tool/function call made by an LLM."""

    name: str
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Any] = None


@dataclass
class LLMTestCase:
    """
    A single-turn LLM evaluation test case.

    Required fields: input, actual_output
    Optional fields: expected_output, context, retrieval_context,
                     tools_called, expected_tools, reasoning, latency, cost
    """

    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
    expected_tools: Optional[List[str]] = None
    reasoning: Optional[str] = None
    latency: Optional[float] = None
    cost: Optional[float] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("LLMTestCase.input cannot be empty.")
        if self.actual_output is None:
            raise ValueError("LLMTestCase.actual_output cannot be None.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "tools_called": [
                {"name": t.name, "input_parameters": t.input_parameters, "output": t.output}
                for t in (self.tools_called or [])
            ],
            "expected_tools": self.expected_tools,
            "reasoning": self.reasoning,
            "latency": self.latency,
            "cost": self.cost,
            "additional_metadata": self.additional_metadata,
        }


@dataclass
class Message:
    """A single message turn in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
    llm_test_case: Optional[LLMTestCase] = None


@dataclass
class ConversationalTestCase:
    """
    A multi-turn conversational test case for evaluating chatbots and dialogue systems.

    messages: ordered list of Message objects representing the dialogue
    """

    messages: List[Message]
    chatbot_role: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ConversationalTestCase.messages cannot be empty.")

    @property
    def turns(self) -> int:
        return len(self.messages)

    def get_turns_with_test_cases(self) -> List[Message]:
        return [m for m in self.messages if m.llm_test_case is not None]


@dataclass
class Golden:
    """
    A template for generating LLMTestCase instances (stored in datasets).
    Does not require actual_output — filled in at evaluation time.
    """

    input: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None

    def to_test_case(self, actual_output: str, **kwargs) -> LLMTestCase:
        return LLMTestCase(
            input=self.input,
            actual_output=actual_output,
            expected_output=self.expected_output,
            context=self.context,
            retrieval_context=self.retrieval_context,
            expected_tools=self.expected_tools,
            additional_metadata={**self.additional_metadata, **kwargs},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "expected_tools": self.expected_tools,
            "additional_metadata": self.additional_metadata,
            "source_file": self.source_file,
        }


@dataclass
class ConversationalGolden:
    """Template for a multi-turn conversation (stored in datasets)."""

    messages: List[Message]
    chatbot_role: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
