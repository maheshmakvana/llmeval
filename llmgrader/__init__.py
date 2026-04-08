"""
llmgrader — Open-source LLM evaluation framework.

50+ research-backed metrics for RAG, agents, safety, and more.
Pytest-native, provider-agnostic, extensible.

Quick start:
    from llmgrader import LLMTestCase, assert_test
    from llmgrader.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    tc = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=["France is a country in Europe. Its capital city is Paris."],
    )
    assert_test(tc, metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.8),
    ])
"""

__version__ = "1.0.0"
__author__ = "Mahesh Makvana"

# Core
from .test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    Message,
    ToolCall,
    Golden,
    ConversationalGolden,
)

# Dataset
from .dataset import EvaluationDataset

# Evaluation engine
from .evaluate import evaluate, assert_test, EvaluationResult, TestResult

# Synthesizer
from .synthesizer import Synthesizer

# Tracing
from .tracing import observe, Tracer, Span, Trace, get_current_tracer
from .tracing.tracer import set_tracer, clear_tracer

# Providers
from .providers import (
    LLMProvider,
    EmbeddingProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)

# All metrics
from .metrics import (
    BaseMetric,
    MetricResult,
    # RAG
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    # Custom
    GEvalMetric,
    DAGMetric,
    # Safety
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    PIILeakageMetric,
    MisuseMetric,
    # Agentic
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    StepEfficiencyMetric,
    ArgumentCorrectnessMetric,
    # Conversational
    ConversationalRelevancyMetric,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    # Other
    JSONCorrectnessMetric,
    SummarizationMetric,
)

__all__ = [
    "__version__",
    # Core
    "LLMTestCase", "LLMTestCaseParams", "ConversationalTestCase",
    "Message", "ToolCall", "Golden", "ConversationalGolden",
    # Dataset
    "EvaluationDataset",
    # Evaluation
    "evaluate", "assert_test", "EvaluationResult", "TestResult",
    # Synthesizer
    "Synthesizer",
    # Tracing
    "observe", "Tracer", "Span", "Trace", "get_current_tracer", "set_tracer", "clear_tracer",
    # Providers
    "LLMProvider", "EmbeddingProvider", "OpenAIProvider", "AnthropicProvider", "OllamaProvider",
    # Metrics
    "BaseMetric", "MetricResult",
    "AnswerRelevancyMetric", "FaithfulnessMetric", "ContextualRelevancyMetric",
    "ContextualPrecisionMetric", "ContextualRecallMetric",
    "GEvalMetric", "DAGMetric",
    "HallucinationMetric", "BiasMetric", "ToxicityMetric", "PIILeakageMetric", "MisuseMetric",
    "TaskCompletionMetric", "ToolCorrectnessMetric", "StepEfficiencyMetric", "ArgumentCorrectnessMetric",
    "ConversationalRelevancyMetric", "ConversationCompletenessMetric", "RoleAdherenceMetric", "KnowledgeRetentionMetric",
    "JSONCorrectnessMetric", "SummarizationMetric",
]
