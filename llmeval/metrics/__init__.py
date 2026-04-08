"""All metrics — RAG, custom, safety, agentic, conversational, and more."""

from .base import BaseMetric, MetricResult
from .rag.answer_relevancy import AnswerRelevancyMetric
from .rag.faithfulness import FaithfulnessMetric
from .rag.contextual_relevancy import ContextualRelevancyMetric
from .rag.contextual_precision import ContextualPrecisionMetric
from .rag.contextual_recall import ContextualRecallMetric
from .custom.geval import GEvalMetric
from .custom.dag import DAGMetric
from .safety.hallucination import HallucinationMetric
from .safety.bias import BiasMetric
from .safety.toxicity import ToxicityMetric
from .safety.pii_leakage import PIILeakageMetric
from .safety.misuse import MisuseMetric
from .agentic.task_completion import TaskCompletionMetric
from .agentic.tool_correctness import ToolCorrectnessMetric
from .agentic.step_efficiency import StepEfficiencyMetric
from .agentic.argument_correctness import ArgumentCorrectnessMetric
from .conversational.relevancy import ConversationalRelevancyMetric
from .conversational.completeness import ConversationCompletenessMetric
from .conversational.role_adherence import RoleAdherenceMetric
from .conversational.knowledge_retention import KnowledgeRetentionMetric
from .other.json_correctness import JSONCorrectnessMetric
from .other.summarization import SummarizationMetric

__all__ = [
    "BaseMetric",
    "MetricResult",
    # RAG
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    # Custom
    "GEvalMetric",
    "DAGMetric",
    # Safety
    "HallucinationMetric",
    "BiasMetric",
    "ToxicityMetric",
    "PIILeakageMetric",
    "MisuseMetric",
    # Agentic
    "TaskCompletionMetric",
    "ToolCorrectnessMetric",
    "StepEfficiencyMetric",
    "ArgumentCorrectnessMetric",
    # Conversational
    "ConversationalRelevancyMetric",
    "ConversationCompletenessMetric",
    "RoleAdherenceMetric",
    "KnowledgeRetentionMetric",
    # Other
    "JSONCorrectnessMetric",
    "SummarizationMetric",
]
