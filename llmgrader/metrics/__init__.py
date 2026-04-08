"""All metrics — RAG, custom, safety, agentic, conversational, multimodal, and more."""

from .base import BaseMetric, MetricResult

# RAG
from .rag.answer_relevancy import AnswerRelevancyMetric
from .rag.faithfulness import FaithfulnessMetric
from .rag.contextual_relevancy import ContextualRelevancyMetric
from .rag.contextual_precision import ContextualPrecisionMetric
from .rag.contextual_recall import ContextualRecallMetric

# Custom
from .custom.geval import GEvalMetric
from .custom.dag import DAGMetric

# Safety
from .safety.hallucination import HallucinationMetric
from .safety.bias import BiasMetric
from .safety.toxicity import ToxicityMetric
from .safety.pii_leakage import PIILeakageMetric
from .safety.misuse import MisuseMetric
from .safety.non_advice import NonAdviceMetric
from .safety.role_violation import RoleViolationMetric

# Agentic
from .agentic.task_completion import TaskCompletionMetric
from .agentic.tool_correctness import ToolCorrectnessMetric
from .agentic.step_efficiency import StepEfficiencyMetric
from .agentic.argument_correctness import ArgumentCorrectnessMetric
from .agentic.goal_accuracy import GoalAccuracyMetric
from .agentic.plan_adherence import PlanAdherenceMetric
from .agentic.plan_quality import PlanQualityMetric

# Conversational
from .conversational.relevancy import ConversationalRelevancyMetric
from .conversational.completeness import ConversationCompletenessMetric
from .conversational.role_adherence import RoleAdherenceMetric
from .conversational.knowledge_retention import KnowledgeRetentionMetric
from .conversational.turn_relevancy import TurnRelevancyMetric
from .conversational.turn_faithfulness import TurnFaithfulnessMetric
from .conversational.turn_contextual_metrics import (
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
)
from .conversational.conversational_geval import ConversationalGEvalMetric
from .conversational.conversational_dag import ConversationalDAGMetric, ConversationalDAGNode

# Other
from .other.json_correctness import JSONCorrectnessMetric
from .other.summarization import SummarizationMetric
from .other.exact_match import ExactMatchMetric
from .other.pattern_match import PatternMatchMetric
from .other.prompt_alignment import PromptAlignmentMetric
from .other.topic_adherence import TopicAdherenceMetric
from .other.arena_geval import ArenaGEvalMetric

# Multimodal
from .multimodal.text_to_image import TextToImageMetric
from .multimodal.image_editing import ImageEditingMetric
from .multimodal.image_coherence import ImageCoherenceMetric
from .multimodal.image_helpfulness import ImageHelpfulnessMetric
from .multimodal.image_reference import ImageReferenceMetric

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
    "NonAdviceMetric",
    "RoleViolationMetric",
    # Agentic
    "TaskCompletionMetric",
    "ToolCorrectnessMetric",
    "StepEfficiencyMetric",
    "ArgumentCorrectnessMetric",
    "GoalAccuracyMetric",
    "PlanAdherenceMetric",
    "PlanQualityMetric",
    # Conversational
    "ConversationalRelevancyMetric",
    "ConversationCompletenessMetric",
    "RoleAdherenceMetric",
    "KnowledgeRetentionMetric",
    "TurnRelevancyMetric",
    "TurnFaithfulnessMetric",
    "TurnContextualPrecisionMetric",
    "TurnContextualRecallMetric",
    "TurnContextualRelevancyMetric",
    "ConversationalGEvalMetric",
    "ConversationalDAGMetric",
    "ConversationalDAGNode",
    # Other
    "JSONCorrectnessMetric",
    "SummarizationMetric",
    "ExactMatchMetric",
    "PatternMatchMetric",
    "PromptAlignmentMetric",
    "TopicAdherenceMetric",
    "ArenaGEvalMetric",
    # Multimodal
    "TextToImageMetric",
    "ImageEditingMetric",
    "ImageCoherenceMetric",
    "ImageHelpfulnessMetric",
    "ImageReferenceMetric",
]
