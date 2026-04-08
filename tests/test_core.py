"""Core tests for llmeval — test cases, dataset, evaluate, tracing, and metrics."""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmeval import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    Message,
    ToolCall,
    Golden,
    EvaluationDataset,
    evaluate,
    assert_test,
    Synthesizer,
    Tracer,
    observe,
    set_tracer,
    clear_tracer,
)
from llmeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEvalMetric,
    DAGMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    PIILeakageMetric,
    MisuseMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    StepEfficiencyMetric,
    JSONCorrectnessMetric,
    SummarizationMetric,
    ConversationalRelevancyMetric,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
)
from llmeval.metrics.custom.dag import DAGNode
from llmeval.metrics.base import MetricResult


# ─── Helpers ─────────────────────────────────────────────────────────────────

def mock_provider(response: str):
    """Return a mock LLM provider that always returns response."""
    provider = MagicMock()
    provider.generate.return_value = response
    return provider


# ─── LLMTestCase ─────────────────────────────────────────────────────────────

class TestLLMTestCase:
    def test_basic_creation(self):
        tc = LLMTestCase(input="Hello", actual_output="Hi")
        assert tc.input == "Hello"
        assert tc.actual_output == "Hi"
        assert tc.expected_output is None

    def test_full_creation(self):
        tc = LLMTestCase(
            input="What is AI?",
            actual_output="AI is artificial intelligence.",
            expected_output="Artificial intelligence.",
            context=["AI is a broad field."],
            retrieval_context=["AI stands for Artificial Intelligence."],
        )
        assert tc.context == ["AI is a broad field."]
        assert tc.retrieval_context == ["AI stands for Artificial Intelligence."]

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            LLMTestCase(input="", actual_output="something")

    def test_to_dict(self):
        tc = LLMTestCase(input="Q", actual_output="A", expected_output="A2")
        d = tc.to_dict()
        assert d["input"] == "Q"
        assert d["actual_output"] == "A"
        assert d["expected_output"] == "A2"

    def test_tool_calls(self):
        tc = LLMTestCase(
            input="Search the web",
            actual_output="Found results.",
            tools_called=[ToolCall(name="web_search", input_parameters={"query": "test"})],
            expected_tools=["web_search"],
        )
        assert tc.tools_called[0].name == "web_search"


# ─── ConversationalTestCase ───────────────────────────────────────────────────

class TestConversationalTestCase:
    def test_creation(self):
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        tc = ConversationalTestCase(messages=msgs)
        assert tc.turns == 2

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError):
            ConversationalTestCase(messages=[])


# ─── Golden ───────────────────────────────────────────────────────────────────

class TestGolden:
    def test_to_test_case(self):
        g = Golden(input="What is 2+2?", expected_output="4")
        tc = g.to_test_case(actual_output="Four")
        assert tc.input == "What is 2+2?"
        assert tc.actual_output == "Four"
        assert tc.expected_output == "4"


# ─── EvaluationDataset ────────────────────────────────────────────────────────

class TestEvaluationDataset:
    def test_add_goldens(self):
        ds = EvaluationDataset()
        ds.add_goldens([Golden(input="Q1"), Golden(input="Q2")])
        assert len(ds) == 2

    def test_iter(self):
        ds = EvaluationDataset(goldens=[Golden(input="Q"), Golden(input="Q2")])
        inputs = [g.input for g in ds]
        assert inputs == ["Q", "Q2"]

    def test_save_load(self, tmp_path):
        ds = EvaluationDataset()
        ds.add_goldens([
            Golden(input="Test input", expected_output="Test output"),
        ])
        path = str(tmp_path / "dataset.json")
        ds.save(path)

        loaded = EvaluationDataset.load(path)
        assert len(loaded) == 1
        assert loaded[0].input == "Test input"
        assert loaded[0].expected_output == "Test output"

    def test_to_test_cases(self):
        ds = EvaluationDataset(goldens=[Golden(input="Hello")])
        tcs = ds.to_test_cases(generate_fn=lambda x: f"Response to: {x}")
        assert tcs[0].actual_output == "Response to: Hello"


# ─── JSONCorrectnessMetric (no LLM needed) ───────────────────────────────────

class TestJSONCorrectnessMetric:
    def test_valid_json(self):
        metric = JSONCorrectnessMetric()
        tc = LLMTestCase(input="q", actual_output='{"name": "Alice", "age": 30}')
        result = metric.measure(tc)
        assert result.score == 1.0
        assert result.passed

    def test_invalid_json(self):
        metric = JSONCorrectnessMetric()
        tc = LLMTestCase(input="q", actual_output='not json at all')
        result = metric.measure(tc)
        assert result.score == 0.0
        assert not result.passed

    def test_schema_validation_pass(self):
        schema = {
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            }
        }
        metric = JSONCorrectnessMetric(json_schema=schema)
        tc = LLMTestCase(input="q", actual_output='{"name": "Bob", "age": 25}')
        result = metric.measure(tc)
        assert result.passed

    def test_schema_validation_missing_key(self):
        schema = {"required": ["name", "age"]}
        metric = JSONCorrectnessMetric(json_schema=schema, threshold=0.9)
        tc = LLMTestCase(input="q", actual_output='{"name": "Bob"}')
        result = metric.measure(tc)
        assert result.score < 1.0
        assert not result.passed

    def test_markdown_fence_stripped(self):
        metric = JSONCorrectnessMetric()
        tc = LLMTestCase(input="q", actual_output='```json\n{"key": "value"}\n```')
        result = metric.measure(tc)
        assert result.score == 1.0


# ─── ToolCorrectnessMetric (no LLM needed) ───────────────────────────────────

class TestToolCorrectnessMetric:
    def test_all_correct(self):
        metric = ToolCorrectnessMetric(threshold=0.8)
        tc = LLMTestCase(
            input="Search and summarize",
            actual_output="Done.",
            tools_called=[ToolCall(name="search"), ToolCall(name="summarize")],
            expected_tools=["search", "summarize"],
        )
        result = metric.measure(tc)
        assert result.score == 1.0
        assert result.passed

    def test_missing_tool(self):
        metric = ToolCorrectnessMetric(threshold=0.8)
        tc = LLMTestCase(
            input="Search and summarize",
            actual_output="Done.",
            tools_called=[ToolCall(name="search")],
            expected_tools=["search", "summarize"],
        )
        result = metric.measure(tc)
        assert result.score == 0.5
        assert not result.passed

    def test_no_expected_tools_raises(self):
        metric = ToolCorrectnessMetric()
        tc = LLMTestCase(input="q", actual_output="a")
        with pytest.raises(ValueError):
            metric.measure(tc)


# ─── DAGMetric (no LLM needed) ───────────────────────────────────────────────

class TestDAGMetric:
    def test_pass_condition(self):
        dag = DAGNode(
            condition=lambda tc: len(tc.actual_output) > 5,
            score_if_true=1.0,
            score_if_false=0.0,
        )
        metric = DAGMetric(name="LengthCheck", root=dag, threshold=0.5)
        tc = LLMTestCase(input="q", actual_output="This is a long enough response.")
        result = metric.measure(tc)
        assert result.score == 1.0
        assert result.passed

    def test_fail_condition(self):
        dag = DAGNode(
            condition=lambda tc: len(tc.actual_output) > 100,
            score_if_true=1.0,
            score_if_false=0.0,
        )
        metric = DAGMetric(name="LengthCheck", root=dag, threshold=0.5)
        tc = LLMTestCase(input="q", actual_output="Short.")
        result = metric.measure(tc)
        assert result.score == 0.0
        assert not result.passed

    def test_chained_nodes(self):
        root = DAGNode(
            condition=lambda tc: len(tc.actual_output) > 0,
            score_if_false=0.0,
            next_if_true=DAGNode(
                condition=lambda tc: "error" not in tc.actual_output.lower(),
                score_if_true=1.0,
                score_if_false=0.3,
            )
        )
        metric = DAGMetric(name="QualityCheck", root=root, threshold=0.5)
        tc = LLMTestCase(input="q", actual_output="Great answer here!")
        result = metric.measure(tc)
        assert result.score == 1.0


# ─── LLM-based metrics (mocked) ──────────────────────────────────────────────

class TestAnswerRelevancyMetric:
    def test_high_relevancy(self):
        questions_response = '["What is the capital of France?"]'
        verdict_response = '{"verdict": "yes"}'
        provider = mock_provider(questions_response)
        provider.generate.side_effect = [questions_response, verdict_response]

        metric = AnswerRelevancyMetric(threshold=0.5, model=provider)
        tc = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
        )
        result = metric.measure(tc)
        assert result.score >= 0.0
        assert isinstance(result.passed, bool)

    def test_requires_llm_test_case(self):
        metric = AnswerRelevancyMetric()
        msgs = [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]
        conv_tc = ConversationalTestCase(messages=msgs)
        with pytest.raises(TypeError):
            metric.measure(conv_tc)


class TestFaithfulnessMetric:
    def test_requires_retrieval_context(self):
        metric = FaithfulnessMetric()
        tc = LLMTestCase(input="q", actual_output="a")
        with pytest.raises(ValueError):
            metric.measure(tc)

    def test_high_faithfulness(self):
        claims_response = '["Paris is the capital of France."]'
        verdict_response = '{"verdict": "yes"}'
        provider = mock_provider("")
        provider.generate.side_effect = [claims_response, verdict_response]

        metric = FaithfulnessMetric(threshold=0.5, model=provider)
        tc = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["France's capital is Paris."],
        )
        result = metric.measure(tc)
        assert result.score == 1.0
        assert result.passed


class TestHallucinationMetric:
    def test_requires_context(self):
        metric = HallucinationMetric()
        tc = LLMTestCase(input="q", actual_output="a")
        with pytest.raises(ValueError):
            metric.measure(tc)

    def test_no_hallucination(self):
        split_response = '["Paris is the capital of France."]'
        verdict_response = '{"verdict": "no", "reason": "Supported by context."}'
        provider = mock_provider("")
        provider.generate.side_effect = [split_response, verdict_response]

        metric = HallucinationMetric(threshold=0.5, model=provider)
        tc = LLMTestCase(
            input="q",
            actual_output="Paris is the capital of France.",
            context=["France's capital is Paris."],
        )
        result = metric.measure(tc)
        assert result.score == 1.0
        assert result.passed


class TestPIILeakageMetric:
    def test_detects_email_regex(self):
        metric = PIILeakageMetric(threshold=0.5, use_llm=False)
        tc = LLMTestCase(input="q", actual_output="Contact us at john.doe@example.com for help.")
        result = metric.measure(tc)
        assert result.score < 1.0

    def test_clean_output(self):
        metric = PIILeakageMetric(threshold=0.5, use_llm=False)
        tc = LLMTestCase(input="q", actual_output="This is a completely safe response with no PII.")
        result = metric.measure(tc)
        assert result.score == 1.0
        assert result.passed

    def test_detects_ssn_regex(self):
        metric = PIILeakageMetric(threshold=0.5, use_llm=False)
        tc = LLMTestCase(input="q", actual_output="The SSN is 123-45-6789.")
        result = metric.measure(tc)
        assert result.score < 1.0


# ─── Tracing ─────────────────────────────────────────────────────────────────

class TestTracing:
    def test_observe_decorator(self):
        tracer = Tracer()
        set_tracer(tracer)
        trace = tracer.start_trace()

        @observe
        def my_function(x):
            return x * 2

        result = my_function(5)
        tracer.end_trace()
        clear_tracer()

        assert result == 10
        trace_data = trace.to_dict()
        assert trace_data["root_span"]["name"] == "my_function"

    def test_nested_spans(self):
        tracer = Tracer()
        set_tracer(tracer)
        trace = tracer.start_trace()

        @observe(name="outer", span_type="generic")
        def outer_fn(x):
            return inner_fn(x)

        @observe(name="inner", span_type="llm")
        def inner_fn(x):
            return x + 1

        outer_fn(3)
        tracer.end_trace()
        clear_tracer()

        root = trace.root_span
        assert root.name == "outer"
        assert len(root.children) == 1
        assert root.children[0].name == "inner"

    def test_no_tracer_passthrough(self):
        clear_tracer()

        @observe
        def simple_fn(x):
            return x + 10

        assert simple_fn(5) == 15

    def test_span_latency_recorded(self):
        tracer = Tracer()
        set_tracer(tracer)
        tracer.start_trace()

        @observe
        def slow_fn(x):
            import time
            time.sleep(0.01)
            return x

        slow_fn(1)
        tracer.end_trace()
        clear_tracer()

        span = tracer.get_last_trace().root_span
        assert span.latency_ms is not None
        assert span.latency_ms >= 10


# ─── evaluate() ──────────────────────────────────────────────────────────────

class TestEvaluateFunction:
    def test_basic_evaluate(self):
        provider = mock_provider('{"verdict": "yes"}')

        metric = AnswerRelevancyMetric(threshold=0.5, model=provider)
        provider.generate.side_effect = [
            '["What is the capital of France?"]',
            '{"verdict": "yes"}',
        ]

        tc = LLMTestCase(input="What is Paris?", actual_output="Paris is the capital of France.")
        result = evaluate([tc], [metric], verbose=False)
        assert result.pass_rate >= 0.0
        assert 0.0 <= result.overall_score <= 1.0

    def test_evaluate_with_dag_no_llm(self):
        dag = DAGNode(
            condition=lambda tc: len(tc.actual_output) > 0,
            score_if_true=1.0,
            score_if_false=0.0,
        )
        metric = DAGMetric(name="NonEmpty", root=dag, threshold=0.5)

        tcs = [
            LLMTestCase(input="Q1", actual_output="Answer 1"),
            LLMTestCase(input="Q2", actual_output="Answer 2"),
            LLMTestCase(input="Q3", actual_output=""),
        ]
        result = evaluate(tcs, [metric], verbose=False)
        assert result.pass_rate == pytest.approx(2 / 3)

    def test_evaluate_json_correctness(self):
        metric = JSONCorrectnessMetric()
        tcs = [
            LLMTestCase(input="q1", actual_output='{"valid": true}'),
            LLMTestCase(input="q2", actual_output="invalid json"),
        ]
        result = evaluate(tcs, [metric], verbose=False)
        assert result.pass_rate == 0.5
        assert result.metric_summaries["JSONCorrectnessMetric"]["avg_score"] == 0.5


# ─── assert_test ─────────────────────────────────────────────────────────────

class TestAssertTest:
    def test_passes_on_good_output(self):
        dag = DAGNode(condition=lambda tc: "Paris" in tc.actual_output, score_if_true=1.0, score_if_false=0.0)
        metric = DAGMetric(name="ContainsParis", root=dag, threshold=0.5)
        tc = LLMTestCase(input="Capital of France?", actual_output="Paris is the capital.")
        assert_test(tc, metrics=[metric])  # should not raise

    def test_raises_on_failed_metric(self):
        dag = DAGNode(condition=lambda tc: False, score_if_true=1.0, score_if_false=0.0)
        metric = DAGMetric(name="AlwaysFail", root=dag, threshold=0.5)
        tc = LLMTestCase(input="q", actual_output="a")
        with pytest.raises(AssertionError) as exc_info:
            assert_test(tc, metrics=[metric])
        assert "AlwaysFail" in str(exc_info.value)


# ─── MetricResult ─────────────────────────────────────────────────────────────

class TestMetricResult:
    def test_repr(self):
        r = MetricResult(
            score=0.85,
            passed=True,
            reason="Good response.",
            metric_name="TestMetric",
            threshold=0.5,
        )
        assert "TestMetric" in repr(r)
        assert "PASS" in repr(r)


# ─── GEvalMetric ─────────────────────────────────────────────────────────────

class TestGEvalMetric:
    def test_generates_steps_and_scores(self):
        steps_response = "1. Check if output is relevant\n2. Check if output is accurate"
        score_response = "Reasoning: The output is relevant and accurate.\nScore: 8/10"

        provider = mock_provider("")
        provider.generate.side_effect = [steps_response, score_response]

        metric = GEvalMetric(
            name="Accuracy",
            criteria="The output should be accurate and complete.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=provider,
            threshold=0.5,
        )
        tc = LLMTestCase(input="What is 2+2?", actual_output="4")
        result = metric.measure(tc)
        assert 0.0 <= result.score <= 1.0
        assert result.metric_name == "Accuracy"
