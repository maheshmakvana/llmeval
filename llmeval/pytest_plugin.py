"""pytest plugin for llmeval — enables @pytest.mark.llmeval and llmeval_test fixtures."""

from __future__ import annotations

from typing import List, Union

import pytest

from .evaluate import assert_test
from .metrics.base import BaseMetric
from .test_case import ConversationalTestCase, LLMTestCase


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "llmeval: mark test as an llmeval LLM evaluation test",
    )


@pytest.fixture
def llmeval_assert():
    """
    Fixture that provides the assert_test function for use in pytest.

    Example:
        def test_answer_quality(llmeval_assert):
            tc = LLMTestCase(input="What is 2+2?", actual_output="4")
            llmeval_assert(tc, metrics=[AnswerRelevancyMetric(threshold=0.7)])
    """
    return assert_test


@pytest.fixture
def make_test_case():
    """
    Fixture factory for creating LLMTestCase instances.

    Example:
        def test_my_llm(make_test_case):
            tc = make_test_case(input="Hello", actual_output="Hi there!")
            ...
    """
    def factory(
        input: str,
        actual_output: str,
        expected_output: str = None,
        context: list = None,
        retrieval_context: list = None,
        **kwargs,
    ) -> LLMTestCase:
        return LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            **kwargs,
        )
    return factory


def llmeval_parametrize(test_cases: List[dict]):
    """
    Parametrize decorator for running the same evaluation across multiple inputs.

    Example:
        @llmeval_parametrize([
            {"input": "What is 2+2?", "actual_output": "4"},
            {"input": "Capital of France?", "actual_output": "Paris"},
        ])
        def test_llm(tc_data):
            tc = LLMTestCase(**tc_data)
            assert_test(tc, metrics=[AnswerRelevancyMetric()])
    """
    return pytest.mark.parametrize("tc_data", test_cases)
