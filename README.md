# llmgrader

![llmgrader banner](docs/images/banner.jpg)

**Open-source LLM evaluation framework** — 50+ research-backed metrics for RAG pipelines, AI agents, safety, and conversational systems. Pytest-native. Provider-agnostic.

```bash
pip install llmgrader
```

---

## Quick Start

```python
from llmgrader import LLMTestCase, assert_test
from llmgrader.metrics import AnswerRelevancyMetric, FaithfulnessMetric

tc = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    retrieval_context=["France is a country in Western Europe. Its capital is Paris."],
)

assert_test(tc, metrics=[
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.8),
])
```

---

## Features

| Category | What it does |
|---|---|
| **RAG Evaluation** | Answer relevancy, faithfulness, contextual precision/recall/relevancy |
| **Custom Metrics** | GEval (LLM-as-judge + CoT), DAG (deterministic decision-tree) |
| **Safety** | Hallucination, bias, toxicity, PII leakage, misuse detection |
| **Agent Evaluation** | Task completion, tool correctness, step efficiency, argument correctness |
| **Conversational** | Relevancy, completeness, role adherence, knowledge retention |
| **Other** | JSON correctness (with schema), summarization quality |
| **Pytest Native** | `assert_test()`, fixtures, parametrize helpers |
| **Tracing** | `@observe` decorator, span/trace trees, component-level evaluation |
| **Bulk Evaluation** | `evaluate()` with concurrent execution and aggregated reports |
| **Dataset Tools** | `EvaluationDataset`, versioned JSON storage, CSV import |
| **Synthesizer** | Auto-generate Goldens from documents (4-step pipeline) |
| **Providers** | OpenAI, Azure OpenAI, Anthropic, Ollama, custom LLM base class |
| **Integrations** | LangChain, LlamaIndex, CrewAI |
| **CLI** | `llmgrader test`, `llmgrader set-openai`, `llmgrader list-metrics` |

---

## Installation

```bash
# Core
pip install llmgrader

# With LangChain integration
pip install "llmgrader[langchain]"

# With LlamaIndex
pip install "llmgrader[llamaindex]"

# Everything
pip install "llmgrader[all]"
```

---

## Metrics Reference

### RAG Metrics

![RAG evaluation](docs/images/rag.jpg)

```python
from llmgrader.metrics import (
    AnswerRelevancyMetric,       # Is the answer relevant to the question?
    FaithfulnessMetric,          # Are claims grounded in retrieved context?
    ContextualRelevancyMetric,   # Are retrieved chunks relevant to the query?
    ContextualPrecisionMetric,   # Do relevant chunks rank higher?
    ContextualRecallMetric,      # Does context cover expected answer claims?
)

tc = LLMTestCase(
    input="What causes rain?",
    actual_output="Rain is caused by water vapor condensing in clouds.",
    expected_output="Rain is caused by condensation of water vapor.",
    retrieval_context=[
        "The water cycle involves evaporation and condensation.",
        "Rain forms when water vapor cools and condenses around particles.",
    ],
)

result = FaithfulnessMetric(threshold=0.8).measure(tc)
print(result.score, result.reason)
```

### Custom: GEval (LLM-as-Judge)

```python
from llmgrader import GEvalMetric, LLMTestCaseParams

metric = GEvalMetric(
    name="Correctness",
    criteria="The output should be factually correct and directly answer the question.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

result = metric.measure(tc)
```

### Custom: DAG (Deterministic)

```python
from llmgrader import DAGMetric
from llmgrader.metrics.custom.dag import DAGNode

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
```

### Safety Metrics

![Safety metrics](docs/images/safety.jpg)

```python
from llmgrader.metrics import (
    HallucinationMetric,   # Detects factual hallucinations vs context
    BiasMetric,            # Gender, racial, political, religious bias
    ToxicityMetric,        # Hate speech, harassment, harmful content
    PIILeakageMetric,      # SSN, email, phone, credit card detection
    MisuseMetric,          # Weapons, illegal activity enablement
)

result = BiasMetric(threshold=0.7).measure(tc)
```

### Agentic Metrics

![Agentic evaluation](docs/images/agentic.jpg)

```python
from llmgrader import ToolCall
from llmgrader.metrics import (
    TaskCompletionMetric,      # Did the agent accomplish the goal?
    ToolCorrectnessMetric,     # Were the right tools called?
    StepEfficiencyMetric,      # Were unnecessary steps avoided?
    ArgumentCorrectnessMetric, # Were tool arguments correct?
)

tc = LLMTestCase(
    input="Search for the latest news on AI and summarize it.",
    actual_output="Here is a summary of recent AI news...",
    tools_called=[
        ToolCall(name="web_search", input_parameters={"query": "latest AI news"}),
        ToolCall(name="summarize", input_parameters={"max_length": 200}),
    ],
    expected_tools=["web_search", "summarize"],
)

result = ToolCorrectnessMetric(threshold=0.8).measure(tc)
```

### Conversational Metrics

```python
from llmgrader import ConversationalTestCase, Message
from llmgrader.metrics import (
    ConversationalRelevancyMetric,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
)

tc = ConversationalTestCase(
    messages=[
        Message(role="user", content="My name is Alice and I like Python."),
        Message(role="assistant", content="Nice to meet you, Alice! Python is great."),
        Message(role="user", content="What's my name again?"),
        Message(role="assistant", content="Your name is Alice."),
    ],
    chatbot_role="A helpful assistant that remembers user preferences.",
)

result = KnowledgeRetentionMetric(threshold=0.7).measure(tc)
```

---

## Bulk Evaluation

```python
from llmgrader import evaluate, LLMTestCase
from llmgrader.metrics import AnswerRelevancyMetric, JSONCorrectnessMetric

test_cases = [
    LLMTestCase(input="What is 2+2?", actual_output="4"),
    LLMTestCase(input="Capital of Japan?", actual_output="Tokyo"),
    LLMTestCase(input="Return JSON", actual_output='{"status": "ok"}'),
]

result = evaluate(
    test_cases=test_cases,
    metrics=[AnswerRelevancyMetric(), JSONCorrectnessMetric()],
    max_concurrent=4,
    verbose=True,
)

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Overall score: {result.overall_score:.3f}")
result.print_summary()
```

---

## Pytest Integration

```python
# test_my_llm.py
import pytest
from llmgrader import LLMTestCase, assert_test
from llmgrader.metrics import AnswerRelevancyMetric, FaithfulnessMetric

def test_rag_answer():
    tc = LLMTestCase(
        input="What causes lightning?",
        actual_output=my_rag_pipeline("What causes lightning?"),
        retrieval_context=get_context("lightning"),
    )
    assert_test(tc, metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.8),
    ])


# Run with: llmgrader test test_my_llm.py
# Or:       pytest test_my_llm.py
```

---

## Tracing & Component-Level Evaluation

![Tracing](docs/images/tracing.jpg)

```python
from llmgrader import observe, Tracer, set_tracer, clear_tracer

tracer = Tracer()
set_tracer(tracer)
trace = tracer.start_trace()

@observe(span_type="retriever")
def retrieve(query: str) -> list:
    return vector_db.search(query)

@observe(span_type="llm")
def generate(context: list, query: str) -> str:
    return llm.generate(f"Context: {context}\nQuestion: {query}")

def rag_pipeline(query: str) -> str:
    context = retrieve(query)
    return generate(context, query)

answer = rag_pipeline("What is quantum computing?")
tracer.end_trace()
clear_tracer()

tracer.print_last_trace()  # Shows span tree with latencies
```

---

## Dataset Management

```python
from llmgrader import EvaluationDataset, Golden

# Build a dataset
ds = EvaluationDataset()
ds.add_goldens([
    Golden(input="What is AI?", expected_output="Artificial intelligence."),
    Golden(input="Capital of Germany?", expected_output="Berlin"),
])
ds.save("my_dataset.json")

# Load and use
ds = EvaluationDataset.load("my_dataset.json")
test_cases = ds.to_test_cases(generate_fn=my_llm.generate)
```

---

## Synthetic Dataset Generation

```python
from llmgrader import Synthesizer

synth = Synthesizer()

docs = [
    "The Python programming language was created by Guido van Rossum...",
    "Machine learning is a branch of artificial intelligence...",
]

goldens = synth.generate_goldens_from_docs(
    documents=docs,
    max_goldens_per_doc=5,
    filter_questions=True,
    evolve_questions=True,
    generate_expected_outputs=True,
)

print(f"Generated {len(goldens)} golden test cases")
for g in goldens[:2]:
    print(f"Q: {g.input}")
    print(f"A: {g.expected_output}\n")
```

---

## LLM Providers

```python
from llmgrader.providers import OpenAIProvider, AnthropicProvider, OllamaProvider

# OpenAI
provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")

# Anthropic Claude
provider = AnthropicProvider(model="claude-sonnet-4-6")

# Ollama (local)
provider = OllamaProvider(model="llama3")

# Custom provider
from llmgrader.providers import LLMProvider

class MyProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return my_llm_api.call(prompt)

# Use in any metric
metric = AnswerRelevancyMetric(model=MyProvider())
```

---

## LangChain Integration

```python
from langchain_openai import ChatOpenAI
from llmgrader.integrations.langchain import LangChainCallbackHandler, evaluate_chain
from llmgrader.metrics import AnswerRelevancyMetric

llm = ChatOpenAI(model="gpt-4o")
chain = llm  # or any LangChain runnable

result = evaluate_chain(
    chain=chain,
    inputs=["What is the capital of France?", "Who invented Python?"],
    metrics=[AnswerRelevancyMetric(threshold=0.7)],
)
```

---

## CLI

```bash
# Run evaluation tests
llmgrader test tests/test_llm.py
llmgrader test tests/ -n 4          # 4 parallel workers

# Configure providers
llmgrader set-openai --key sk-... --model gpt-4o
llmgrader set-anthropic --key sk-... --model claude-sonnet-4-6
llmgrader set-ollama --model llama3

# List all metrics
llmgrader list-metrics

# Version
llmgrader version
```

## Changelog

### v1.2.0 (2026-04-10)
- Added Changelog section to README for release traceability
- Added RegressionTracker, ScoreTrend, EvaluationReport, CustomBenchmarkBuilder, EvaluationFilter, MetricWeightedScorer, evaluate_async

### v1.1.0
- Added 17 new metrics, multimodal support, benchmarks module

### v1.0.1
- Renamed package from `llmeval` to `llmgrader`

### v1.0.0
- Initial release: 50+ LLM evaluation metrics, RAG, safety, agentic, conversational

---

## License

Apache 2.0 — see LICENSE.
