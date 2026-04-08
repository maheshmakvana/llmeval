"""LLM Benchmarks — standard benchmarks for evaluating LLM capabilities.

Available benchmarks:
- MMLUBenchmark: Massive Multitask Language Understanding (57 subjects)
- HellaSwagBenchmark: Commonsense NLI (physical situation grounding)
- GSM8KBenchmark: Grade School Math word problems

Quick start:
    from llmgrader.benchmarks import MMLUBenchmark, GSM8KBenchmark
    from llmgrader.providers import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o")

    # Run MMLU (uses built-in samples without data_path)
    benchmark = MMLUBenchmark(model=provider, max_samples=20)
    result = benchmark.run()
    print(result.summary())

    # Run GSM8K with your own data
    benchmark = GSM8KBenchmark(model=provider, data_path="gsm8k/test.jsonl")
    result = benchmark.run()
    print(f"Accuracy: {result.accuracy:.1%}")
"""

from .base import BaseBenchmark, BenchmarkResult, BenchmarkSample
from .mmlu import MMLUBenchmark
from .hellaswag import HellaSwagBenchmark
from .gsm8k import GSM8KBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkSample",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "GSM8KBenchmark",
]
