"""BaseBenchmark — abstract base class for all LLM benchmarks."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BenchmarkSample:
    """A single sample from a benchmark dataset."""

    id: str
    question: str
    choices: Optional[List[str]] = None  # For multiple-choice benchmarks
    correct_answer: Optional[str] = None  # Ground truth answer
    subject: Optional[str] = None  # Category/subject (e.g., MMLU subject)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""

    benchmark_name: str
    total_samples: int
    correct: int
    accuracy: float
    subject_scores: Dict[str, float] = field(default_factory=dict)
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.benchmark_name}: accuracy={self.accuracy:.3f} "
            f"[{self.correct}/{self.total_samples}])"
        )

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Model: {self.model_name or 'Unknown'}",
            f"Accuracy: {self.accuracy:.1%} ({self.correct}/{self.total_samples})",
            f"Time: {self.elapsed_seconds:.1f}s",
        ]
        if self.subject_scores:
            lines.append("\nPer-subject scores:")
            for subj, score in sorted(self.subject_scores.items()):
                lines.append(f"  {subj}: {score:.1%}")
        return "\n".join(lines)


class BaseBenchmark(ABC):
    """
    Abstract base class for LLM benchmarks.

    Subclass and implement `load_samples()` and `_evaluate_sample()`.

    Example:
        benchmark = MMLUBenchmark(model=my_provider, max_samples=100)
        result = benchmark.run()
        print(result.summary())
    """

    def __init__(
        self,
        model=None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
        shots: int = 0,  # Number of few-shot examples (0 = zero-shot)
    ) -> None:
        self.model = model
        self.max_samples = max_samples
        self.verbose = verbose
        self.shots = shots

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _get_default_provider(self):
        """Return a default OpenAI provider if none is set."""
        from ..providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model="gpt-4o", temperature=0.0)

    @abstractmethod
    def load_samples(self) -> List[BenchmarkSample]:
        """Load and return benchmark samples."""

    @abstractmethod
    def _evaluate_sample(
        self,
        sample: BenchmarkSample,
        provider,
    ) -> Tuple[bool, str]:
        """
        Evaluate a single sample.

        Returns:
            (is_correct, model_answer)
        """

    def _format_prompt(self, sample: BenchmarkSample, few_shot_examples: str = "") -> str:
        """Format a sample into a prompt string. Override for custom formatting."""
        prompt = ""
        if few_shot_examples:
            prompt += few_shot_examples + "\n\n"
        prompt += f"Question: {sample.question}\n"
        if sample.choices:
            for i, choice in enumerate(sample.choices):
                label = chr(ord("A") + i)
                prompt += f"{label}. {choice}\n"
            prompt += "\nAnswer with the letter only (A, B, C, or D):"
        else:
            prompt += "\nAnswer:"
        return prompt

    def run(self, samples: Optional[List[BenchmarkSample]] = None) -> BenchmarkResult:
        """Run the benchmark and return a BenchmarkResult."""
        provider = self.model or self._get_default_provider()
        model_name = getattr(provider, "model", str(provider))

        if samples is None:
            samples = self.load_samples()

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        start = time.time()
        correct = 0
        per_sample_results = []
        subject_correct: Dict[str, int] = {}
        subject_total: Dict[str, int] = {}

        for i, sample in enumerate(samples):
            if self.verbose:
                print(f"[{i+1}/{len(samples)}] Evaluating sample {sample.id}...")

            try:
                is_correct, model_answer = self._evaluate_sample(sample, provider)
            except Exception as e:
                is_correct = False
                model_answer = f"ERROR: {e}"

            if is_correct:
                correct += 1

            # Track per-subject scores
            if sample.subject:
                subject_correct.setdefault(sample.subject, 0)
                subject_total.setdefault(sample.subject, 0)
                subject_total[sample.subject] += 1
                if is_correct:
                    subject_correct[sample.subject] += 1

            per_sample_results.append({
                "id": sample.id,
                "question": sample.question[:100],
                "correct_answer": sample.correct_answer,
                "model_answer": model_answer,
                "is_correct": is_correct,
                "subject": sample.subject,
            })

        elapsed = time.time() - start
        total = len(samples)
        accuracy = correct / total if total > 0 else 0.0

        subject_scores = {
            subj: subject_correct.get(subj, 0) / subject_total[subj]
            for subj in subject_total
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            total_samples=total,
            correct=correct,
            accuracy=accuracy,
            subject_scores=subject_scores,
            per_sample_results=per_sample_results,
            elapsed_seconds=elapsed,
            model_name=model_name,
        )
