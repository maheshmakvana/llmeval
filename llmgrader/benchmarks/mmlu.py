"""MMLU Benchmark — Massive Multitask Language Understanding."""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Optional, Tuple

from .base import BaseBenchmark, BenchmarkResult, BenchmarkSample

# MMLU has 57 subjects across STEM, humanities, social sciences, and more.
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# Built-in sample questions for zero-dependency testing (subset of real MMLU style)
_SAMPLE_QUESTIONS: List[Dict] = [
    {
        "id": "mmlu_sample_0",
        "subject": "college_mathematics",
        "question": "What is the value of 21! / (21-3)! / 3!?",
        "choices": ["1330", "1540", "5985", "7140"],
        "answer": "A",
    },
    {
        "id": "mmlu_sample_1",
        "subject": "high_school_physics",
        "question": "A particle moves along the x-axis with velocity v(t) = 3t^2 - 2t. "
                    "What is its acceleration at t = 2?",
        "choices": ["8", "10", "12", "14"],
        "answer": "B",
    },
    {
        "id": "mmlu_sample_2",
        "subject": "computer_security",
        "question": "Which of the following is NOT a common type of SQL injection attack?",
        "choices": [
            "Union-based injection",
            "Boolean-based blind injection",
            "Header injection",
            "Heap spray injection",
        ],
        "answer": "D",
    },
    {
        "id": "mmlu_sample_3",
        "subject": "moral_philosophy",
        "question": "According to Kant's categorical imperative, which action would be morally permissible?",
        "choices": [
            "Lying to save someone's life",
            "Keeping a promise even when it causes harm",
            "Breaking a promise when it benefits the majority",
            "Lying when everyone else does it",
        ],
        "answer": "B",
    },
    {
        "id": "mmlu_sample_4",
        "subject": "anatomy",
        "question": "Which structure is responsible for producing bile in the liver?",
        "choices": ["Hepatocytes", "Kupffer cells", "Stellate cells", "Cholangiocytes"],
        "answer": "A",
    },
]

_ZERO_SHOT_PROMPT = """You are an expert in {subject}. Answer the following multiple-choice question.

Question: {question}

Choices:
{choices}

Respond with only the letter of the correct answer (A, B, C, or D).
Answer:"""

_FEW_SHOT_EXAMPLE = """Question: {question}

Choices:
{choices}

Answer: {answer}"""


class MMLUBenchmark(BaseBenchmark):
    """
    Massive Multitask Language Understanding (MMLU) benchmark.

    Tests LLM performance across 57 subjects including STEM, humanities,
    social sciences, and professional knowledge.

    Without a data_path, uses built-in sample questions for quick testing.
    For full MMLU evaluation, provide data_path pointing to MMLU CSV files.

    Args:
        model: LLM provider to evaluate
        subjects: List of MMLU subjects to evaluate (None = all/available)
        data_path: Path to directory containing MMLU CSV files
        max_samples: Maximum number of samples per subject (or total)
        shots: Number of few-shot examples (0=zero-shot, default)
        verbose: Print progress

    Reference: Hendrycks et al. (2021) https://arxiv.org/abs/2009.03300
    """

    def __init__(
        self,
        model=None,
        subjects: Optional[List[str]] = None,
        data_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        shots: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(model=model, max_samples=max_samples, verbose=verbose, shots=shots)
        self.subjects = subjects or MMLU_SUBJECTS
        self.data_path = data_path

    @property
    def name(self) -> str:
        return "MMLU"

    def load_samples(self) -> List[BenchmarkSample]:
        """Load MMLU samples from CSV files or return built-in samples."""
        if self.data_path:
            return self._load_from_csv()
        # Fall back to built-in samples
        return [
            BenchmarkSample(
                id=q["id"],
                question=q["question"],
                choices=q["choices"],
                correct_answer=q["answer"],
                subject=q["subject"],
            )
            for q in _SAMPLE_QUESTIONS
        ]

    def _load_from_csv(self) -> List[BenchmarkSample]:
        """Load from standard MMLU CSV format: question, A, B, C, D, answer."""
        samples = []
        for subject in self.subjects:
            # MMLU CSV filenames: {subject}_test.csv
            csv_path = os.path.join(self.data_path, f"{subject}_test.csv")
            if not os.path.exists(csv_path):
                if self.verbose:
                    print(f"Warning: {csv_path} not found, skipping.")
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if len(row) < 6:
                        continue
                    samples.append(
                        BenchmarkSample(
                            id=f"{subject}_{i}",
                            question=row[0],
                            choices=[row[1], row[2], row[3], row[4]],
                            correct_answer=row[5].strip().upper(),
                            subject=subject,
                        )
                    )
        return samples

    def _evaluate_sample(
        self, sample: BenchmarkSample, provider
    ) -> Tuple[bool, str]:
        choices_str = "\n".join(
            f"{chr(ord('A') + i)}. {c}"
            for i, c in enumerate(sample.choices or [])
        )
        prompt = _ZERO_SHOT_PROMPT.format(
            subject=sample.subject or "general knowledge",
            question=sample.question,
            choices=choices_str,
        )
        response = provider.generate(prompt).strip()

        # Extract answer letter
        import re
        match = re.search(r"\b([A-D])\b", response.upper())
        model_answer = match.group(1) if match else response[:1].upper()

        is_correct = model_answer == (sample.correct_answer or "").upper()
        return is_correct, model_answer
