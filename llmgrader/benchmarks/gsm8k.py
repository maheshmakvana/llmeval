"""GSM8K Benchmark — Grade School Math word problems."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseBenchmark, BenchmarkSample

# Built-in sample questions in GSM8K style
_SAMPLE_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "gsm8k_sample_0",
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every "
                    "morning and bakes muffins for her friends every day with 4 eggs. "
                    "She sells the remainder at the farmers' market daily for $2 per "
                    "fresh duck egg. How much in dollars does she make every day at "
                    "the farmers' market?",
        "answer": "18",
        "solution": "Janet sells 16 - 3 - 4 = 9 duck eggs per day. She makes 9 * 2 = $18 every day.",
    },
    {
        "id": "gsm8k_sample_1",
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                    "How many bolts in total does it take?",
        "answer": "3",
        "solution": "It takes 2/2 = 1 bolt of white fiber. So the total is 2 + 1 = 3 bolts.",
    },
    {
        "id": "gsm8k_sample_2",
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 "
                    "and then puts in $50,000 in repairs. This increased the value of the "
                    "house by 150%. How much profit did he make?",
        "answer": "70000",
        "solution": (
            "The cost is 80,000 + 50,000 = $130,000. "
            "The house increased in value by 80,000 * 1.5 = $120,000, "
            "so it's worth 80,000 + 120,000 = $200,000. "
            "Profit = 200,000 - 130,000 = $70,000."
        ),
    },
    {
        "id": "gsm8k_sample_3",
        "question": "There are 15 trees in the grove. Grove workers will plant trees in "
                    "the grove today. After they are done, there will be 21 trees. How "
                    "many trees did the grove workers plant today?",
        "answer": "6",
        "solution": "21 - 15 = 6 trees were planted.",
    },
    {
        "id": "gsm8k_sample_4",
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
                    "many pieces do they have left in total?",
        "answer": "39",
        "solution": "Total = 32 + 42 = 74. After eating 35: 74 - 35 = 39.",
    },
    {
        "id": "gsm8k_sample_5",
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, "
                    "how many cars are in the parking lot?",
        "answer": "5",
        "solution": "3 + 2 = 5 cars.",
    },
]

_ZERO_SHOT_PROMPT = """Solve the following math problem step by step.

Problem: {question}

Show your work and reasoning. At the end, write your final answer as:
Final Answer: <number>

Solution:"""

_FEW_SHOT_EXAMPLES = """Here are some example math problems with solutions:

Problem: There are 3 birds in a tree. 2 more land. How many birds are there total?
Solution: 3 + 2 = 5 birds.
Final Answer: 5

Problem: A store has 10 apples. They sell 4. How many remain?
Solution: 10 - 4 = 6 apples remain.
Final Answer: 6

Now solve:"""


class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K (Grade School Math 8K) benchmark.

    Tests mathematical reasoning ability using grade school math word problems.
    Each problem requires multi-step reasoning and basic arithmetic.

    Without a data_path, uses built-in sample questions for quick testing.
    For full GSM8K evaluation, provide data_path pointing to the .jsonl file.

    Args:
        model: LLM provider to evaluate
        data_path: Path to GSM8K JSONL file (test.jsonl or train.jsonl)
        max_samples: Maximum number of samples to evaluate
        shots: Number of few-shot examples (0=zero-shot, default)
        verbose: Print progress

    Reference: Cobbe et al. (2021) https://arxiv.org/abs/2110.14168
    """

    def __init__(
        self,
        model=None,
        data_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        shots: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(model=model, max_samples=max_samples, verbose=verbose, shots=shots)
        self.data_path = data_path

    @property
    def name(self) -> str:
        return "GSM8K"

    def load_samples(self) -> List[BenchmarkSample]:
        """Load GSM8K samples from JSONL or return built-in samples."""
        if self.data_path and os.path.exists(self.data_path):
            return self._load_from_jsonl()
        return [
            BenchmarkSample(
                id=q["id"],
                question=q["question"],
                correct_answer=q["answer"],
                subject="math",
                metadata={"solution": q["solution"]},
            )
            for q in _SAMPLE_QUESTIONS
        ]

    def _load_from_jsonl(self) -> List[BenchmarkSample]:
        """Load from GSM8K JSONL format: {'question': ..., 'answer': ...}"""
        samples = []
        with open(self.data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    question = obj["question"]
                    raw_answer = obj["answer"]

                    # GSM8K answers end with #### <number>
                    answer_match = re.search(r"####\s*(-?\d+(?:,\d+)*)", raw_answer)
                    if answer_match:
                        # Normalize: remove commas from numbers like 1,000
                        numeric_answer = answer_match.group(1).replace(",", "")
                    else:
                        numeric_answer = raw_answer.strip()

                    samples.append(
                        BenchmarkSample(
                            id=f"gsm8k_{i}",
                            question=question,
                            correct_answer=numeric_answer,
                            subject="math",
                            metadata={"full_solution": raw_answer},
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
        return samples

    def _extract_final_answer(self, response: str) -> str:
        """Extract numeric answer from model response."""
        # Look for "Final Answer: <number>"
        match = re.search(
            r"(?i)final\s+answer\s*[:\-]?\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
            response,
        )
        if match:
            return match.group(1).replace(",", "")

        # Look for #### pattern (few-shot format)
        match = re.search(r"####\s*(-?\d+(?:,\d+)*)", response)
        if match:
            return match.group(1).replace(",", "")

        # Last number mentioned in the response
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", response)
        if numbers:
            return numbers[-1].replace(",", "")

        return ""

    def _evaluate_sample(
        self, sample: BenchmarkSample, provider
    ) -> Tuple[bool, str]:
        few_shot = _FEW_SHOT_EXAMPLES if self.shots > 0 else ""
        prompt = (few_shot + "\n\n" if few_shot else "") + _ZERO_SHOT_PROMPT.format(
            question=sample.question
        )

        response = provider.generate(prompt).strip()
        model_answer = self._extract_final_answer(response)

        # Normalize expected answer
        expected = (sample.correct_answer or "").replace(",", "").strip()

        # Compare numerically if possible, otherwise string
        try:
            is_correct = abs(float(model_answer) - float(expected)) < 1e-6
        except (ValueError, TypeError):
            is_correct = model_answer.strip() == expected.strip()

        return is_correct, model_answer
