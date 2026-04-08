"""HellaSwag Benchmark — commonsense NLI for grounding in physical situations."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseBenchmark, BenchmarkSample

# Built-in sample questions in HellaSwag style
_SAMPLE_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "hellaswag_sample_0",
        "activity_label": "Grooming a dog",
        "ctx": "He is grooming the dog. He uses the brush on the dog's fur. ",
        "endings": [
            "He puts the brush away and starts vacuuming.",
            "He carefully brushes out any tangles, working from tips to roots.",
            "He throws the brush at the dog, angering it.",
            "He stops brushing and starts playing video games.",
        ],
        "label": 1,
    },
    {
        "id": "hellaswag_sample_1",
        "activity_label": "Baking a cake",
        "ctx": "She preheats the oven to 350 degrees. She mixes flour, sugar, and eggs together. ",
        "endings": [
            "She pours the batter into a greased pan and puts it in the oven.",
            "She immediately eats the raw batter with a spoon.",
            "She puts the mixing bowl in the freezer.",
            "She calls a friend to come over and bake instead.",
        ],
        "label": 0,
    },
    {
        "id": "hellaswag_sample_2",
        "activity_label": "Changing a tire",
        "ctx": "He notices the tire is flat. He opens the trunk and takes out the spare tire and jack. ",
        "endings": [
            "He uses the jack to lift the car before loosening the lug nuts.",
            "He loosens the lug nuts first, then uses the jack to lift the car.",
            "He leaves the flat tire on and drives slowly to a gas station.",
            "He puts the spare tire on the roof of the car.",
        ],
        "label": 1,
    },
    {
        "id": "hellaswag_sample_3",
        "activity_label": "Making coffee",
        "ctx": "She fills the coffee maker with water and adds ground coffee to the filter. ",
        "endings": [
            "She presses start and waits for the coffee to brew.",
            "She puts the coffee maker in the microwave.",
            "She pours the unbrewed water directly into her mug.",
            "She adds the coffee grounds to a pot of boiling water on the stove.",
        ],
        "label": 0,
    },
    {
        "id": "hellaswag_sample_4",
        "activity_label": "Parallel parking",
        "ctx": "He signals and pulls up alongside the car ahead of the parking space. ",
        "endings": [
            "He reverses at an angle into the space, straightening the wheels as he goes.",
            "He drives forward past the space, then stops.",
            "He immediately turns the wheel hard right and drives into the space.",
            "He gets out of the car and pushes it into the space manually.",
        ],
        "label": 0,
    },
]

_HELLASWAG_PROMPT = """Complete the following activity description by choosing the most plausible continuation.

Activity: {activity}
Context: {context}

Which ending is the most plausible continuation?
A. {ending_a}
B. {ending_b}
C. {ending_c}
D. {ending_d}

Respond with only the letter of the most plausible ending (A, B, C, or D).
Answer:"""


class HellaSwagBenchmark(BaseBenchmark):
    """
    HellaSwag benchmark for commonsense natural language inference.

    Tests an LLM's ability to select the most physically plausible
    continuation of an activity description from four options.

    Without a data_path, uses built-in sample questions for quick testing.
    For full HellaSwag evaluation, provide data_path pointing to .jsonl files.

    Args:
        model: LLM provider to evaluate
        data_path: Path to HellaSwag .jsonl file (train/val/test)
        max_samples: Maximum number of samples to evaluate
        shots: Number of few-shot examples (0=zero-shot, default)
        verbose: Print progress

    Reference: Zellers et al. (2019) https://arxiv.org/abs/1905.07830
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
        return "HellaSwag"

    def load_samples(self) -> List[BenchmarkSample]:
        """Load HellaSwag samples from JSONL or return built-in samples."""
        if self.data_path and os.path.exists(self.data_path):
            return self._load_from_jsonl()
        return [
            BenchmarkSample(
                id=q["id"],
                question=q["ctx"],
                choices=q["endings"],
                correct_answer=chr(ord("A") + q["label"]),
                subject=q["activity_label"],
                metadata={"activity_label": q["activity_label"]},
            )
            for q in _SAMPLE_QUESTIONS
        ]

    def _load_from_jsonl(self) -> List[BenchmarkSample]:
        """Load from standard HellaSwag JSONL format."""
        samples = []
        with open(self.data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # HellaSwag fields: ind, activity_label, ctx_a, ctx_b, ctx, endings, label
                    ctx = obj.get("ctx", obj.get("ctx_a", "") + " " + obj.get("ctx_b", ""))
                    endings = obj.get("endings", [])
                    label = int(obj.get("label", 0))
                    samples.append(
                        BenchmarkSample(
                            id=str(obj.get("ind", i)),
                            question=ctx.strip(),
                            choices=endings,
                            correct_answer=chr(ord("A") + label),
                            subject=obj.get("activity_label", ""),
                            metadata={"activity_label": obj.get("activity_label", "")},
                        )
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return samples

    def _evaluate_sample(
        self, sample: BenchmarkSample, provider
    ) -> Tuple[bool, str]:
        choices = sample.choices or []
        # Pad to 4 choices if needed
        while len(choices) < 4:
            choices.append("N/A")

        activity = sample.metadata.get("activity_label", sample.subject or "general")
        prompt = _HELLASWAG_PROMPT.format(
            activity=activity,
            context=sample.question,
            ending_a=choices[0],
            ending_b=choices[1],
            ending_c=choices[2],
            ending_d=choices[3],
        )
        response = provider.generate(prompt).strip()

        import re
        match = re.search(r"\b([A-D])\b", response.upper())
        model_answer = match.group(1) if match else response[:1].upper()

        is_correct = model_answer == (sample.correct_answer or "").upper()
        return is_correct, model_answer
