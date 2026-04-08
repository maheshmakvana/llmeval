"""EvaluationDataset — stores and manages Golden test case collections."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .test_case import ConversationalGolden, Golden, LLMTestCase


class EvaluationDataset:
    """
    A versioned collection of Golden test case templates.

    Goldens are stored locally as JSON. The dataset supports:
    - Adding / removing goldens
    - Saving/loading from disk (JSON)
    - Iterating as LLMTestCase objects (by supplying an LLM function)
    - Dataset versioning (history log)

    Example:
        dataset = EvaluationDataset()
        dataset.add_goldens([
            Golden(input="What is 2+2?", expected_output="4"),
        ])
        dataset.save("my_dataset.json")
    """

    def __init__(self, goldens: Optional[List[Golden]] = None) -> None:
        self.goldens: List[Golden] = goldens or []
        self._version: str = str(uuid.uuid4())[:8]
        self._created_at: str = datetime.utcnow().isoformat()
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_goldens(self, goldens: List[Golden]) -> None:
        self.goldens.extend(goldens)
        self._log_event("add", f"Added {len(goldens)} golden(s)")

    def remove_golden(self, index: int) -> None:
        removed = self.goldens.pop(index)
        self._log_event("remove", f"Removed golden: '{removed.input[:50]}'")

    def clear(self) -> None:
        self.goldens.clear()
        self._log_event("clear", "Cleared all goldens")

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.goldens)

    def __iter__(self) -> Iterator[Golden]:
        return iter(self.goldens)

    def __getitem__(self, idx: int) -> Golden:
        return self.goldens[idx]

    def to_test_cases(self, generate_fn: Callable[[str], str]) -> List[LLMTestCase]:
        """
        Convert goldens to LLMTestCases by calling generate_fn(input) for each.

        Args:
            generate_fn: A callable that takes an input string and returns
                         the LLM's actual_output string.
        """
        test_cases = []
        for golden in self.goldens:
            actual_output = generate_fn(golden.input)
            test_cases.append(golden.to_test_case(actual_output))
        return test_cases

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the dataset to a JSON file."""
        data = {
            "version": self._version,
            "created_at": self._created_at,
            "goldens": [g.to_dict() for g in self.goldens],
            "history": self._history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":
        """Load a dataset from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        goldens = []
        for g in data.get("goldens", []):
            goldens.append(Golden(
                input=g["input"],
                expected_output=g.get("expected_output"),
                context=g.get("context"),
                retrieval_context=g.get("retrieval_context"),
                expected_tools=g.get("expected_tools"),
                additional_metadata=g.get("additional_metadata", {}),
                source_file=g.get("source_file"),
            ))

        dataset = cls(goldens=goldens)
        dataset._version = data.get("version", "unknown")
        dataset._created_at = data.get("created_at", "")
        dataset._history = data.get("history", [])
        return dataset

    @classmethod
    def from_csv(cls, path: str, input_col: str = "input", expected_col: str = "expected_output") -> "EvaluationDataset":
        """Load goldens from a CSV file."""
        import csv
        goldens = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                goldens.append(Golden(
                    input=row.get(input_col, ""),
                    expected_output=row.get(expected_col),
                ))
        return cls(goldens=goldens)

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, description: str) -> None:
        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "description": description,
        })

    def print_history(self) -> None:
        for entry in self._history:
            print(f"[{entry['timestamp']}] {entry['type'].upper()}: {entry['description']}")

    def __repr__(self) -> str:
        return f"EvaluationDataset(goldens={len(self.goldens)}, version='{self._version}')"
