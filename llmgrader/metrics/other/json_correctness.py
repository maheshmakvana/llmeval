"""JSONCorrectnessMetric — verifies the output is valid JSON matching a schema."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..base import BaseMetric, MetricResult
from ...test_case import LLMTestCase


class JSONCorrectnessMetric(BaseMetric):
    """
    Validates that actual_output is valid JSON and optionally matches a schema.

    If a json_schema dict is provided, checks that all required keys exist
    and values match expected types.

    Requires: actual_output
    """

    def __init__(
        self,
        json_schema: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, strict_mode=strict_mode, verbose_mode=verbose_mode)
        self.json_schema = json_schema

    def measure(self, test_case: LLMTestCase) -> MetricResult:
        steps = []
        output = test_case.actual_output.strip()

        # Remove markdown code fences if present
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1]) if len(lines) > 2 else output

        # Step 1: Valid JSON?
        try:
            parsed = json.loads(output)
            steps.append("Output is valid JSON.")
            json_valid = True
        except json.JSONDecodeError as e:
            steps.append(f"Invalid JSON: {e}")
            score = 0.0
            self._score = score
            self._reason = f"Output is not valid JSON: {e}"
            self._result = MetricResult(
                score=score, passed=False, reason=self._reason,
                metric_name=self.name, threshold=self.threshold,
                strict_mode=self.strict_mode, evaluation_steps=steps,
            )
            return self._result

        # Step 2: Schema validation
        schema_errors = []
        if self.json_schema:
            schema_errors = self._validate_schema(parsed, self.json_schema)
            if schema_errors:
                for err in schema_errors:
                    steps.append(f"Schema error: {err}")
            else:
                steps.append("Output matches the expected schema.")

        schema_ok = len(schema_errors) == 0
        if self.json_schema:
            score = 1.0 if schema_ok else max(0.0, 1.0 - len(schema_errors) * 0.2)
        else:
            score = 1.0

        passed = self._pass_fail(score)
        reason = "Output is valid JSON." if not schema_errors else f"Schema issues: {'; '.join(schema_errors)}"

        self._score = score
        self._reason = reason
        self._result = MetricResult(
            score=score, passed=passed, reason=reason,
            metric_name=self.name, threshold=self.threshold,
            strict_mode=self.strict_mode, evaluation_steps=steps,
        )
        return self._result

    @staticmethod
    def _validate_schema(data: Any, schema: Dict[str, Any]) -> list:
        errors = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if not isinstance(data, dict):
            return ["Output is not a JSON object."]

        for key in required:
            if key not in data:
                errors.append(f"Missing required key: '{key}'")

        for key, spec in properties.items():
            if key in data:
                expected_type = spec.get("type")
                if expected_type:
                    type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}
                    py_type = type_map.get(expected_type)
                    if py_type and not isinstance(data[key], py_type):
                        errors.append(f"Key '{key}' should be {expected_type}, got {type(data[key]).__name__}")

        return errors
