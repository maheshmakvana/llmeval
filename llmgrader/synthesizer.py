"""Synthesizer — auto-generate Golden test cases from documents or prompts."""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from .test_case import Golden

_INPUT_GEN_PROMPT = """Given the following context document, generate {n} diverse and realistic user questions that could be answered using this document.
Make the questions varied in complexity (simple factual, multi-hop reasoning, summarization, comparison).
Return ONLY a JSON array of strings (the questions).

Context:
{context}

Questions (JSON array):"""

_FILTER_PROMPT = """Is the following question a clear, answerable, and high-quality evaluation question?
Return ONLY a JSON object: {{"verdict": "yes"/"no", "reason": "..."}}

Question: {question}

Verdict:"""

_EVOLVE_PROMPT = """Rewrite the following question to be more complex, specific, or challenging while keeping it answerable.
Return ONLY the rewritten question (plain text, no quotes).

Original: {question}

Rewritten:"""

_EXPECTED_OUTPUT_PROMPT = """Given the following context and question, generate a concise and accurate expected answer.

Context:
{context}

Question: {question}

Answer:"""


class Synthesizer:
    """
    Generates synthetic Golden datasets from documents/contexts.

    Pipeline:
    1. Input generation — produce questions from context
    2. Filtration — remove low-quality questions
    3. Evolution — increase question complexity
    4. Output generation — generate expected answers

    Example:
        synth = Synthesizer(model=OpenAIProvider())
        goldens = synth.generate_goldens_from_docs(
            documents=["The Eiffel Tower is 330 meters tall..."],
            max_goldens_per_doc=5,
        )
    """

    def __init__(self, model=None) -> None:
        self._model = model

    def _get_provider(self):
        if self._model:
            return self._model
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model="gpt-4o", temperature=0.7)

    def _generate(self, prompt: str) -> str:
        return self._get_provider().generate(prompt)

    def _parse_list(self, response: str) -> List[str]:
        import json
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        lines = [l.strip().lstrip("-•1234567890. ") for l in response.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 5]

    def generate_goldens_from_docs(
        self,
        documents: List[str],
        max_goldens_per_doc: int = 5,
        filter_questions: bool = True,
        evolve_questions: bool = True,
        generate_expected_outputs: bool = True,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ) -> List[Golden]:
        """
        Generate Golden test cases from a list of document strings.

        Args:
            documents: List of raw document texts.
            max_goldens_per_doc: Max questions to generate per document.
            filter_questions: Remove low-quality questions using LLM.
            evolve_questions: Make questions more complex.
            generate_expected_outputs: Auto-generate expected_output for each golden.
            chunk_size: Characters per chunk (for long documents).
            chunk_overlap: Overlap between chunks.
        """
        goldens = []
        provider = self._get_provider()

        for doc in documents:
            chunks = self._chunk_text(doc, chunk_size, chunk_overlap)
            for chunk in chunks:
                # Step 1: Generate
                raw = self._generate(
                    _INPUT_GEN_PROMPT.format(n=max_goldens_per_doc, context=chunk)
                )
                questions = self._parse_list(raw)[:max_goldens_per_doc]

                # Step 2: Filter
                if filter_questions:
                    filtered = []
                    for q in questions:
                        fraw = self._generate(_FILTER_PROMPT.format(question=q))
                        import json
                        try:
                            import re as _re
                            m = _re.search(r"\{.*?\}", fraw, _re.DOTALL)
                            obj = json.loads(m.group()) if m else {}
                            if obj.get("verdict", "yes").lower() == "yes":
                                filtered.append(q)
                        except Exception:
                            filtered.append(q)
                    questions = filtered

                # Step 3: Evolve
                if evolve_questions:
                    evolved = []
                    for q in questions:
                        eraw = self._generate(_EVOLVE_PROMPT.format(question=q)).strip()
                        evolved.append(eraw if eraw else q)
                    questions = evolved

                # Step 4: Generate expected outputs and build Goldens
                for q in questions:
                    expected = None
                    if generate_expected_outputs:
                        expected = self._generate(
                            _EXPECTED_OUTPUT_PROMPT.format(context=chunk, question=q)
                        ).strip()

                    goldens.append(Golden(
                        input=q,
                        expected_output=expected,
                        retrieval_context=[chunk],
                    ))

        return goldens

    def generate_goldens_from_inputs(
        self,
        inputs: List[str],
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> List[Golden]:
        """
        Create Goldens from raw input strings.
        Optionally generate expected_output using a provided function.
        """
        goldens = []
        for inp in inputs:
            expected = generate_fn(inp) if generate_fn else None
            goldens.append(Golden(input=inp, expected_output=expected))
        return goldens

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
