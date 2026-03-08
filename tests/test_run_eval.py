from __future__ import annotations

import unittest

from app.utils.models import AnswerResult, RetrievedChunk
from scripts.run_eval import evaluate_generation, evaluate_retrieval, gold_targets


def make_chunk(chunk_id: str, filename: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="chunk text",
        metadata={"filename": filename},
    )


class RunEvalTests(unittest.TestCase):
    def test_gold_targets_prefers_chunk_ids(self) -> None:
        target_type, values = gold_targets(
            {
                "question": "q",
                "gold_chunk_ids": ["chunk-1", "chunk-2"],
                "gold_filenames": ["doc.md"],
            }
        )

        self.assertEqual(target_type, "chunk_id")
        self.assertEqual(values, ["chunk-1", "chunk-2"])

    def test_evaluate_retrieval_supports_filename_targets(self) -> None:
        examples = [
            {
                "question": "refund",
                "gold_filenames": ["product_guide.md"],
            }
        ]

        metrics, failures = evaluate_retrieval(
            examples,
            lambda question: [
                make_chunk("chunk-1", "product_guide.md"),
                make_chunk("chunk-2", "product_guide.md"),
            ],
            top_k=5,
        )

        self.assertEqual(metrics["evaluated_examples"], 1)
        self.assertAlmostEqual(metrics["HitRate@5"], 1.0)
        self.assertAlmostEqual(metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(metrics["MRR@5"], 1.0)
        self.assertAlmostEqual(metrics["nDCG@5"], 1.0)
        self.assertEqual(failures, [])

    def test_evaluate_generation_tracks_phrase_refusal_and_citations(self) -> None:
        examples = [
            {
                "question": "refund",
                "answer_contains": ["30 calendar days"],
            },
            {
                "question": "gym policy",
                "expected_refusal": True,
            },
        ]

        def answer_fn(question: str) -> AnswerResult:
            if question == "refund":
                return AnswerResult(answer_markdown="Refunds are allowed for 30 calendar days. [C1]")
            return AnswerResult(
                answer_markdown="I don't know based on the uploaded documents.",
                abstained=True,
            )

        metrics, failures = evaluate_generation(examples, answer_fn)

        self.assertEqual(metrics["checked_examples"], 2)
        self.assertAlmostEqual(metrics["answer_contains_pass_rate"], 1.0)
        self.assertAlmostEqual(metrics["refusal_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["citation_coverage"], 1.0)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
