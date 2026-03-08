from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from langchain_core.documents import Document

from app.config import Settings
from app.rerank.pinecone import PineconeReranker
from app.retrieval.pinecone import PineconeRetrieval
from app.utils.models import RetrievedChunk


def make_settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        pinecone_api_key="test-key",
        raw_data_dir=root / "raw",
        processed_data_dir=root / "processed",
        eval_data_dir=root / "eval",
        faiss_dir=root / "faiss",
        bm25_dir=root / "bm25",
        metadata_db_path=root / "data" / "metadata.db",
    )


class PineconeAdapterTests(unittest.TestCase):
    def test_hits_to_chunks_preserves_metadata(self) -> None:
        response = SimpleNamespace(
            result=SimpleNamespace(
                hits=[
                    SimpleNamespace(
                        _id="chunk-1",
                        _score=0.75,
                        fields={
                            "chunk_text": "Refunds are available for 30 days.",
                            "filename": "policy.md",
                            "page_number": 2,
                            "checksum": "abc123",
                        },
                    )
                ]
            )
        )

        chunks = PineconeRetrieval._hits_to_chunks(response, score_kind="dense")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "chunk-1")
        self.assertEqual(chunks[0].text, "Refunds are available for 30 days.")
        self.assertEqual(chunks[0].metadata["filename"], "policy.md")
        self.assertEqual(chunks[0].metadata["checksum"], "abc123")
        self.assertEqual(chunks[0].dense_score, 0.75)
        self.assertEqual(chunks[0].dense_rank, 1)

    def test_document_to_record_includes_chunk_text_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retrieval = PineconeRetrieval(make_settings(Path(temp_dir)))
            record = retrieval._document_to_record(
                Document(
                    page_content="chunk body",
                    metadata={
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "filename": "policy.md",
                        "page_number": 4,
                        "section_heading": "Refunds",
                        "checksum": "abc123",
                        "source_path": "/tmp/policy.md",
                    },
                )
            )

        self.assertEqual(record["_id"], "chunk-1")
        self.assertEqual(record["chunk_text"], "chunk body")
        self.assertEqual(record["filename"], "policy.md")
        self.assertEqual(record["checksum"], "abc123")

    def test_document_to_record_omits_null_metadata_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            retrieval = PineconeRetrieval(make_settings(Path(temp_dir)))
            record = retrieval._document_to_record(
                Document(
                    page_content="chunk body",
                    metadata={
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "filename": "policy.md",
                        "page_number": 4,
                        "section_heading": None,
                        "checksum": None,
                        "source_path": None,
                    },
                )
            )

        self.assertNotIn("section_heading", record)
        self.assertNotIn("checksum", record)
        self.assertNotIn("source_path", record)

    def test_iter_document_batches_respects_batch_size_and_token_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_settings(Path(temp_dir))
            settings.pinecone_upsert_batch_size = 3
            settings.pinecone_upsert_max_batch_tokens = 5
            retrieval = PineconeRetrieval(settings)

            batches = list(
                retrieval._iter_document_batches(
                    [
                        Document(page_content="abcdefgh", metadata={"chunk_id": "chunk-1"}),
                        Document(page_content="ijklmnop", metadata={"chunk_id": "chunk-2"}),
                        Document(page_content="qrstuvwx", metadata={"chunk_id": "chunk-3"}),
                    ]
                )
            )

        self.assertEqual([len(batch) for batch, _ in batches], [2, 1])
        self.assertEqual([tokens for _, tokens in batches], [4, 2])

    def test_upsert_retries_rate_limit_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_settings(Path(temp_dir))
            settings.enable_sparse_retrieval = False
            settings.pinecone_upsert_batch_size = 1
            settings.pinecone_upsert_retry_attempts = 2
            settings.pinecone_upsert_retry_base_delay_seconds = 0.0
            retrieval = PineconeRetrieval(settings)

        class FakeRateLimitError(RuntimeError):
            status = 429

        class FakeIndex:
            def __init__(self) -> None:
                self.calls = 0

            def upsert_records(self, namespace, records) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise FakeRateLimitError("RESOURCE_EXHAUSTED")

        fake_index = FakeIndex()
        with mock.patch.object(retrieval, "_get_index", return_value=fake_index), mock.patch(
            "app.retrieval.pinecone.time.sleep",
            return_value=None,
        ) as sleep_mock:
            retrieval.upsert_documents(
                [
                    Document(
                        page_content="chunk body",
                        metadata={"chunk_id": "chunk-1", "filename": "policy.md"},
                    )
                ]
            )

        self.assertEqual(fake_index.calls, 2)
        sleep_mock.assert_called_once()

    def test_pinecone_reranker_respects_ranked_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reranker = PineconeReranker(make_settings(Path(temp_dir)))

        calls = {}

        class FakeInference:
            def rerank(self, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, score=0.91),
                        SimpleNamespace(index=0, score=0.52),
                    ]
                )

        reranker._client = SimpleNamespace(inference=FakeInference())
        chunks = [
            RetrievedChunk(chunk_id="chunk-1", text="first", metadata={}),
            RetrievedChunk(chunk_id="chunk-2", text="second", metadata={}),
        ]

        ranked = reranker.rerank("refund policy", chunks, top_n=2)

        self.assertEqual([chunk.chunk_id for chunk in ranked], ["chunk-2", "chunk-1"])
        self.assertEqual(calls["model"], "bge-reranker-v2-m3")
        self.assertEqual(calls["top_n"], 2)
        self.assertAlmostEqual(ranked[0].rerank_score or 0.0, 0.91)


if __name__ == "__main__":
    unittest.main()
