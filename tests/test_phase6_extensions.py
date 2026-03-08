from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.documents import Document

from app.chains.rag import RAGPipeline
from app.config import Settings
from app.utils.background_jobs import BackgroundIndexManager
from app.utils.models import RetrievalFilters, RetrievedChunk
from app.utils.telemetry import StructuredLogger


def make_settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        retrieval_provider="local",
        rerank_provider="local",
        answer_mode="extractive",
        enable_dense_retrieval=False,
        enable_sparse_retrieval=True,
        enable_reranker=False,
        raw_data_dir=root / "raw",
        processed_data_dir=root / "processed",
        eval_data_dir=root / "eval",
        faiss_dir=root / "faiss",
        bm25_dir=root / "bm25",
        metadata_db_path=root / "data" / "metadata.db",
        telemetry_log_path=root / "processed" / "telemetry.jsonl",
    )


class Phase6ExtensionsTests(unittest.TestCase):
    def test_retrieval_cache_reuses_results_for_same_query_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            pipeline = RAGPipeline(settings=settings)
            pipeline.store.replace_document_chunks(
                pipeline_store_set("doc-1", "checksum-1", "doc.txt", "ops"),
                [pipeline_store_chunk("chunk-1", "doc-1", "hello world", "doc.txt", "ops")],
            )
            returned_chunk = RetrievedChunk(
                chunk_id="chunk-1",
                text="hello world",
                metadata={"filename": "doc.txt", "collection_name": "ops"},
            )
            filters = RetrievalFilters(filenames=["doc.txt"], collection_names=["ops"])

            with mock.patch.object(pipeline, "_ensure_sparse_index", return_value=None), mock.patch.object(
                pipeline.sparse_index,
                "search",
                return_value=[returned_chunk],
            ) as sparse_search:
                first = pipeline.retrieve("hello", filters=filters)
                second = pipeline.retrieve("hello", filters=filters)

            self.assertEqual(first[0].chunk_id, "chunk-1")
            self.assertEqual(second[0].chunk_id, "chunk-1")
            self.assertEqual(sparse_search.call_count, 1)

    def test_collections_are_persisted_and_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.enable_sparse_retrieval = False
            pipeline = RAGPipeline(settings=settings)
            sample_file = root / "policy.txt"
            sample_file.write_text("refunds are available", encoding="utf-8")

            summary = pipeline.index_paths([sample_file], collection_name="policies")

            self.assertEqual(summary.files_indexed, 1)
            self.assertIn("policies", pipeline.list_collections())

    def test_structured_logger_writes_jsonl_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            logger = StructuredLogger(settings)

            logger.log_event("phase6_test", {"ok": True, "count": 1})
            events = logger.read_recent(limit=5)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "phase6_test")
            self.assertEqual(events[0]["payload"]["ok"], True)

    def test_background_index_job_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.enable_sparse_retrieval = False
            sample_file = root / "notes.txt"
            sample_file.write_text("hello from background indexing", encoding="utf-8")
            manager = BackgroundIndexManager()

            job_id = manager.start_index_job(settings, [sample_file], "background")

            deadline = time.time() + 8.0
            status = manager.get_job(job_id)
            while status is not None and status.status in {"queued", "running"} and time.time() < deadline:
                time.sleep(0.05)
                status = manager.get_job(job_id)

            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status.status, "completed")
            self.assertIsNotNone(status.summary)
            assert status.summary is not None
            self.assertEqual(status.summary.collection_name, "background")


def pipeline_store_set(doc_id: str, checksum: str, filename: str, collection_name: str):
    from app.utils.models import LoadedDocumentSet

    return LoadedDocumentSet(
        doc_id=doc_id,
        filename=filename,
        file_path="/tmp/{filename}".format(filename=filename),
        checksum=checksum,
        file_type="txt",
        documents=[],
        collection_name=collection_name,
    )


def pipeline_store_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    filename: str,
    collection_name: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "doc_id": doc_id,
            "filename": filename,
            "collection_name": collection_name,
        },
    )


if __name__ == "__main__":
    unittest.main()
