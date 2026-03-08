from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.documents import Document

from app.ingestion import loaders
from app.chains.rag import RAGPipeline
from app.config import Settings
from app.ingestion.loaders import build_doc_id
from app.storage.metadata_store import MetadataStore
from app.utils.models import LoadedDocumentSet, RetrievedChunk


def make_settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        ollama_base_url="http://127.0.0.1:11434",
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
    )


def make_loaded_set(doc_id: str, checksum: str, filename: str) -> LoadedDocumentSet:
    return LoadedDocumentSet(
        doc_id=doc_id,
        filename=filename,
        file_path="/tmp/{filename}".format(filename=filename),
        checksum=checksum,
        file_type="txt",
        documents=[],
    )


def make_chunk(chunk_id: str, doc_id: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "doc_id": doc_id,
            "filename": "{doc_id}.txt".format(doc_id=doc_id),
        },
    )


class PipelineGuardTests(unittest.TestCase):
    def test_build_doc_id_depends_on_checksum_not_filename(self) -> None:
        checksum = "abc123"
        first = build_doc_id(Path("first.pdf"), checksum)
        second = build_doc_id(Path("second.pdf"), checksum)
        self.assertEqual(first, second)

    def test_groq_chat_model_is_not_treated_as_ollama_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_settings(Path(temp_dir))
            settings.answer_mode = "llm"
            settings.llm_provider = "groq"
            settings.retrieval_provider = "pinecone"
            settings.rerank_provider = "pinecone"
            settings.enable_dense_retrieval = True
            settings.enable_sparse_retrieval = True
            settings.enable_reranker = True

            self.assertEqual(settings.required_ollama_models(), [])

    def test_missing_pinecone_key_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_settings(Path(temp_dir))
            settings.retrieval_provider = "pinecone"
            settings.enable_dense_retrieval = True
            settings.enable_sparse_retrieval = True
            pipeline = RAGPipeline(settings=settings)
            pipeline.store.replace_document_chunks(
                make_loaded_set("doc-1", "checksum-1", "doc.txt"),
                [make_chunk("chunk-1", "doc-1", "hello world")],
            )

            with self.assertRaisesRegex(RuntimeError, "RAG_PINECONE_API_KEY"):
                pipeline.retrieve("hello")

    def test_cloud_indexing_upserts_new_chunks_without_local_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.retrieval_provider = "pinecone"
            settings.rerank_provider = "pinecone"
            settings.pinecone_api_key = "test-key"
            settings.enable_dense_retrieval = True
            settings.enable_sparse_retrieval = True
            pipeline = RAGPipeline(settings=settings)
            file_path = root / "doc.txt"
            file_path.write_text("hello world", encoding="utf-8")

            with mock.patch.object(pipeline.pinecone_retrieval, "upsert_documents", return_value=None) as upsert_mock, mock.patch.object(
                pipeline.dense_index,
                "build",
                side_effect=AssertionError("local dense build should not run"),
            ), mock.patch.object(
                pipeline.sparse_index,
                "build",
                side_effect=AssertionError("local sparse build should not run"),
            ):
                summary = pipeline.index_paths([file_path])

            self.assertEqual(summary.files_indexed, 1)
            upsert_mock.assert_called_once()

    def test_cloud_indexing_does_not_store_failed_remote_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.retrieval_provider = "pinecone"
            settings.rerank_provider = "pinecone"
            settings.pinecone_api_key = "test-key"
            settings.enable_dense_retrieval = True
            settings.enable_sparse_retrieval = True
            pipeline = RAGPipeline(settings=settings)
            file_path = root / "doc.txt"
            file_path.write_text("hello world", encoding="utf-8")

            with mock.patch.object(
                pipeline.pinecone_retrieval,
                "upsert_documents",
                side_effect=RuntimeError("bad payload"),
            ):
                summary = pipeline.index_paths([file_path])

            self.assertEqual(summary.files_indexed, 0)
            self.assertEqual(pipeline.indexed_document_count(), 0)
            self.assertEqual(len(summary.failed_files), 1)

    def test_remove_duplicate_documents_keeps_latest_checksum_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.ensure_directories()
            store = MetadataStore(settings.metadata_db_path)

            store.replace_document_chunks(
                make_loaded_set("doc-old", "same-checksum", "old.txt"),
                [make_chunk("chunk-old", "doc-old", "old content")],
            )
            store.replace_document_chunks(
                make_loaded_set("doc-new", "same-checksum", "new.txt"),
                [make_chunk("chunk-new", "doc-new", "new content")],
            )

            removed = store.remove_duplicate_documents()

            self.assertEqual(removed, 1)
            self.assertEqual(store.document_count(), 1)
            self.assertEqual(store.chunk_count(), 1)
            remaining = store.load_all_chunk_documents()
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].metadata["chunk_id"], "chunk-new")

    def test_small_talk_short_circuits_rag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(settings=make_settings(Path(temp_dir)))

            with mock.patch.object(pipeline, "retrieve", side_effect=AssertionError("retrieve should not run")):
                result = pipeline.answer("hi")

            self.assertTrue(result.abstained)
            self.assertIn("Ask a question about the indexed documents", result.answer_markdown)

    def test_sparse_index_is_reused_when_chunk_count_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = RAGPipeline(settings=make_settings(root))
            pipeline.store.replace_document_chunks(
                make_loaded_set("doc-1", "checksum-1", "doc.txt"),
                [make_chunk("chunk-1", "doc-1", "hello world")],
            )

            with mock.patch.object(pipeline, "_ensure_ollama_runtime", return_value=None), mock.patch.object(
                pipeline.store,
                "load_all_chunk_documents",
                wraps=pipeline.store.load_all_chunk_documents,
            ) as load_all_mock, mock.patch.object(
                pipeline.sparse_index,
                "search",
                return_value=[],
            ):
                pipeline.retrieve("first question")
                pipeline.retrieve("second question")

            self.assertEqual(load_all_mock.call_count, 1)

    def test_index_paths_skips_files_with_existing_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.ensure_directories()
            pipeline = RAGPipeline(settings=settings)
            file_path = root / "duplicate.txt"
            file_path.write_text("same text", encoding="utf-8")

            with mock.patch.object(pipeline.dense_index, "build", return_value=None), mock.patch.object(
                pipeline.sparse_index,
                "build",
                return_value=None,
            ):
                first_summary = pipeline.index_paths([file_path])
                second_summary = pipeline.index_paths([file_path])

            self.assertEqual(first_summary.files_indexed, 1)
            self.assertEqual(second_summary.files_indexed, 0)
            self.assertEqual(len(second_summary.skipped_files), 1)
            self.assertEqual(pipeline.indexed_document_count(), 1)

    def test_index_paths_skips_existing_checksum_before_loading_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = RAGPipeline(settings=make_settings(root))
            pipeline.store.replace_document_chunks(
                make_loaded_set("doc-1", "checksum-1", "doc.txt"),
                [make_chunk("chunk-1", "doc-1", "hello world")],
            )

            with mock.patch.object(pipeline.dense_index, "load", return_value=True), mock.patch(
                "app.chains.rag.compute_checksum",
                return_value="checksum-1",
            ), mock.patch(
                "app.chains.rag.load_file",
                side_effect=AssertionError("load_file should not run for existing checksums"),
            ):
                summary = pipeline.index_paths([root / "already-indexed.pdf"])

            self.assertEqual(summary.files_indexed, 0)
            self.assertEqual(len(summary.skipped_files), 1)

    def test_refresh_indexes_uses_stored_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = RAGPipeline(settings=make_settings(root))
            pipeline.store.replace_document_chunks(
                make_loaded_set("doc-1", "checksum-1", "doc.txt"),
                [make_chunk("chunk-1", "doc-1", "hello world")],
            )

            with mock.patch.object(pipeline.dense_index, "build", return_value=None) as dense_build, mock.patch.object(
                pipeline.sparse_index,
                "build",
                return_value=None,
            ) as sparse_build:
                summary = pipeline.refresh_indexes()

            dense_build.assert_not_called()
            sparse_build.assert_called_once()
            self.assertEqual(summary.total_documents, 1)
            self.assertEqual(summary.total_chunks, 1)

    def test_pdf_loading_prefers_pymupdf_when_available(self) -> None:
        expected = [Document(page_content="fast path", metadata={})]
        with mock.patch.object(loaders, "_load_pdf_with_pymupdf", return_value=expected) as pymupdf_loader, mock.patch.object(
            loaders,
            "_load_pdf_with_pypdf",
            side_effect=AssertionError("pypdf should not run when pymupdf succeeds"),
        ):
            result = loaders._load_pdf(Path("sample.pdf"), "doc-1")

        pymupdf_loader.assert_called_once()
        self.assertEqual(result, expected)

    def test_answer_uses_extractive_mode_without_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = RAGPipeline(settings=make_settings(root))
            chunk = RetrievedChunk(
                chunk_id="chunk-1",
                text="Refunds are available for 30 calendar days after purchase.",
                metadata={"filename": "product_guide.md"},
            )

            with mock.patch.object(pipeline, "retrieve", return_value=[chunk]), mock.patch.object(
                pipeline,
                "_ensure_ollama_runtime",
                side_effect=AssertionError("extractive mode should not require Ollama"),
            ):
                result = pipeline.answer("How long is the refund window?")

            self.assertFalse(result.abstained)
            self.assertIn("Relevant passages", result.answer_markdown)
            self.assertIn("[C1]", result.answer_markdown)
            self.assertEqual(len(result.citations), 1)


if __name__ == "__main__":
    unittest.main()
