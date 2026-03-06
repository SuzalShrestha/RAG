from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from app.config import Settings, get_settings
from app.generation.answer_chain import AnswerGenerator
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_file
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseIndex
from app.rerank.cross_encoder import CrossEncoderReranker
from app.storage.metadata_store import MetadataStore
from app.utils.models import AnswerResult, IndexingSummary, RetrievedChunk


class RAGPipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.store = MetadataStore(self.settings.metadata_db_path)
        self.dense_index = DenseIndex(self.settings)
        self.sparse_index = SparseIndex()
        self.hybrid_retriever = HybridRetriever(
            dense_index=self.dense_index,
            sparse_index=self.sparse_index,
            settings=self.settings,
        )
        self.reranker: Optional[CrossEncoderReranker] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self._refresh_sparse_index()

    def index_paths(self, paths: Iterable[Path]) -> IndexingSummary:
        loaded_sets = []
        failed_files = []
        for path in paths:
            raw_path = Path(path)
            try:
                loaded_sets.append(load_file(raw_path))
            except Exception as error:
                failed_files.append("{path}: {error}".format(path=raw_path, error=error))

        files_indexed = 0
        chunks_indexed = 0

        for loaded_set in loaded_sets:
            chunks = chunk_documents(
                loaded_set.documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self.store.replace_document_chunks(loaded_set, chunks)
            files_indexed += 1
            chunks_indexed += len(chunks)

        all_chunks = self.store.load_all_chunk_documents()
        self.dense_index.build(all_chunks)
        self.sparse_index.build(all_chunks, self.settings.sparse_top_k)

        return IndexingSummary(
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
            total_documents=self.store.document_count(),
            total_chunks=self.store.chunk_count(),
            failed_files=failed_files,
        )

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        self._refresh_sparse_index()
        return self.hybrid_retriever.retrieve(question)

    def answer(self, question: str) -> AnswerResult:
        retrieved = self.retrieve(question)
        reranker = self._get_reranker()
        reranked = reranker.rerank(
            question,
            retrieved,
            top_n=self.settings.final_context_k,
        )
        answer_generator = self._get_answer_generator()
        result = answer_generator.answer(question, reranked)
        result.retrieved_chunks = retrieved
        return result

    def indexed_document_count(self) -> int:
        return self.store.document_count()

    def indexed_chunk_count(self) -> int:
        return self.store.chunk_count()

    def _refresh_sparse_index(self) -> None:
        all_chunks = self.store.load_all_chunk_documents()
        self.sparse_index.build(all_chunks, self.settings.sparse_top_k)

    def _get_reranker(self) -> CrossEncoderReranker:
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(self.settings.reranker_model)
        return self.reranker

    def _get_answer_generator(self) -> AnswerGenerator:
        if self.answer_generator is None:
            self.answer_generator = AnswerGenerator(self.settings)
        return self.answer_generator
