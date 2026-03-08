from __future__ import annotations

import copy
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple

from app.config import Settings, get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import compute_checksum, load_file
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pinecone import PineconeRetrieval
from app.retrieval.sparse import SparseIndex
from app.storage.metadata_store import MetadataStore
from app.utils.telemetry import StructuredLogger
from app.utils.runtime_checks import (
    groq_api_key_is_configured,
    list_ollama_models,
    missing_ollama_models,
    ollama_is_running,
    pinecone_api_key_is_configured,
)
from app.utils.models import (
    AnswerResult,
    Citation,
    DocumentRecord,
    IndexingSummary,
    IndexProgress,
    OperationMetrics,
    RetrievalFilters,
    RetrievedChunk,
)

if TYPE_CHECKING:
    from app.generation.answer_chain import AnswerGenerator
    from app.rerank.cross_encoder import CrossEncoderReranker
    from app.rerank.pinecone import PineconeReranker


SMALL_TALK_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|sup|hola|namaste|good morning|good afternoon|good evening)[!.? ]*$",
    re.IGNORECASE,
)


class RAGPipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.store = MetadataStore(self.settings.metadata_db_path)
        self.dense_index = DenseIndex(self.settings)
        self.sparse_index = SparseIndex()
        self.pinecone_retrieval = PineconeRetrieval(self.settings)
        self.hybrid_retriever = HybridRetriever(
            dense_index=self.dense_index,
            sparse_index=self.sparse_index,
            settings=self.settings,
        )
        self.reranker: Optional[CrossEncoderReranker] = None
        self.pinecone_reranker: Optional[PineconeReranker] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self._validated_model_sets: Set[Tuple[str, ...]] = set()
        self.logger = StructuredLogger(self.settings)
        self._retrieval_cache: "OrderedDict[Tuple[str, Tuple[Tuple[str, ...], Tuple[str, ...]]], List[RetrievedChunk]]" = OrderedDict()
        self._last_retrieval_cache_hit = False
        self._deduplicate_corpus()

    def index_paths(
        self,
        paths: Iterable[Path],
        collection_name: Optional[str] = None,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
    ) -> IndexingSummary:
        started_at = time.perf_counter()
        path_list = [Path(path) for path in paths]
        active_collection = self._normalize_collection_name(collection_name)
        loaded_sets = []
        failed_files = []
        skipped_files = []
        total_paths = len(path_list)
        self._emit_progress(
            progress_callback,
            stage="loading",
            current=0,
            total=total_paths,
            message="Scanning files for indexing.",
        )

        for position, path in enumerate(path_list, start=1):
            raw_path = Path(path)
            try:
                checksum = compute_checksum(raw_path)
            except Exception as error:
                failed_files.append("{path}: {error}".format(path=raw_path, error=error))
                self._emit_progress(
                    progress_callback,
                    stage="loading",
                    current=position,
                    total=total_paths,
                    message="Failed to read {filename}.".format(filename=raw_path.name),
                )
                continue

            if self.store.has_checksum(checksum):
                skipped_files.append(
                    "{filename}: identical content is already indexed".format(
                        filename=raw_path.name,
                    )
                )
                self._emit_progress(
                    progress_callback,
                    stage="loading",
                    current=position,
                    total=total_paths,
                    message="Skipped {filename}; identical content already exists.".format(
                        filename=raw_path.name,
                    ),
                )
                continue

            try:
                loaded_set = load_file(
                    raw_path,
                    checksum=checksum,
                    collection_name=active_collection,
                    enable_ocr=self.settings.enable_ocr,
                    ocr_language=self.settings.ocr_language,
                )
            except Exception as error:
                failed_files.append("{path}: {error}".format(path=raw_path, error=error))
                self._emit_progress(
                    progress_callback,
                    stage="loading",
                    current=position,
                    total=total_paths,
                    message="Failed to extract {filename}.".format(filename=raw_path.name),
                )
                continue

            loaded_sets.append(loaded_set)
            self._emit_progress(
                progress_callback,
                stage="loading",
                current=position,
                total=total_paths,
                message="Prepared {filename} for indexing.".format(filename=loaded_set.filename),
            )

        files_indexed = 0
        chunks_indexed = 0

        if self.settings.uses_pinecone_retrieval() and loaded_sets:
            self._ensure_pinecone_runtime()

        for position, loaded_set in enumerate(loaded_sets, start=1):
            self._emit_progress(
                progress_callback,
                stage="indexing",
                current=position - 1,
                total=len(loaded_sets),
                message="Indexing {filename}.".format(filename=loaded_set.filename),
            )
            chunks = chunk_documents(
                loaded_set.documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for chunk in chunks:
                chunk.metadata["checksum"] = loaded_set.checksum
            if self.settings.uses_pinecone_retrieval():
                try:
                    self.pinecone_retrieval.upsert_documents(chunks)
                except Exception as error:
                    failed_files.append(
                        "{filename}: {error}".format(
                            filename=loaded_set.filename,
                            error=error,
                        )
                    )
                    self._emit_progress(
                        progress_callback,
                        stage="indexing",
                        current=position,
                        total=len(loaded_sets),
                        message="Failed to index {filename}.".format(filename=loaded_set.filename),
                    )
                    continue

            self.store.replace_document_chunks(loaded_set, chunks)
            files_indexed += 1
            chunks_indexed += len(chunks)
            self._emit_progress(
                progress_callback,
                stage="indexing",
                current=position,
                total=len(loaded_sets),
                message="Indexed {filename}.".format(filename=loaded_set.filename),
            )

        if self.settings.uses_pinecone_retrieval():
            summary = IndexingSummary(
                files_indexed=files_indexed,
                chunks_indexed=chunks_indexed,
                total_documents=self.store.document_count(),
                total_chunks=self.store.chunk_count(),
                failed_files=failed_files,
                skipped_files=skipped_files,
                duration_seconds=time.perf_counter() - started_at,
                collection_name=active_collection,
            )
            if files_indexed:
                self._clear_retrieval_cache()
            self.logger.log_event("index_paths", summary)
            return summary

        should_refresh_indexes = bool(loaded_sets)
        if not should_refresh_indexes:
            total_chunks = self.store.chunk_count()
            if total_chunks > 0:
                dense_ready = True
                if self.settings.uses_dense_retrieval():
                    self.dense_index.set_expected_chunk_count(total_chunks)
                    dense_ready = self.dense_index.load(expected_chunk_count=total_chunks)
                sparse_ready = (not self.settings.uses_sparse_retrieval()) or self.sparse_index.is_current(total_chunks)
                should_refresh_indexes = not (dense_ready and sparse_ready)

        if should_refresh_indexes:
            self._emit_progress(
                progress_callback,
                stage="refreshing",
                current=0,
                total=1,
                message="Refreshing local retrieval indexes.",
            )
            all_chunks = self.store.load_all_chunk_documents()
            if self.settings.uses_dense_retrieval():
                self._ensure_ollama_runtime([self.settings.embedding_model])
                self.dense_index.build(all_chunks)
            else:
                self.dense_index.clear()

            if self.settings.uses_sparse_retrieval():
                self.sparse_index.build(all_chunks, self.settings.sparse_top_k)
            else:
                self.sparse_index.clear()
            self._emit_progress(
                progress_callback,
                stage="refreshing",
                current=1,
                total=1,
                message="Local retrieval indexes refreshed.",
            )

        summary = IndexingSummary(
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
            total_documents=self.store.document_count(),
            total_chunks=self.store.chunk_count(),
            failed_files=failed_files,
            skipped_files=skipped_files,
            duration_seconds=time.perf_counter() - started_at,
            collection_name=active_collection,
        )
        if files_indexed:
            self._clear_retrieval_cache()
        self.logger.log_event("index_paths", summary)
        return summary

    def retrieve(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            return []

        chunk_count = self.store.chunk_count()
        if chunk_count == 0:
            return []

        active_filters = (filters or RetrievalFilters()).normalized()
        cache_key = (normalized_question, active_filters.cache_key())
        cached_chunks = self._get_cached_retrieval(cache_key)
        if cached_chunks is not None:
            return cached_chunks

        if self.settings.uses_pinecone_retrieval():
            self._ensure_pinecone_runtime()
            dense_results = self.retrieve_dense(normalized_question, filters=active_filters)
            sparse_results = self.retrieve_sparse(normalized_question, filters=active_filters)
            if self.settings.uses_dense_retrieval() and self.settings.uses_sparse_retrieval():
                fused = self.hybrid_retriever.combine(dense_results, sparse_results)
                self._store_retrieval_cache(cache_key, fused)
                return copy.deepcopy(fused)
            if self.settings.uses_sparse_retrieval():
                self._store_retrieval_cache(cache_key, sparse_results)
                return copy.deepcopy(sparse_results)
            if self.settings.uses_dense_retrieval():
                self._store_retrieval_cache(cache_key, dense_results)
                return copy.deepcopy(dense_results)
            return []

        uses_dense = self.settings.uses_dense_retrieval()
        uses_sparse = self.settings.uses_sparse_retrieval()
        if uses_dense:
            self._ensure_ollama_runtime([self.settings.embedding_model])
            self.dense_index.set_expected_chunk_count(chunk_count)
        if uses_sparse:
            self._ensure_sparse_index(chunk_count)

        if uses_dense and uses_sparse:
            retrieved = self.hybrid_retriever.retrieve(normalized_question)
            filtered = self._apply_filters(retrieved, active_filters)
            self._store_retrieval_cache(cache_key, filtered)
            return copy.deepcopy(filtered)
        if uses_sparse:
            retrieved = self.sparse_index.search(normalized_question, self.settings.sparse_top_k)
            filtered = self._apply_filters(retrieved, active_filters)
            self._store_retrieval_cache(cache_key, filtered)
            return copy.deepcopy(filtered)
        if uses_dense:
            retrieved = self.dense_index.search(normalized_question, self.settings.dense_top_k)
            filtered = self._apply_filters(retrieved, active_filters)
            self._store_retrieval_cache(cache_key, filtered)
            return copy.deepcopy(filtered)
        return []

    def retrieve_dense(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            return []

        chunk_count = self.store.chunk_count()
        if chunk_count == 0:
            return []

        if not self.settings.uses_dense_retrieval():
            return []

        if self.settings.uses_pinecone_retrieval():
            self._ensure_pinecone_runtime()
            return self._apply_filters(
                self.pinecone_retrieval.search_dense(normalized_question, self.settings.dense_top_k),
                filters,
            )

        self._ensure_ollama_runtime([self.settings.embedding_model])
        self.dense_index.set_expected_chunk_count(chunk_count)
        return self._apply_filters(
            self.dense_index.search(normalized_question, self.settings.dense_top_k),
            filters,
        )

    def retrieve_sparse(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            return []

        chunk_count = self.store.chunk_count()
        if chunk_count == 0:
            return []

        if not self.settings.uses_sparse_retrieval():
            return []

        if self.settings.uses_pinecone_retrieval():
            self._ensure_pinecone_runtime()
            return self._apply_filters(
                self.pinecone_retrieval.search_sparse(normalized_question, self.settings.sparse_top_k),
                filters,
            )

        self._ensure_sparse_index(chunk_count)
        return self._apply_filters(
            self.sparse_index.search(normalized_question, self.settings.sparse_top_k),
            filters,
        )

    def refresh_indexes(
        self,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
    ) -> IndexingSummary:
        started_at = time.perf_counter()
        total_documents = self.store.document_count()
        total_chunks = self.store.chunk_count()
        self._emit_progress(
            progress_callback,
            stage="refreshing",
            current=0,
            total=max(1, total_chunks),
            message="Refreshing retrieval indexes from stored chunks.",
        )
        if total_chunks == 0:
            if self.settings.uses_local_retrieval():
                self.dense_index.clear()
                self.sparse_index.clear()
            summary = IndexingSummary(
                files_indexed=0,
                chunks_indexed=0,
                total_documents=total_documents,
                total_chunks=total_chunks,
                duration_seconds=time.perf_counter() - started_at,
                collection_name=self.settings.default_collection_name,
            )
            self.logger.log_event("refresh_indexes", summary)
            return summary

        all_chunks = self.store.load_all_chunk_documents()
        if self.settings.uses_pinecone_retrieval():
            self._ensure_pinecone_runtime()
            self.pinecone_retrieval.upsert_documents(all_chunks)
            self._emit_progress(
                progress_callback,
                stage="refreshing",
                current=total_chunks,
                total=total_chunks,
                message="Cloud retrieval indexes refreshed.",
            )
            summary = IndexingSummary(
                files_indexed=0,
                chunks_indexed=0,
                total_documents=total_documents,
                total_chunks=total_chunks,
                duration_seconds=time.perf_counter() - started_at,
                collection_name=self.settings.default_collection_name,
            )
            self._clear_retrieval_cache()
            self.logger.log_event("refresh_indexes", summary)
            return summary

        if self.settings.uses_dense_retrieval():
            self._ensure_ollama_runtime([self.settings.embedding_model])
            self.dense_index.build(all_chunks)
        else:
            self.dense_index.clear()

        if self.settings.uses_sparse_retrieval():
            self.sparse_index.build(all_chunks, self.settings.sparse_top_k)
        else:
            self.sparse_index.clear()
        self._emit_progress(
            progress_callback,
            stage="refreshing",
            current=total_chunks,
            total=total_chunks,
            message="Local retrieval indexes refreshed.",
        )
        summary = IndexingSummary(
            files_indexed=0,
            chunks_indexed=0,
            total_documents=total_documents,
            total_chunks=total_chunks,
            duration_seconds=time.perf_counter() - started_at,
            collection_name=self.settings.default_collection_name,
        )
        self._clear_retrieval_cache()
        self.logger.log_event("refresh_indexes", summary)
        return summary

    def rerank_chunks(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not self.settings.uses_reranker():
            return chunks[:top_n] if top_n is not None else chunks
        if self.settings.uses_pinecone_reranker():
            self._ensure_pinecone_runtime()
            reranker = self._get_pinecone_reranker()
            return reranker.rerank(question, chunks, top_n=top_n or self.settings.final_context_k)
        reranker = self._get_reranker()
        return reranker.rerank(question, chunks, top_n=top_n or self.settings.final_context_k)

    def answer(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
    ) -> AnswerResult:
        total_started_at = time.perf_counter()
        if self._is_small_talk(question):
            return AnswerResult(
                answer_markdown="Hi. Ask a question about the indexed documents and I will answer from the uploaded files.",
                abstained=True,
            )

        retrieval_started_at = time.perf_counter()
        active_filters = (filters or RetrievalFilters()).normalized()
        retrieved = self.retrieve(question, filters=active_filters)
        retrieval_seconds = time.perf_counter() - retrieval_started_at
        if not retrieved:
            result = AnswerResult(
                answer_markdown="I don't know based on the uploaded documents.",
                abstained=True,
                retrieved_chunks=[],
            )
            result.metrics = OperationMetrics(
                retrieval_seconds=retrieval_seconds,
                total_seconds=time.perf_counter() - total_started_at,
                retrieved_chunks=0,
                used_chunks=0,
            )
            self.logger.log_event(
                "answer",
                {
                    "question": question,
                    "filters": active_filters.as_dict(),
                    "abstained": True,
                    "metrics": result.metrics,
                },
            )
            return result

        rerank_started_at = time.perf_counter()
        selected_chunks = self.rerank_chunks(
            question,
            retrieved,
            top_n=self._answer_chunk_limit(),
        )
        rerank_seconds = time.perf_counter() - rerank_started_at
        if not self.settings.answer_uses_llm():
            result = self._build_extractive_answer(question, selected_chunks, retrieved)
            result.metrics = OperationMetrics(
                retrieval_seconds=retrieval_seconds,
                rerank_seconds=rerank_seconds,
                generation_seconds=0.0,
                total_seconds=time.perf_counter() - total_started_at,
                retrieved_chunks=len(retrieved),
                used_chunks=len(selected_chunks),
            )
            self.logger.log_event(
                "answer",
                {
                    "question": question,
                    "filters": active_filters.as_dict(),
                    "abstained": result.abstained,
                    "metrics": result.metrics,
                },
            )
            return result

        generation_started_at = time.perf_counter()
        self._ensure_answer_runtime()
        answer_generator = self._get_answer_generator()
        result = answer_generator.answer(question, selected_chunks)
        generation_seconds = time.perf_counter() - generation_started_at
        result.retrieved_chunks = retrieved
        result.metrics = OperationMetrics(
            retrieval_seconds=retrieval_seconds,
            rerank_seconds=rerank_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - total_started_at,
            retrieved_chunks=len(retrieved),
            used_chunks=len(selected_chunks),
        )
        self.logger.log_event(
            "answer",
            {
                "question": question,
                "filters": active_filters.as_dict(),
                "abstained": result.abstained,
                "metrics": result.metrics,
            },
        )
        return result

    def indexed_document_count(self) -> int:
        return self.store.document_count()

    def indexed_chunk_count(self) -> int:
        return self.store.chunk_count()

    def list_documents(self) -> List[DocumentRecord]:
        return self.store.list_documents()

    def list_collections(self) -> List[str]:
        return self.store.list_collections()

    def recent_events(self, limit: int = 20) -> List[dict]:
        return self.logger.read_recent(limit)

    def _ensure_sparse_index(self, chunk_count: int) -> None:
        if not self.settings.uses_sparse_retrieval():
            self.sparse_index.clear()
            return

        if self.sparse_index.is_current(chunk_count):
            return

        all_chunks = self.store.load_all_chunk_documents()
        self.sparse_index.build(all_chunks, self.settings.sparse_top_k)

    def _get_reranker(self) -> CrossEncoderReranker:
        if self.reranker is None:
            from app.rerank.cross_encoder import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(
                self.settings.reranker_model,
                device=self.settings.reranker_device,
            )
        return self.reranker

    def _get_pinecone_reranker(self) -> PineconeReranker:
        if self.pinecone_reranker is None:
            from app.rerank.pinecone import PineconeReranker

            self.pinecone_reranker = PineconeReranker(self.settings)
        return self.pinecone_reranker

    def _get_answer_generator(self) -> AnswerGenerator:
        if self.answer_generator is None:
            from app.generation.answer_chain import AnswerGenerator

            self.answer_generator = AnswerGenerator(self.settings)
        return self.answer_generator

    def _ensure_pinecone_runtime(self) -> None:
        if pinecone_api_key_is_configured(self.settings.pinecone_api_key):
            return
        raise RuntimeError("Pinecone API key is missing. Set `RAG_PINECONE_API_KEY` in `.env`.")

    def _ensure_answer_runtime(self) -> None:
        if not self.settings.answer_uses_llm():
            return

        if self.settings.answer_uses_groq():
            if groq_api_key_is_configured(self.settings.groq_api_key):
                return
            raise RuntimeError("Groq API key is missing. Set `RAG_GROQ_API_KEY` in `.env`.")

        if self.settings.answer_uses_ollama():
            self._ensure_ollama_runtime([self.settings.chat_model])
            return

        raise RuntimeError(
            "Unsupported LLM provider: {provider}".format(
                provider=self.settings.normalized_llm_provider(),
            )
        )

    def _ensure_ollama_runtime(self, required_models: List[str]) -> None:
        model_key = tuple(sorted(set(required_models)))
        if model_key in self._validated_model_sets:
            return

        if not ollama_is_running(self.settings.ollama_base_url):
            raise RuntimeError(
                "Ollama is not reachable at {url}. Start `ollama serve` and try again.".format(
                    url=self.settings.ollama_base_url,
                )
            )

        missing_models = missing_ollama_models(self.settings.ollama_base_url, required_models)
        if not missing_models:
            self._validated_model_sets.add(model_key)
            return

        installed_models = list_ollama_models(self.settings.ollama_base_url)
        installed_display = ", ".join(installed_models) if installed_models else "none"
        raise RuntimeError(
            "Missing Ollama model(s): {missing}. Installed models: {installed}. Update `.env` or run `ollama pull ...`.".format(
                missing=", ".join(missing_models),
                installed=installed_display,
            )
        )

    def _is_small_talk(self, question: str) -> bool:
        normalized = question.strip()
        if not normalized:
            return False
        return bool(SMALL_TALK_PATTERN.fullmatch(normalized))

    def _deduplicate_corpus(self) -> None:
        removed_documents = self.store.remove_duplicate_documents()
        if removed_documents:
            self.dense_index.clear()
            self.sparse_index.clear()
            self._clear_retrieval_cache()

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[IndexProgress], None]],
        stage: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            IndexProgress(
                stage=stage,
                current=current,
                total=total,
                message=message,
            )
        )

    def _normalize_collection_name(self, collection_name: Optional[str]) -> str:
        normalized = (collection_name or self.settings.default_collection_name).strip()
        return normalized or self.settings.default_collection_name

    def _apply_filters(
        self,
        chunks: List[RetrievedChunk],
        filters: Optional[RetrievalFilters],
    ) -> List[RetrievedChunk]:
        if filters is None:
            return chunks

        active_filters = filters.normalized()
        if not active_filters.filenames and not active_filters.collection_names:
            return chunks

        filenames = {name.lower() for name in active_filters.filenames}
        collections = {name.lower() for name in active_filters.collection_names}
        filtered_chunks = []
        for chunk in chunks:
            filename = str(chunk.metadata.get("filename", "")).lower()
            collection_name = str(chunk.metadata.get("collection_name", self.settings.default_collection_name)).lower()
            if filenames and filename not in filenames:
                continue
            if collections and collection_name not in collections:
                continue
            filtered_chunks.append(chunk)
        return filtered_chunks

    def _get_cached_retrieval(
        self,
        cache_key: Tuple[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
    ) -> Optional[List[RetrievedChunk]]:
        cached = self._retrieval_cache.get(cache_key)
        if cached is None:
            return None
        self._retrieval_cache.move_to_end(cache_key)
        return copy.deepcopy(cached)

    def _store_retrieval_cache(
        self,
        cache_key: Tuple[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
        chunks: List[RetrievedChunk],
    ) -> None:
        if self.settings.retrieval_cache_size <= 0:
            return
        self._retrieval_cache[cache_key] = copy.deepcopy(chunks)
        self._retrieval_cache.move_to_end(cache_key)
        while len(self._retrieval_cache) > self.settings.retrieval_cache_size:
            self._retrieval_cache.popitem(last=False)

    def _clear_retrieval_cache(self) -> None:
        self._retrieval_cache.clear()

    def _answer_chunk_limit(self) -> int:
        if self.settings.answer_uses_llm():
            return max(1, self.settings.final_context_k)
        return max(1, self.settings.extractive_context_k)

    def _build_extractive_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        retrieved_chunks: List[RetrievedChunk],
    ) -> AnswerResult:
        citations = self._build_citations(chunks)
        bullets = []
        for citation, chunk in zip(citations, chunks):
            snippet = self._best_snippet(question, chunk.text)
            bullets.append("[{label}] {snippet}".format(label=citation.label, snippet=snippet))

        answer_text = "Relevant passages:\n\n- {bullets}".format(
            bullets="\n- ".join(bullets),
        )
        return AnswerResult(
            answer_markdown=answer_text,
            citations=citations,
            used_chunks=chunks,
            retrieved_chunks=retrieved_chunks,
            abstained=False,
            raw_response={"mode": "extractive"},
        )

    def _build_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        citations = []
        for index, chunk in enumerate(chunks, start=1):
            citations.append(
                Citation(
                    label="C{index}".format(index=index),
                    chunk_id=chunk.chunk_id,
                    filename=str(chunk.metadata.get("filename", "unknown")),
                    page_number=chunk.metadata.get("page_number"),
                    section_heading=chunk.metadata.get("section_heading"),
                    excerpt=chunk.text[:280],
                )
            )
        return citations

    def _best_snippet(self, question: str, text: str) -> str:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return ""

        candidates = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", normalized_text) if segment.strip()]
        if not candidates:
            return normalized_text[: self.settings.extractive_max_chars]

        query_terms = {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) >= 3
        }

        scored_candidates = []
        for index, candidate in enumerate(candidates):
            candidate_terms = set(re.findall(r"[a-z0-9]+", candidate.lower()))
            overlap = len(query_terms.intersection(candidate_terms))
            scored_candidates.append((overlap, -index, candidate))

        scored_candidates.sort(reverse=True)
        best = scored_candidates[0][2]
        if len(best) <= self.settings.extractive_max_chars:
            return best
        return "{snippet}...".format(
            snippet=best[: self.settings.extractive_max_chars].rstrip(),
        )
