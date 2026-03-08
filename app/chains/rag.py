from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple

from app.config import Settings, get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import compute_checksum, load_file
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pinecone import PineconeRetrieval
from app.retrieval.sparse import SparseIndex
from app.storage.metadata_store import MetadataStore
from app.utils.runtime_checks import (
    groq_api_key_is_configured,
    list_ollama_models,
    missing_ollama_models,
    ollama_is_running,
    pinecone_api_key_is_configured,
)
from app.utils.models import AnswerResult, Citation, IndexingSummary, RetrievedChunk

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
        self._deduplicate_corpus()

    def index_paths(self, paths: Iterable[Path]) -> IndexingSummary:
        loaded_sets = []
        failed_files = []
        skipped_files = []
        for path in paths:
            raw_path = Path(path)
            try:
                checksum = compute_checksum(raw_path)
            except Exception as error:
                failed_files.append("{path}: {error}".format(path=raw_path, error=error))
                continue

            if self.store.has_checksum(checksum):
                skipped_files.append(
                    "{filename}: identical content is already indexed".format(
                        filename=raw_path.name,
                    )
                )
                continue

            try:
                loaded_set = load_file(raw_path, checksum=checksum)
            except Exception as error:
                failed_files.append("{path}: {error}".format(path=raw_path, error=error))
                continue

            loaded_sets.append(loaded_set)

        files_indexed = 0
        chunks_indexed = 0

        if self.settings.uses_pinecone_retrieval() and loaded_sets:
            self._ensure_pinecone_runtime()

        for loaded_set in loaded_sets:
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
                    continue

            self.store.replace_document_chunks(loaded_set, chunks)
            files_indexed += 1
            chunks_indexed += len(chunks)

        if self.settings.uses_pinecone_retrieval():
            return IndexingSummary(
                files_indexed=files_indexed,
                chunks_indexed=chunks_indexed,
                total_documents=self.store.document_count(),
                total_chunks=self.store.chunk_count(),
                failed_files=failed_files,
                skipped_files=skipped_files,
            )

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

        return IndexingSummary(
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
            total_documents=self.store.document_count(),
            total_chunks=self.store.chunk_count(),
            failed_files=failed_files,
            skipped_files=skipped_files,
        )

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            return []

        chunk_count = self.store.chunk_count()
        if chunk_count == 0:
            return []

        if self.settings.uses_pinecone_retrieval():
            self._ensure_pinecone_runtime()
            dense_results = self.retrieve_dense(normalized_question)
            sparse_results = self.retrieve_sparse(normalized_question)
            if self.settings.uses_dense_retrieval() and self.settings.uses_sparse_retrieval():
                return self.hybrid_retriever.combine(dense_results, sparse_results)
            if self.settings.uses_sparse_retrieval():
                return sparse_results
            if self.settings.uses_dense_retrieval():
                return dense_results
            return []

        uses_dense = self.settings.uses_dense_retrieval()
        uses_sparse = self.settings.uses_sparse_retrieval()
        if uses_dense:
            self._ensure_ollama_runtime([self.settings.embedding_model])
            self.dense_index.set_expected_chunk_count(chunk_count)
        if uses_sparse:
            self._ensure_sparse_index(chunk_count)

        if uses_dense and uses_sparse:
            return self.hybrid_retriever.retrieve(normalized_question)
        if uses_sparse:
            return self.sparse_index.search(normalized_question, self.settings.sparse_top_k)
        if uses_dense:
            return self.dense_index.search(normalized_question, self.settings.dense_top_k)
        return []

    def retrieve_dense(self, question: str) -> List[RetrievedChunk]:
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
            return self.pinecone_retrieval.search_dense(normalized_question, self.settings.dense_top_k)

        self._ensure_ollama_runtime([self.settings.embedding_model])
        self.dense_index.set_expected_chunk_count(chunk_count)
        return self.dense_index.search(normalized_question, self.settings.dense_top_k)

    def retrieve_sparse(self, question: str) -> List[RetrievedChunk]:
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
            return self.pinecone_retrieval.search_sparse(normalized_question, self.settings.sparse_top_k)

        self._ensure_sparse_index(chunk_count)
        return self.sparse_index.search(normalized_question, self.settings.sparse_top_k)

    def refresh_indexes(self) -> IndexingSummary:
        total_documents = self.store.document_count()
        total_chunks = self.store.chunk_count()
        if total_chunks == 0:
            if self.settings.uses_local_retrieval():
                self.dense_index.clear()
                self.sparse_index.clear()
            return IndexingSummary(
                files_indexed=0,
                chunks_indexed=0,
                total_documents=total_documents,
                total_chunks=total_chunks,
            )

        all_chunks = self.store.load_all_chunk_documents()
        if self.settings.uses_pinecone_retrieval():
            self._ensure_pinecone_runtime()
            self.pinecone_retrieval.upsert_documents(all_chunks)
            return IndexingSummary(
                files_indexed=0,
                chunks_indexed=0,
                total_documents=total_documents,
                total_chunks=total_chunks,
            )

        if self.settings.uses_dense_retrieval():
            self._ensure_ollama_runtime([self.settings.embedding_model])
            self.dense_index.build(all_chunks)
        else:
            self.dense_index.clear()

        if self.settings.uses_sparse_retrieval():
            self.sparse_index.build(all_chunks, self.settings.sparse_top_k)
        else:
            self.sparse_index.clear()
        return IndexingSummary(
            files_indexed=0,
            chunks_indexed=0,
            total_documents=total_documents,
            total_chunks=total_chunks,
        )

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

    def answer(self, question: str) -> AnswerResult:
        if self._is_small_talk(question):
            return AnswerResult(
                answer_markdown="Hi. Ask a question about the indexed documents and I will answer from the uploaded files.",
                abstained=True,
            )

        retrieved = self.retrieve(question)
        if not retrieved:
            return AnswerResult(
                answer_markdown="I don't know based on the uploaded documents.",
                abstained=True,
                retrieved_chunks=[],
            )

        selected_chunks = self.rerank_chunks(
            question,
            retrieved,
            top_n=self._answer_chunk_limit(),
        )
        if not self.settings.answer_uses_llm():
            return self._build_extractive_answer(question, selected_chunks, retrieved)

        self._ensure_answer_runtime()
        answer_generator = self._get_answer_generator()
        result = answer_generator.answer(question, selected_chunks)
        result.retrieved_chunks = retrieved
        return result

    def indexed_document_count(self) -> int:
        return self.store.document_count()

    def indexed_chunk_count(self) -> int:
        return self.store.chunk_count()

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
