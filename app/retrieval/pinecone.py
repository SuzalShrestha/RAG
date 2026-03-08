from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from app.config import Settings
from app.utils.models import RetrievedChunk


CHUNK_TEXT_FIELD = "chunk_text"
PINECONE_RETURN_FIELDS = [
    CHUNK_TEXT_FIELD,
    "doc_id",
    "filename",
    "page_number",
    "section_heading",
    "checksum",
    "source_path",
    "collection_name",
]
UPSERT_WINDOW_SECONDS = 60.0


class PineconeRetrieval:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._indexes: Dict[str, Any] = {}
        self._recent_upsert_usage: Dict[str, Deque[Tuple[float, int]]] = {}

    def upsert_documents(self, documents: Iterable[Document]) -> None:
        documents = list(documents)
        if not documents:
            return

        if self.settings.uses_dense_retrieval():
            self._upsert(
                index_name=self.settings.pinecone_dense_index,
                model_name=self.settings.pinecone_dense_model,
                documents=documents,
            )

        if self.settings.uses_sparse_retrieval():
            self._upsert(
                index_name=self.settings.pinecone_sparse_index,
                model_name=self.settings.pinecone_sparse_model,
                documents=documents,
            )

    def search_dense(self, query: str, k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self.settings.uses_dense_retrieval():
            return []
        response = self._search(
            index_name=self.settings.pinecone_dense_index,
            model_name=self.settings.pinecone_dense_model,
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )
        return self._hits_to_chunks(response, score_kind="dense")

    def search_sparse(self, query: str, k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self.settings.uses_sparse_retrieval():
            return []
        response = self._search(
            index_name=self.settings.pinecone_sparse_index,
            model_name=self.settings.pinecone_sparse_model,
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )
        return self._hits_to_chunks(response, score_kind="sparse")

    def ensure_remote_indexes(self) -> None:
        if self.settings.uses_dense_retrieval():
            self._get_index(self.settings.pinecone_dense_index, self.settings.pinecone_dense_model)
        if self.settings.uses_sparse_retrieval():
            self._get_index(self.settings.pinecone_sparse_index, self.settings.pinecone_sparse_model)

    def _get_client(self):
        if self._client is None:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.settings.pinecone_api_key)
        return self._client

    def _get_index(self, index_name: str, model_name: str):
        index = self._indexes.get(index_name)
        if index is not None:
            return index

        client = self._get_client()
        if not client.has_index(index_name):
            client.create_index_for_model(
                name=index_name,
                cloud=self.settings.pinecone_cloud,
                region=self.settings.pinecone_region,
                embed={
                    "model": model_name,
                    "field_map": {
                        "text": CHUNK_TEXT_FIELD,
                    },
                },
                timeout=300,
            )

        description = client.describe_index(index_name)
        index = client.Index(host=description.host)
        self._indexes[index_name] = index
        return index

    def _upsert(self, index_name: str, model_name: str, documents: List[Document]) -> None:
        index = self._get_index(index_name, model_name)
        for batch, estimated_tokens in self._iter_document_batches(documents):
            records = [self._document_to_record(document) for document in batch]
            self._wait_for_upsert_budget(model_name, estimated_tokens)
            self._upsert_records_with_retries(index, records, model_name, estimated_tokens)
            self._record_upsert_usage(model_name, estimated_tokens)

    def _search(
        self,
        index_name: str,
        model_name: str,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        index = self._get_index(index_name, model_name)
        search_query = {
            "inputs": {
                "text": query,
            },
            "top_k": k,
        }
        if metadata_filter:
            search_query["filter"] = metadata_filter
        return index.search(
            namespace=self.settings.pinecone_namespace,
            query=search_query,
            fields=PINECONE_RETURN_FIELDS,
        )

    def _document_to_record(self, document: Document) -> Dict[str, Any]:
        metadata = dict(document.metadata)
        record = {
            "_id": str(metadata.get("chunk_id")),
            CHUNK_TEXT_FIELD: document.page_content,
        }
        optional_fields = {
            "doc_id": metadata.get("doc_id"),
            "filename": metadata.get("filename"),
            "page_number": metadata.get("page_number"),
            "section_heading": metadata.get("section_heading"),
            "checksum": metadata.get("checksum"),
            "source_path": metadata.get("source_path"),
            "collection_name": metadata.get("collection_name"),
        }
        for key, value in optional_fields.items():
            normalized = self._normalize_record_value(value)
            if normalized is not None:
                record[key] = normalized
        return record

    @staticmethod
    def _normalize_record_value(value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, set)):
            normalized_values = [str(item) for item in value if item is not None]
            return normalized_values or None
        return str(value)

    def _iter_document_batches(
        self,
        documents: List[Document],
    ) -> Iterator[Tuple[List[Document], int]]:
        max_records = max(1, self.settings.pinecone_upsert_batch_size)
        max_tokens = max(1, self.settings.pinecone_upsert_max_batch_tokens)
        batch: List[Document] = []
        batch_tokens = 0

        for document in documents:
            document_tokens = self._estimate_text_tokens(document.page_content)
            would_exceed_records = len(batch) >= max_records
            would_exceed_tokens = bool(batch) and batch_tokens + document_tokens > max_tokens
            if would_exceed_records or would_exceed_tokens:
                yield batch, batch_tokens
                batch = []
                batch_tokens = 0

            batch.append(document)
            batch_tokens += document_tokens

        if batch:
            yield batch, batch_tokens

    def _wait_for_upsert_budget(self, model_name: str, batch_tokens: int) -> None:
        budget = max(1, self.settings.pinecone_upsert_tokens_per_minute)
        effective_budget = max(budget, batch_tokens)
        usage = self._recent_upsert_usage.setdefault(model_name, deque())

        while True:
            now = time.monotonic()
            self._trim_usage(usage, now)
            used_tokens = sum(tokens for _, tokens in usage)
            if used_tokens + batch_tokens <= effective_budget:
                return

            oldest_timestamp, _ = usage[0]
            sleep_seconds = max(0.5, UPSERT_WINDOW_SECONDS - (now - oldest_timestamp) + 0.1)
            time.sleep(sleep_seconds)

    def _record_upsert_usage(self, model_name: str, batch_tokens: int) -> None:
        now = time.monotonic()
        usage = self._recent_upsert_usage.setdefault(model_name, deque())
        self._trim_usage(usage, now)
        usage.append((now, batch_tokens))

    def _upsert_records_with_retries(
        self,
        index: Any,
        records: List[Dict[str, Any]],
        model_name: str,
        batch_tokens: int,
    ) -> None:
        max_attempts = max(1, self.settings.pinecone_upsert_retry_attempts)
        for attempt in range(max_attempts):
            try:
                index.upsert_records(self.settings.pinecone_namespace, records)
                return
            except Exception as error:
                if not self._is_retryable_upsert_error(error) or attempt == max_attempts - 1:
                    raise
                time.sleep(self._retry_delay_seconds(batch_tokens, attempt))
                self._wait_for_upsert_budget(model_name, batch_tokens)

    def _retry_delay_seconds(self, batch_tokens: int, attempt: int) -> float:
        budget = max(1, self.settings.pinecone_upsert_tokens_per_minute)
        proportional_delay = (batch_tokens / budget) * UPSERT_WINDOW_SECONDS
        base_delay = max(self.settings.pinecone_upsert_retry_base_delay_seconds, proportional_delay)
        return min(UPSERT_WINDOW_SECONDS, base_delay * (attempt + 1))

    @staticmethod
    def _trim_usage(usage: Deque[Tuple[float, int]], now: float) -> None:
        while usage and now - usage[0][0] >= UPSERT_WINDOW_SECONDS:
            usage.popleft()

    @staticmethod
    def _is_retryable_upsert_error(error: Exception) -> bool:
        status = str(getattr(error, "status", "") or "")
        message = str(error)
        return status in {"429", "503"} or "RESOURCE_EXHAUSTED" in message or "Too Many Requests" in message

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        normalized = text or ""
        return max(1, math.ceil(len(normalized) / 4))

    @staticmethod
    def _hits_to_chunks(response: Any, score_kind: str) -> List[RetrievedChunk]:
        hits = getattr(getattr(response, "result", None), "hits", []) or []
        chunks = []

        for rank, hit in enumerate(hits, start=1):
            fields = dict(getattr(hit, "fields", {}) or {})
            text = str(fields.pop(CHUNK_TEXT_FIELD, "") or "")
            chunk = RetrievedChunk(
                chunk_id=str(getattr(hit, "_id", "")),
                text=text,
                metadata=fields,
            )
            score = float(getattr(hit, "_score", 0.0))
            if score_kind == "dense":
                chunk.dense_score = score
                chunk.dense_rank = rank
            elif score_kind == "sparse":
                chunk.sparse_rank = rank
            chunks.append(chunk)

        return chunks
