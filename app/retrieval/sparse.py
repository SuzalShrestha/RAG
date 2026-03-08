from __future__ import annotations

from typing import Iterable, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.utils.models import RetrievedChunk


class SparseIndex:
    def __init__(self) -> None:
        self.retriever = None
        self.chunk_count = 0

    def build(self, documents: Iterable[Document], k: int) -> None:
        documents = list(documents)
        if not documents:
            self.clear()
            return

        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = k
        self.chunk_count = len(documents)

    def clear(self) -> None:
        self.retriever = None
        self.chunk_count = 0

    def is_current(self, expected_chunk_count: int) -> bool:
        return self.retriever is not None and self.chunk_count == expected_chunk_count

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        if self.retriever is None:
            return []

        self.retriever.k = k
        results = self.retriever.invoke(query)
        chunks = []

        for rank, document in enumerate(results, start=1):
            metadata = dict(document.metadata)
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id")),
                    text=document.page_content,
                    metadata=metadata,
                    sparse_rank=rank,
                )
            )

        return chunks
