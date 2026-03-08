from __future__ import annotations

from typing import List, Optional

from app.config import Settings
from app.utils.models import RetrievedChunk


class PineconeReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        results = self._get_client().inference.rerank(
            model=self.settings.pinecone_rerank_model,
            query=query,
            documents=[{"text": chunk.text} for chunk in chunks],
            rank_fields=["text"],
            return_documents=False,
            top_n=top_n or len(chunks),
        )

        reranked = []
        for item in getattr(results, "data", []) or []:
            original = chunks[int(item.index)]
            original.rerank_score = float(item.score)
            reranked.append(original)
        return reranked

    def _get_client(self):
        if self._client is None:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.settings.pinecone_api_key)
        return self._client
