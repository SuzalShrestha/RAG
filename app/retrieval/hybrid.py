from __future__ import annotations

from typing import Dict, List

from app.config import Settings
from app.retrieval.dense import DenseIndex
from app.retrieval.sparse import SparseIndex
from app.utils.models import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        settings: Settings,
    ) -> None:
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.settings = settings

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        dense_results = []
        sparse_results = []
        if self.settings.uses_dense_retrieval():
            dense_results = self.dense_index.search(query, self.settings.dense_top_k)
        if self.settings.uses_sparse_retrieval():
            sparse_results = self.sparse_index.search(query, self.settings.sparse_top_k)

        return self.combine(dense_results, sparse_results)

    def combine(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        
        merged = {}
        self._merge_results(
            merged,
            dense_results,
            weight=self.settings.dense_weight,
            key_name="dense_rank",
        )
        self._merge_results(
            merged,
            sparse_results,
            weight=self.settings.sparse_weight,
            key_name="sparse_rank",
        )

        ranked = sorted(
            merged.values(),
            key=lambda chunk: chunk.fused_score,
            reverse=True,
        )
        return ranked[: self.settings.fused_top_k]

    def _merge_results(
        self,
        merged: Dict[str, RetrievedChunk],
        results: List[RetrievedChunk],
        weight: float,
        key_name: str,
    ) -> None:
        for rank, chunk in enumerate(results, start=1):
            score = weight / float(self.settings.rrf_k + rank)
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
                existing = merged[chunk.chunk_id]

            existing.fused_score += score
            if key_name == "dense_rank":
                existing.dense_rank = rank
                existing.dense_score = chunk.dense_score
            elif key_name == "sparse_rank":
                existing.sparse_rank = rank
