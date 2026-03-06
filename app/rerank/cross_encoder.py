from __future__ import annotations

from typing import List, Optional

from sentence_transformers import CrossEncoder

from app.utils.models import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._get_model().predict(pairs)

        rescored = []
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
            rescored.append(chunk)

        rescored.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else float("-inf"),
                item.fused_score,
            ),
            reverse=True,
        )

        if top_n is None:
            return rescored
        return rescored[:top_n]
