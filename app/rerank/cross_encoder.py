from __future__ import annotations

import logging
from typing import List, Optional

from app.utils.models import RetrievedChunk


LOGGER = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._disabled = False

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            model_kwargs = {"local_files_only": True}
            if self.device:
                model_kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **model_kwargs)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        if self._disabled:
            return chunks[:top_n] if top_n is not None else chunks

        pairs = [(query, chunk.text) for chunk in chunks]
        try:
            scores = self._get_model().predict(pairs)
        except Exception as error:
            self._disabled = True
            LOGGER.warning("Disabling reranker %s after failure: %s", self.model_name, error)
            return chunks[:top_n] if top_n is not None else chunks

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
