from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import Settings
from app.utils.models import RetrievedChunk


class DenseIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = None
        self.index_path = Path(settings.faiss_dir)
        self.metadata_path = self.index_path / "index_metadata.json"
        self.expected_chunk_count: Optional[int] = None
        self.vectorstore: Any = None

    def _get_embeddings(self):
        if self.embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            self.embeddings = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_base_url,
            )
        return self.embeddings

    def build(self, documents: Iterable[Document]) -> None:
        documents = list(documents)
        if not documents:
            self.clear()
            return

        self.vectorstore = FAISS.from_documents(documents, self._get_embeddings())
        self.vectorstore.save_local(str(self.index_path))
        self.expected_chunk_count = len(documents)
        self.metadata_path.write_text(
            json.dumps({"chunk_count": len(documents)}, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.vectorstore = None
        self.expected_chunk_count = None
        for path in (
            self.index_path / "index.faiss",
            self.index_path / "index.pkl",
            self.metadata_path,
        ):
            if path.exists():
                path.unlink()

    def set_expected_chunk_count(self, chunk_count: Optional[int]) -> None:
        self.expected_chunk_count = chunk_count

    def _stored_chunk_count(self) -> Optional[int]:
        if not self.metadata_path.exists():
            return None

        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return int(payload.get("chunk_count")) if payload.get("chunk_count") is not None else None

    def load(self, expected_chunk_count: Optional[int] = None) -> bool:
        faiss_file = self.index_path / "index.faiss"
        pickle_file = self.index_path / "index.pkl"
        if not faiss_file.exists() or not pickle_file.exists():
            return False

        expected = self.expected_chunk_count if expected_chunk_count is None else expected_chunk_count
        if expected is not None and self._stored_chunk_count() != expected:
            return False

        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self._get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return True

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        if self.vectorstore is None and not self.load():
            return []

        if self.expected_chunk_count is not None and self._stored_chunk_count() != self.expected_chunk_count:
            self.vectorstore = None
            return []

        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        chunks = []

        for rank, (document, distance) in enumerate(results, start=1):
            metadata = dict(document.metadata)
            similarity = 1.0 / (1.0 + float(distance))
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id")),
                    text=document.page_content,
                    metadata=metadata,
                    dense_score=similarity,
                    dense_rank=rank,
                )
            )

        return chunks
