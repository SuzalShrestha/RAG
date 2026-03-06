from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import Settings
from app.utils.models import RetrievedChunk


class DenseIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.index_path = Path(settings.faiss_dir)
        self.vectorstore = None

    def build(self, documents: Iterable[Document]) -> None:
        documents = list(documents)
        if not documents:
            self.vectorstore = None
            return

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save_local(str(self.index_path))

    def load(self) -> bool:
        faiss_file = self.index_path / "index.faiss"
        pickle_file = self.index_path / "index.pkl"
        if not faiss_file.exists() or not pickle_file.exists():
            return False

        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        if self.vectorstore is None and not self.load():
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
