from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_provider: str = "groq"
    groq_api_key: str = ""
    chat_model: str = "llama-3.1-8b-instant"
    retrieval_provider: str = "pinecone"
    rerank_provider: str = "pinecone"
    pinecone_api_key: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "default"
    pinecone_dense_index: str = "rag-dense"
    pinecone_sparse_index: str = "rag-sparse"
    pinecone_dense_model: str = "multilingual-e5-large"
    pinecone_sparse_model: str = "pinecone-sparse-english-v0"
    pinecone_rerank_model: str = "bge-reranker-v2-m3"
    pinecone_upsert_batch_size: int = 24
    pinecone_upsert_max_batch_tokens: int = 12000
    pinecone_upsert_tokens_per_minute: int = 180000
    pinecone_upsert_retry_attempts: int = 6
    pinecone_upsert_retry_base_delay_seconds: float = 5.0
    embedding_model: str = "embeddinggemma"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    answer_mode: str = "llm"
    enable_dense_retrieval: bool = True
    enable_sparse_retrieval: bool = True
    enable_reranker: bool = True
    temperature: float = 0.0

    chunk_size: int = 900
    chunk_overlap: int = 50
    dense_top_k: int = 4
    sparse_top_k: int = 4
    fused_top_k: int = 4
    final_context_k: int = 2
    extractive_context_k: int = 2
    extractive_max_chars: int = 320
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    raw_data_dir: Path = Field(default=DATA_DIR / "raw")
    processed_data_dir: Path = Field(default=DATA_DIR / "processed")
    eval_data_dir: Path = Field(default=DATA_DIR / "eval")
    faiss_dir: Path = Field(default=INDEX_DIR / "faiss")
    bm25_dir: Path = Field(default=INDEX_DIR / "bm25")
    metadata_db_path: Path = Field(default=PROJECT_ROOT / "data" / "metadata.db")

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.eval_data_dir,
            self.faiss_dir,
            self.bm25_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)

    def normalized_answer_mode(self) -> str:
        return self.answer_mode.strip().lower()

    def normalized_llm_provider(self) -> str:
        return self.llm_provider.strip().lower()

    def normalized_retrieval_provider(self) -> str:
        return self.retrieval_provider.strip().lower()

    def normalized_rerank_provider(self) -> str:
        return self.rerank_provider.strip().lower()

    def uses_dense_retrieval(self) -> bool:
        return self.enable_dense_retrieval and self.dense_top_k > 0

    def uses_sparse_retrieval(self) -> bool:
        return self.enable_sparse_retrieval and self.sparse_top_k > 0

    def uses_reranker(self) -> bool:
        return self.enable_reranker and self.final_context_k > 0

    def answer_uses_llm(self) -> bool:
        return self.normalized_answer_mode() == "llm"

    def answer_uses_groq(self) -> bool:
        return self.answer_uses_llm() and self.normalized_llm_provider() == "groq"

    def answer_uses_ollama(self) -> bool:
        return self.answer_uses_llm() and self.normalized_llm_provider() == "ollama"

    def uses_pinecone_retrieval(self) -> bool:
        return self.normalized_retrieval_provider() == "pinecone"

    def uses_local_retrieval(self) -> bool:
        return self.normalized_retrieval_provider() == "local"

    def uses_pinecone_reranker(self) -> bool:
        return self.uses_reranker() and self.normalized_rerank_provider() == "pinecone"

    def uses_local_reranker(self) -> bool:
        return self.uses_reranker() and self.normalized_rerank_provider() == "local"

    def required_ollama_models(self) -> List[str]:
        required = []
        if self.uses_local_retrieval() and self.uses_dense_retrieval():
            required.append(self.embedding_model)
        if self.answer_uses_ollama():
            required.append(self.chat_model)
        return required


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
