from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "embeddinggemma"
    reranker_model: str = "BAAI/bge-reranker-base"
    temperature: float = 0.0

    chunk_size: int = 750
    chunk_overlap: int = 100
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fused_top_k: int = 20
    final_context_k: int = 6
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


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
