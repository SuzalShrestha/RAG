from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


@dataclass
class LoadedDocumentSet:
    doc_id: str
    filename: str
    file_path: str
    checksum: str
    file_type: str
    documents: List[Document]
    collection_name: str = "default"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    fused_score: float = 0.0
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None


@dataclass
class Citation:
    label: str
    chunk_id: str
    filename: str
    page_number: Optional[int]
    section_heading: Optional[str]
    excerpt: str


@dataclass
class AnswerResult:
    answer_markdown: str
    citations: List[Citation] = field(default_factory=list)
    used_chunks: List[RetrievedChunk] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    raw_response: Optional[Any] = None
    metrics: Optional["OperationMetrics"] = None


@dataclass
class IndexingSummary:
    files_indexed: int
    chunks_indexed: int
    total_documents: int
    total_chunks: int
    failed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    collection_name: str = "default"


@dataclass
class RetrievalFilters:
    filenames: List[str] = field(default_factory=list)
    collection_names: List[str] = field(default_factory=list)

    def normalized(self) -> "RetrievalFilters":
        filenames = sorted({name.strip() for name in self.filenames if name and name.strip()})
        collection_names = sorted({name.strip() for name in self.collection_names if name and name.strip()})
        return RetrievalFilters(filenames=filenames, collection_names=collection_names)

    def cache_key(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        normalized = self.normalized()
        return tuple(normalized.filenames), tuple(normalized.collection_names)

    def as_dict(self) -> Dict[str, List[str]]:
        normalized = self.normalized()
        return {
            "filenames": normalized.filenames,
            "collection_names": normalized.collection_names,
        }

    def is_empty(self) -> bool:
        normalized = self.normalized()
        return not normalized.filenames and not normalized.collection_names


@dataclass
class OperationMetrics:
    retrieval_provider: str = ""
    rerank_provider: str = ""
    llm_provider: str = ""
    retrieval_seconds: float = 0.0
    rerank_seconds: float = 0.0
    generation_seconds: float = 0.0
    total_seconds: float = 0.0
    retrieved_chunks: int = 0
    used_chunks: int = 0
    citation_count: int = 0
    prompt_chars: int = 0
    retrieval_cache_hit: bool = False
    answer_cache_hit: bool = False
    abstained: bool = False
    filters: Dict[str, List[str]] = field(default_factory=dict)
    used_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    file_path: str
    checksum: str
    file_type: str
    indexed_at: str
    collection_name: str = "default"


@dataclass
class IndexProgress:
    stage: str
    current: int
    total: int
    message: str


@dataclass
class IndexJobStatus:
    job_id: str
    status: str
    collection_name: str
    total_files: int
    processed_files: int = 0
    stage: str = "queued"
    message: str = ""
    started_at: str = ""
    finished_at: Optional[str] = None
    summary: Optional[IndexingSummary] = None
    error: Optional[str] = None
