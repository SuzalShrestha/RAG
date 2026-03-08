from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class LoadedDocumentSet:
    doc_id: str
    filename: str
    file_path: str
    checksum: str
    file_type: str
    documents: List[Document]


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


@dataclass
class IndexingSummary:
    files_indexed: int
    chunks_indexed: int
    total_documents: int
    total_chunks: int
    failed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
