from __future__ import annotations

import hashlib
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _approx_token_length(text: str) -> int:
    return len(text.split())


def _build_chunk_id(
    doc_id: str,
    chunk_index: int,
    page_number: object,
    content: str,
) -> str:
    page_value = page_number if page_number is not None else "na"
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return "chunk-{doc_id}-{page_value}-{chunk_index}-{digest}".format(
        doc_id=doc_id[:8],
        page_value=page_value,
        chunk_index=chunk_index,
        digest=digest,
    )


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_approx_token_length,
        add_start_index=True,
    )

    split_documents = splitter.split_documents(list(documents))
    chunked = []

    for index, document in enumerate(split_documents):
        metadata = dict(document.metadata)
        chunk_id = _build_chunk_id(
            doc_id=str(metadata.get("doc_id", "unknown")),
            chunk_index=index,
            page_number=metadata.get("page_number"),
            content=document.page_content,
        )
        metadata["chunk_index"] = index
        metadata["chunk_id"] = chunk_id
        chunked.append(Document(page_content=document.page_content, metadata=metadata))

    return chunked
