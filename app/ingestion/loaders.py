from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

from docx import Document as WordDocument
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

from app.utils.models import LoadedDocumentSet


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_doc_id(path: Path, checksum: str) -> str:
    raw = "{name}:{checksum}".format(name=path.name, checksum=checksum)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    filtered = [line for line in lines if line]
    normalized = "\n".join(filtered)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def load_file(path: Path) -> LoadedDocumentSet:
    if not path.exists():
        raise FileNotFoundError("File not found: {path}".format(path=path))

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type: {suffix}. Supported: {supported}".format(
                suffix=suffix,
                supported=", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
        )

    checksum = compute_checksum(path)
    doc_id = build_doc_id(path, checksum)

    if suffix == ".pdf":
        documents = _load_pdf(path, doc_id)
    elif suffix == ".docx":
        documents = _load_docx(path, doc_id)
    else:
        documents = _load_text(path, doc_id)

    if not documents:
        raise ValueError(
            "No text content extracted from {path}. The PDF may be scanned/image-only or use an unsupported encoding.".format(
                path=path
            )
        )

    return LoadedDocumentSet(
        doc_id=doc_id,
        filename=path.name,
        file_path=str(path.resolve()),
        checksum=checksum,
        file_type=suffix.lstrip("."),
        documents=documents,
    )


def _base_metadata(path: Path, doc_id: str, file_type: str) -> dict:
    return {
        "doc_id": doc_id,
        "filename": path.name,
        "source_path": str(path.resolve()),
        "file_type": file_type,
    }


def _load_pdf(path: Path, doc_id: str) -> List[Document]:
    documents = _load_pdf_with_pypdf(path, doc_id)
    if documents:
        return documents

    documents = _load_pdf_with_pymupdf(path, doc_id)
    if documents:
        return documents

    return []


def _load_pdf_with_pypdf(path: Path, doc_id: str) -> List[Document]:
    base_metadata = _base_metadata(path, doc_id, "pdf")
    documents = []

    try:
        reader = PdfReader(str(path))
    except PdfReadError as error:
        raise ValueError("Could not read PDF {path}: {error}".format(path=path, error=error)) from error

    for index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        metadata = dict(base_metadata)
        metadata["page_number"] = index
        metadata["extractor"] = "pypdf"
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _load_pdf_with_pymupdf(path: Path, doc_id: str) -> List[Document]:
    if fitz is None:
        return []

    base_metadata = _base_metadata(path, doc_id, "pdf")
    documents = []
    pdf = fitz.open(str(path))

    try:
        for index, page in enumerate(pdf, start=1):
            text = normalize_text(page.get_text("text") or "")
            if not text:
                continue
            metadata = dict(base_metadata)
            metadata["page_number"] = index
            metadata["extractor"] = "pymupdf"
            documents.append(Document(page_content=text, metadata=metadata))
    finally:
        pdf.close()

    return documents


def _load_docx(path: Path, doc_id: str) -> List[Document]:
    document = WordDocument(str(path))
    base_metadata = _base_metadata(path, doc_id, "docx")
    parts = []
    current_heading = None

    for paragraph in document.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue

        style_name = getattr(paragraph.style, "name", "") or ""
        if style_name.lower().startswith("heading"):
            current_heading = text
            parts.append("\n{heading}\n".format(heading=text))
            continue

        parts.append(text)

    content = normalize_text("\n".join(parts))
    if not content:
        return []

    metadata = dict(base_metadata)
    metadata["section_heading"] = current_heading
    return [Document(page_content=content, metadata=metadata)]


def _load_text(path: Path, doc_id: str) -> List[Document]:
    base_metadata = _base_metadata(path, doc_id, path.suffix.lower().lstrip("."))
    content = normalize_text(path.read_text(encoding="utf-8"))
    if not content:
        return []
    return [Document(page_content=content, metadata=base_metadata)]
