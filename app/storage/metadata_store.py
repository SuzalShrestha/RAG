from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from app.utils.models import LoadedDocumentSet


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    section_heading TEXT,
                    content TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
                """
            )

    def replace_document_chunks(
        self,
        loaded_set: LoadedDocumentSet,
        chunks: Iterable[Document],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (doc_id, filename, file_path, checksum, file_type, indexed_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    file_path = excluded.file_path,
                    checksum = excluded.checksum,
                    file_type = excluded.file_type,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    loaded_set.doc_id,
                    loaded_set.filename,
                    loaded_set.file_path,
                    loaded_set.checksum,
                    loaded_set.file_type,
                ),
            )
            connection.execute(
                "DELETE FROM chunks WHERE doc_id = ?",
                (loaded_set.doc_id,),
            )

            rows = []
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                rows.append(
                    (
                        metadata["chunk_id"],
                        loaded_set.doc_id,
                        int(metadata.get("chunk_index", 0)),
                        metadata.get("page_number"),
                        metadata.get("section_heading"),
                        chunk.page_content,
                        metadata.get("source_path", loaded_set.file_path),
                        json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                    )
                )

            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    doc_id,
                    chunk_index,
                    page_number,
                    section_heading,
                    content,
                    source_path,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def load_all_chunk_documents(self) -> List[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, content, metadata_json
                FROM chunks
                ORDER BY doc_id, chunk_index
                """
            ).fetchall()

        documents = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            metadata["chunk_id"] = row["chunk_id"]
            documents.append(Document(page_content=row["content"], metadata=metadata))
        return documents

    def document_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])

    def chunk_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])
