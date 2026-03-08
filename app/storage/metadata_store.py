from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from app.utils.models import DocumentRecord, LoadedDocumentSet, RetrievalFilters


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
                    collection_name TEXT NOT NULL DEFAULT 'default',
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
            self._ensure_column(
                connection,
                "documents",
                "collection_name TEXT NOT NULL DEFAULT 'default'",
            )

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table_name: str, column_sql: str) -> None:
        try:
            connection.execute(
                "ALTER TABLE {table_name} ADD COLUMN {column_sql}".format(
                    table_name=table_name,
                    column_sql=column_sql,
                )
            )
        except sqlite3.OperationalError as error:
            if "duplicate column name" not in str(error).lower():
                raise

    def replace_document_chunks(
        self,
        loaded_set: LoadedDocumentSet,
        chunks: Iterable[Document],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (doc_id, filename, file_path, checksum, file_type, collection_name, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    file_path = excluded.file_path,
                    checksum = excluded.checksum,
                    file_type = excluded.file_type,
                    collection_name = excluded.collection_name,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    loaded_set.doc_id,
                    loaded_set.filename,
                    loaded_set.file_path,
                    loaded_set.checksum,
                    loaded_set.file_type,
                    loaded_set.collection_name,
                ),
            )
            connection.execute(
                "DELETE FROM chunks WHERE doc_id = ?",
                (loaded_set.doc_id,),
            )

            rows = []
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                metadata.setdefault("collection_name", loaded_set.collection_name)
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

    def has_checksum(self, checksum: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE checksum = ? LIMIT 1",
                (checksum,),
            ).fetchone()
        return row is not None

    def remove_duplicate_documents(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM documents
                    GROUP BY checksum
                )
                """
            ).fetchall()
            duplicate_ids = [str(row["doc_id"]) for row in rows]
            if not duplicate_ids:
                return 0

            placeholders = ", ".join("?" for _ in duplicate_ids)
            connection.execute(
                "DELETE FROM chunks WHERE doc_id IN ({placeholders})".format(
                    placeholders=placeholders
                ),
                duplicate_ids,
            )
            connection.execute(
                "DELETE FROM documents WHERE doc_id IN ({placeholders})".format(
                    placeholders=placeholders
                ),
                duplicate_ids,
            )

        return len(duplicate_ids)

    def load_all_chunk_documents(self, filters: Optional[RetrievalFilters] = None) -> List[Document]:
        normalized_filters = filters.normalized() if filters is not None else RetrievalFilters()
        clauses = []
        params: List[str] = []
        if normalized_filters.collection_names:
            placeholders = ", ".join("?" for _ in normalized_filters.collection_names)
            clauses.append("documents.collection_name IN ({placeholders})".format(placeholders=placeholders))
            params.extend(normalized_filters.collection_names)
        if normalized_filters.filenames:
            placeholders = ", ".join("?" for _ in normalized_filters.filenames)
            clauses.append("documents.filename IN ({placeholders})".format(placeholders=placeholders))
            params.extend(normalized_filters.filenames)

        where_clause = ""
        if clauses:
            where_clause = "WHERE {clauses}".format(clauses=" AND ".join(clauses))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunks.chunk_id, chunks.content, chunks.metadata_json, documents.checksum, documents.collection_name
                FROM chunks
                JOIN documents ON documents.doc_id = chunks.doc_id
                {where_clause}
                ORDER BY chunks.doc_id, chunks.chunk_index
                """.format(where_clause=where_clause),
                params,
            ).fetchall()

        documents = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            metadata["chunk_id"] = row["chunk_id"]
            if row["checksum"] and "checksum" not in metadata:
                metadata["checksum"] = row["checksum"]
            if row["collection_name"] and "collection_name" not in metadata:
                metadata["collection_name"] = row["collection_name"]
            documents.append(Document(page_content=row["content"], metadata=metadata))
        return documents

    def list_documents(self, filters: Optional[RetrievalFilters] = None) -> List[DocumentRecord]:
        normalized_filters = filters.normalized() if filters is not None else RetrievalFilters()
        clauses = []
        params: List[str] = []
        if normalized_filters.collection_names:
            placeholders = ", ".join("?" for _ in normalized_filters.collection_names)
            clauses.append("collection_name IN ({placeholders})".format(placeholders=placeholders))
            params.extend(normalized_filters.collection_names)
        if normalized_filters.filenames:
            placeholders = ", ".join("?" for _ in normalized_filters.filenames)
            clauses.append("filename IN ({placeholders})".format(placeholders=placeholders))
            params.extend(normalized_filters.filenames)

        where_clause = ""
        if clauses:
            where_clause = "WHERE {clauses}".format(clauses=" AND ".join(clauses))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT doc_id, filename, file_path, checksum, file_type, collection_name, indexed_at
                FROM documents
                {where_clause}
                ORDER BY collection_name, filename
                """
                .format(where_clause=where_clause),
                params,
            ).fetchall()

        return [
            DocumentRecord(
                doc_id=str(row["doc_id"]),
                filename=str(row["filename"]),
                file_path=str(row["file_path"]),
                checksum=str(row["checksum"]),
                file_type=str(row["file_type"]),
                collection_name=str(row["collection_name"] or "default"),
                indexed_at=str(row["indexed_at"]),
            )
            for row in rows
        ]

    def list_collections(self) -> List[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT collection_name
                FROM documents
                ORDER BY collection_name
                """
            ).fetchall()
        return [str(row["collection_name"] or "default") for row in rows]

    def document_count(self, filters: Optional[RetrievalFilters] = None) -> int:
        normalized_filters = filters.normalized() if filters is not None else RetrievalFilters()
        clauses = []
        params: List[str] = []
        if normalized_filters.collection_names:
            placeholders = ", ".join("?" for _ in normalized_filters.collection_names)
            clauses.append("collection_name IN ({placeholders})".format(placeholders=placeholders))
            params.extend(normalized_filters.collection_names)
        if normalized_filters.filenames:
            placeholders = ", ".join("?" for _ in normalized_filters.filenames)
            clauses.append("filename IN ({placeholders})".format(placeholders=placeholders))
            params.extend(normalized_filters.filenames)

        where_clause = ""
        if clauses:
            where_clause = "WHERE {clauses}".format(clauses=" AND ".join(clauses))

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM documents {where_clause}".format(where_clause=where_clause),
                params,
            ).fetchone()
        return int(row["count"])

    def chunk_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])
