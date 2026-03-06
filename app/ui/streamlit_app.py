from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import RAGPipeline
from app.config import get_settings


st.set_page_config(page_title="Local RAG Q&A", layout="wide")


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = RAGPipeline()
    return st.session_state["pipeline"]


def save_uploads(uploaded_files) -> List[Path]:
    settings = get_settings()
    saved_paths = []

    for uploaded_file in uploaded_files:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded_file.name)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        target = settings.raw_data_dir / "{timestamp}_{name}".format(
            timestamp=timestamp,
            name=safe_name,
        )
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)

    return saved_paths


def render_sidebar(pipeline: RAGPipeline) -> None:
    st.sidebar.title("Corpus")
    st.sidebar.caption(
        "{docs} docs indexed, {chunks} chunks total".format(
            docs=pipeline.indexed_document_count(),
            chunks=pipeline.indexed_chunk_count(),
        )
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Index uploaded files", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one file first.")
            return

        saved_paths = save_uploads(uploaded_files)
        with st.spinner("Indexing documents..."):
            summary = pipeline.index_paths(saved_paths)

        if summary.files_indexed:
            st.sidebar.success(
                "Indexed {files} files and {chunks} chunks.".format(
                    files=summary.files_indexed,
                    chunks=summary.chunks_indexed,
                )
            )
        else:
            st.sidebar.error("No files were indexed.")

        if summary.failed_files:
            st.sidebar.warning(
                "Skipped {count} file(s) that could not be read.".format(
                    count=len(summary.failed_files),
                )
            )
            for failure in summary.failed_files:
                st.sidebar.caption(failure)

    settings = pipeline.settings
    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    st.sidebar.write("Chat model: `{model}`".format(model=settings.chat_model))
    st.sidebar.write("Embedding model: `{model}`".format(model=settings.embedding_model))
    st.sidebar.write("Reranker: `{model}`".format(model=settings.reranker_model))
    st.sidebar.write("Hybrid top-k: `{k}`".format(k=settings.fused_top_k))
    st.sidebar.write("Context chunks: `{k}`".format(k=settings.final_context_k))


def render_citations(citations) -> None:
    if not citations:
        return

    with st.expander("Sources", expanded=True):
        for citation in citations:
            location = "page {page}".format(page=citation.page_number) if citation.page_number else (
                citation.section_heading or "section n/a"
            )
            st.markdown(
                "**[{label}] {filename}**  \n{location}".format(
                    label=citation.label,
                    filename=citation.filename,
                    location=location,
                )
            )
            st.caption(citation.excerpt)


def main() -> None:
    pipeline = get_pipeline()
    render_sidebar(pipeline)

    st.title("Hybrid RAG Document Q&A")
    st.caption("Upload documents, build a local index, and ask grounded questions.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations", []))

    question = st.chat_input("Ask a question about the indexed documents")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if pipeline.indexed_chunk_count() == 0:
            answer = "No documents are indexed yet. Upload files and build the index first."
            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer, "citations": []})
            return

        with st.spinner("Retrieving evidence and generating answer..."):
            result = pipeline.answer(question)

        st.markdown(result.answer_markdown)
        render_citations(result.citations)
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result.answer_markdown,
                "citations": result.citations,
            }
        )


if __name__ == "__main__":
    main()
