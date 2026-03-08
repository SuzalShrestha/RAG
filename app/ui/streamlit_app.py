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
from app.utils.runtime_checks import (
    groq_api_key_is_configured,
    list_ollama_models,
    missing_ollama_models,
    ollama_is_running,
    pinecone_api_key_is_configured,
)


st.set_page_config(page_title="Fast RAG MVP", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_data(ttl=5, show_spinner=False)
def get_runtime_status(base_url: str, required_models: tuple[str, ...]) -> dict:
    if not required_models:
        return {
            "running": False,
            "installed_models": [],
            "missing_models": [],
            "required_models": [],
        }

    running = ollama_is_running(base_url)
    installed_models = list_ollama_models(base_url) if running else []
    missing_models = (
        missing_ollama_models(base_url, required_models)
        if running else list(required_models)
    )
    return {
        "running": running,
        "installed_models": installed_models,
        "missing_models": missing_models,
        "required_models": list(required_models),
    }


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
        elif summary.skipped_files:
            st.sidebar.info("No new files were indexed.")
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

        if summary.skipped_files:
            st.sidebar.info(
                "Skipped {count} file(s) that were already indexed.".format(
                    count=len(summary.skipped_files),
                )
            )
            for skipped in summary.skipped_files:
                st.sidebar.caption(skipped)

    settings = pipeline.settings
    runtime_status = get_runtime_status(
        settings.ollama_base_url,
        tuple(settings.required_ollama_models()),
    )
    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    st.sidebar.write("Answer mode: `{mode}`".format(mode=settings.normalized_answer_mode()))
    st.sidebar.write("LLM provider: `{provider}`".format(provider=settings.normalized_llm_provider()))
    st.sidebar.write("Retrieval provider: `{provider}`".format(provider=settings.normalized_retrieval_provider()))
    st.sidebar.write("Rerank provider: `{provider}`".format(provider=settings.normalized_rerank_provider()))
    st.sidebar.write("Dense retrieval: `{enabled}`".format(enabled=settings.uses_dense_retrieval()))
    st.sidebar.write("Sparse retrieval: `{enabled}`".format(enabled=settings.uses_sparse_retrieval()))
    st.sidebar.write("Reranker: `{enabled}`".format(enabled=settings.uses_reranker()))
    context_k = settings.final_context_k if settings.answer_uses_llm() else settings.extractive_context_k
    st.sidebar.write("Context chunks: `{k}`".format(k=context_k))
    if settings.uses_dense_retrieval() and settings.uses_pinecone_retrieval():
        st.sidebar.write("Dense index: `{name}`".format(name=settings.pinecone_dense_index))
        st.sidebar.write("Dense model: `{model}`".format(model=settings.pinecone_dense_model))
    elif settings.uses_dense_retrieval():
        st.sidebar.write("Embedding model: `{model}`".format(model=settings.embedding_model))
    if settings.uses_sparse_retrieval() and settings.uses_pinecone_retrieval():
        st.sidebar.write("Sparse index: `{name}`".format(name=settings.pinecone_sparse_index))
        st.sidebar.write("Sparse model: `{model}`".format(model=settings.pinecone_sparse_model))
    if settings.answer_uses_llm():
        st.sidebar.write("Chat model: `{model}`".format(model=settings.chat_model))
    if settings.uses_pinecone_reranker():
        st.sidebar.write("Rerank model: `{model}`".format(model=settings.pinecone_rerank_model))
    elif settings.uses_reranker():
        st.sidebar.write("Reranker model: `{model}`".format(model=settings.reranker_model))
        st.sidebar.write("Reranker device: `{device}`".format(device=settings.reranker_device))

    if settings.uses_pinecone_retrieval() or settings.uses_pinecone_reranker():
        if pinecone_api_key_is_configured(settings.pinecone_api_key):
            st.sidebar.success("Pinecone API key configured for retrieval.")
        else:
            st.sidebar.error("Pinecone API key is missing. Set `RAG_PINECONE_API_KEY` in `.env`.")

    if settings.answer_uses_groq():
        if groq_api_key_is_configured(settings.groq_api_key):
            st.sidebar.success("Groq API key configured for answer generation.")
        else:
            st.sidebar.error("Groq API key is missing. Set `RAG_GROQ_API_KEY` in `.env`.")
    elif not runtime_status["required_models"]:
        st.sidebar.success("Current MVP mode does not require Ollama for questions.")
    elif runtime_status["required_models"] and not runtime_status["running"]:
        st.sidebar.error(
            "Ollama is not reachable at `{url}`. Start `ollama serve` before asking questions.".format(
                url=settings.ollama_base_url,
            )
        )
    elif runtime_status["missing_models"]:
        installed = ", ".join("`{name}`".format(name=name) for name in runtime_status["installed_models"]) or "none"
        missing = ", ".join("`{name}`".format(name=name) for name in runtime_status["missing_models"])
        st.sidebar.warning(
            "Missing Ollama model(s): {missing}. Installed models: {installed}.".format(
                missing=missing,
                installed=installed,
            )
        )


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

    st.title("Fast RAG MVP")
    st.caption("Upload documents, keep indexing lightweight, and get cited answers quickly.")

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

        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                result = pipeline.answer(question)
        except Exception as error:
            answer = "Unable to answer right now: {error}".format(error=error)
            st.error(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer, "citations": []})
            return

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
