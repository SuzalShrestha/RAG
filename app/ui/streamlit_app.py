from __future__ import annotations

import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import RAGPipeline
from app.config import Settings, get_settings
from app.utils.background_jobs import BackgroundIndexManager
from app.utils.models import OperationMetrics, RetrievalFilters, RetrievedChunk
from app.utils.runtime_checks import (
    groq_api_key_is_configured,
    list_ollama_models,
    missing_ollama_models,
    ollama_is_running,
    pinecone_api_key_is_configured,
)


st.set_page_config(page_title="Cloud RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline(settings_json: str) -> RAGPipeline:
    settings = Settings.model_validate_json(settings_json)
    settings.ensure_directories()
    return RAGPipeline(settings=settings)


@st.cache_resource(show_spinner=False)
def get_index_manager() -> BackgroundIndexManager:
    return BackgroundIndexManager()


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
    missing_models = missing_ollama_models(base_url, required_models) if running else list(required_models)
    return {
        "running": running,
        "installed_models": installed_models,
        "missing_models": missing_models,
        "required_models": list(required_models),
    }


def save_uploads(uploaded_files, settings: Settings) -> List[Path]:
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


def build_active_settings(base_settings: Settings) -> Settings:
    st.sidebar.subheader("Runtime Settings")
    answer_mode = st.sidebar.selectbox(
        "Answer mode",
        options=["llm", "extractive"],
        index=0 if base_settings.normalized_answer_mode() == "llm" else 1,
    )
    llm_provider = st.sidebar.selectbox(
        "LLM provider",
        options=["groq", "ollama"],
        index=0 if base_settings.normalized_llm_provider() == "groq" else 1,
    )
    retrieval_provider = st.sidebar.selectbox(
        "Retrieval provider",
        options=["pinecone", "local"],
        index=0 if base_settings.normalized_retrieval_provider() == "pinecone" else 1,
    )
    rerank_provider = st.sidebar.selectbox(
        "Rerank provider",
        options=["pinecone", "local"],
        index=0 if base_settings.normalized_rerank_provider() == "pinecone" else 1,
    )
    enable_dense = st.sidebar.checkbox("Enable dense retrieval", value=base_settings.enable_dense_retrieval)
    enable_sparse = st.sidebar.checkbox("Enable sparse retrieval", value=base_settings.enable_sparse_retrieval)
    enable_reranker = st.sidebar.checkbox("Enable reranker", value=base_settings.enable_reranker)
    enable_ocr = st.sidebar.checkbox("Enable OCR fallback for PDFs", value=base_settings.enable_ocr)
    dense_top_k = st.sidebar.slider("Dense top-k", min_value=1, max_value=20, value=base_settings.dense_top_k)
    sparse_top_k = st.sidebar.slider("Sparse top-k", min_value=1, max_value=20, value=base_settings.sparse_top_k)
    fused_top_k = st.sidebar.slider("Fused top-k", min_value=1, max_value=20, value=base_settings.fused_top_k)
    final_context_k = st.sidebar.slider(
        "LLM context chunks",
        min_value=1,
        max_value=12,
        value=base_settings.final_context_k,
    )
    extractive_context_k = st.sidebar.slider(
        "Extractive context chunks",
        min_value=1,
        max_value=12,
        value=base_settings.extractive_context_k,
    )
    overrides = {
        "answer_mode": answer_mode,
        "llm_provider": llm_provider,
        "retrieval_provider": retrieval_provider,
        "rerank_provider": rerank_provider,
        "enable_dense_retrieval": enable_dense,
        "enable_sparse_retrieval": enable_sparse,
        "enable_reranker": enable_reranker,
        "enable_ocr": enable_ocr,
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "fused_top_k": fused_top_k,
        "final_context_k": final_context_k,
        "extractive_context_k": extractive_context_k,
    }
    active_settings = base_settings.model_copy(update=overrides, deep=True)
    active_settings.ensure_directories()
    return active_settings


def render_runtime_status(settings: Settings) -> None:
    runtime_status = get_runtime_status(
        settings.ollama_base_url,
        tuple(settings.required_ollama_models()),
    )
    st.sidebar.divider()
    st.sidebar.subheader("Backend Status")
    if settings.uses_pinecone_retrieval() or settings.uses_pinecone_reranker():
        if pinecone_api_key_is_configured(settings.pinecone_api_key):
            st.sidebar.success("Pinecone API key configured.")
        else:
            st.sidebar.error("Pinecone API key is missing. Set `RAG_PINECONE_API_KEY`.")

    if settings.answer_uses_groq():
        if groq_api_key_is_configured(settings.groq_api_key):
            st.sidebar.success("Groq API key configured.")
        else:
            st.sidebar.error("Groq API key is missing. Set `RAG_GROQ_API_KEY`.")
    elif not runtime_status["required_models"]:
        st.sidebar.success("Current settings do not require Ollama.")
    elif not runtime_status["running"]:
        st.sidebar.error(
            "Ollama is not reachable at `{url}`.".format(
                url=settings.ollama_base_url,
            )
        )
    elif runtime_status["missing_models"]:
        installed = ", ".join("`{name}`".format(name=name) for name in runtime_status["installed_models"]) or "none"
        missing = ", ".join("`{name}`".format(name=name) for name in runtime_status["missing_models"])
        st.sidebar.warning(
            "Missing Ollama model(s): {missing}. Installed: {installed}.".format(
                missing=missing,
                installed=installed,
            )
        )
    else:
        st.sidebar.success("Ollama runtime looks ready.")


def render_index_job(index_manager: BackgroundIndexManager) -> None:
    job_id = st.session_state.get("index_job_id")
    if not job_id:
        return

    job = index_manager.get_job(job_id)
    if job is None:
        return

    st.sidebar.divider()
    st.sidebar.subheader("Index Job")
    progress_total = max(1, job.total_files)
    progress_ratio = min(1.0, job.processed_files / progress_total)
    st.sidebar.progress(progress_ratio, text=job.message or job.stage.title())
    st.sidebar.caption(
        "{status} · {processed}/{total} files".format(
            status=job.status,
            processed=job.processed_files,
            total=job.total_files,
        )
    )

    if job.status == "completed":
        summary = job.summary
        if st.session_state.get("index_job_cache_refreshed") != job.job_id:
            get_pipeline.clear()
            st.session_state["index_job_cache_refreshed"] = job.job_id
            st.rerun()
        if summary is not None:
            st.sidebar.success(
                "Indexed {files} files into collection `{collection}`.".format(
                    files=summary.files_indexed,
                    collection=summary.collection_name,
                )
            )
    elif job.status == "failed":
        st.sidebar.error(job.error or "Indexing job failed.")
    else:
        st.sidebar.button("Refresh indexing status", use_container_width=True)


def render_retrieval_debug(chunks: List[RetrievedChunk], title: str) -> None:
    st.markdown("**{title}**".format(title=title))
    if not chunks:
        st.caption("No chunks to show.")
        return

    for chunk in chunks:
        location = "page {page}".format(page=chunk.metadata.get("page_number")) if chunk.metadata.get("page_number") else (
            chunk.metadata.get("section_heading") or "section n/a"
        )
        scores = []
        if chunk.dense_score is not None:
            scores.append("dense={score:.3f}".format(score=chunk.dense_score))
        if chunk.fused_score:
            scores.append("fused={score:.3f}".format(score=chunk.fused_score))
        if chunk.rerank_score is not None:
            scores.append("rerank={score:.3f}".format(score=chunk.rerank_score))
        if chunk.sparse_rank is not None:
            scores.append("sparse_rank={rank}".format(rank=chunk.sparse_rank))
        st.markdown(
            "`{chunk_id}` · `{collection}` · **{filename}** · {location}".format(
                chunk_id=chunk.chunk_id,
                collection=chunk.metadata.get("collection_name", "default"),
                filename=chunk.metadata.get("filename", "unknown"),
                location=location,
            )
        )
        if scores:
            st.caption(" | ".join(scores))
        st.code(chunk.text[:700], language="text")


def render_metrics(metrics: OperationMetrics | None) -> None:
    if metrics is None:
        return
    metrics_payload = asdict(metrics)
    st.caption(
        "retrieval {retrieval:.2f}s · rerank {rerank:.2f}s · generation {generation:.2f}s · total {total:.2f}s".format(
            retrieval=metrics_payload["retrieval_seconds"],
            rerank=metrics_payload["rerank_seconds"],
            generation=metrics_payload["generation_seconds"],
            total=metrics_payload["total_seconds"],
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


def render_sidebar(
    settings: Settings,
    pipeline: RAGPipeline,
    index_manager: BackgroundIndexManager,
) -> dict:
    st.sidebar.title("Corpus")
    st.sidebar.caption(
        "{docs} docs indexed, {chunks} chunks total".format(
            docs=pipeline.indexed_document_count(),
            chunks=pipeline.indexed_chunk_count(),
        )
    )

    upload_collection = st.sidebar.text_input(
        "Collection for new uploads",
        value=settings.default_collection_name,
    ).strip() or settings.default_collection_name
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    run_in_background = st.sidebar.checkbox("Index in background", value=True)

    if st.sidebar.button("Index uploaded files", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one file first.")
        else:
            saved_paths = save_uploads(uploaded_files, settings)
            if run_in_background:
                job_id = index_manager.start_index_job(settings, saved_paths, upload_collection)
                st.session_state["index_job_id"] = job_id
                st.session_state["index_job_cache_refreshed"] = None
                st.sidebar.success("Background indexing started.")
                st.rerun()
            else:
                with st.spinner("Indexing documents..."):
                    summary = pipeline.index_paths(saved_paths, collection_name=upload_collection)
                st.sidebar.success(
                    "Indexed {files} files and {chunks} chunks.".format(
                        files=summary.files_indexed,
                        chunks=summary.chunks_indexed,
                    )
                )
                get_pipeline.clear()
                st.rerun()

    render_index_job(index_manager)

    st.sidebar.divider()
    st.sidebar.subheader("Query Filters")
    documents = pipeline.list_documents()
    collection_options = sorted({document.collection_name for document in documents})
    selected_collections = st.sidebar.multiselect("Collections", options=collection_options)
    document_options = [
        document.filename
        for document in documents
        if not selected_collections or document.collection_name in selected_collections
    ]
    selected_filenames = st.sidebar.multiselect("Documents", options=document_options)
    show_debug = st.sidebar.checkbox("Show retrieval debug", value=False)
    show_logs = st.sidebar.checkbox("Show recent telemetry", value=False)

    render_runtime_status(settings)

    return {
        "filters": RetrievalFilters(
            filenames=selected_filenames,
            collection_names=selected_collections,
        ).normalized(),
        "show_debug": show_debug,
        "show_logs": show_logs,
        "upload_collection": upload_collection,
    }


def main() -> None:
    base_settings = get_settings()
    settings = build_active_settings(base_settings)
    pipeline = get_pipeline(settings.model_dump_json())
    index_manager = get_index_manager()
    sidebar_state = render_sidebar(settings, pipeline, index_manager)

    st.title("Cloud RAG")
    st.caption("Upload docs, scope them into collections, and inspect grounded answers with retrieval telemetry.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_metrics(message.get("metrics"))
                render_citations(message.get("citations", []))
                if sidebar_state["show_debug"]:
                    with st.expander("Retrieval Debug", expanded=False):
                        render_retrieval_debug(message.get("used_chunks", []), "Reranked context")
                        render_retrieval_debug(message.get("retrieved_chunks", []), "Retrieved candidates")

    if sidebar_state["show_logs"]:
        with st.expander("Recent Telemetry", expanded=False):
            st.json(pipeline.recent_events(limit=10))

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
                result = pipeline.answer(question, filters=sidebar_state["filters"])
        except Exception as error:
            answer = "Unable to answer right now: {error}".format(error=error)
            st.error(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer, "citations": []})
            return

        st.markdown(result.answer_markdown)
        render_metrics(result.metrics)
        render_citations(result.citations)
        if sidebar_state["show_debug"]:
            with st.expander("Retrieval Debug", expanded=False):
                render_retrieval_debug(result.used_chunks, "Reranked context")
                render_retrieval_debug(result.retrieved_chunks, "Retrieved candidates")
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result.answer_markdown,
                "citations": result.citations,
                "used_chunks": result.used_chunks,
                "retrieved_chunks": result.retrieved_chunks,
                "metrics": result.metrics,
            }
        )


if __name__ == "__main__":
    main()
