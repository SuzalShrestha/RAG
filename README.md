# Local-First RAG Q&A

This repository contains a local-first document question-answering system built around hybrid retrieval:

- dense retrieval with FAISS
- sparse retrieval with BM25
- reciprocal rank fusion
- CrossEncoder reranking
- grounded answer generation with Ollama
- Streamlit UI for upload, indexing, and Q&A

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and adjust settings if needed:

```bash
cp .env.example .env
```

4. Pull the local models in Ollama:

```bash
ollama pull llama3.1:8b
ollama pull embeddinggemma
```

5. Start the app:

```bash
streamlit run app/ui/streamlit_app.py
```

## CLI

Build or refresh the index from the command line:

```bash
python3 scripts/build_index.py path/to/docs
```

Run a retrieval-focused evaluation against a JSONL file:

```bash
python3 scripts/run_eval.py --dataset data/eval/questions.jsonl
```

Each evaluation row should include:

```json
{"question":"What is the refund policy?","gold_chunk_ids":["chunk-123","chunk-456"]}
```

## Current State

The current implementation covers the first vertical slice:

- project scaffold and config
- document ingestion and chunk persistence
- dense + sparse retrieval with fusion
- reranking and grounded answer generation
- Streamlit interface

Generation-quality evaluation with `ragas` is the next layer to add once a corpus and gold dataset exist.

## Smoke Test

The repo includes a tiny sample corpus under `tests/fixtures/sample_docs/`.

Run retrieval-only smoke validation:

```bash
python3 scripts/smoke_test.py
```

Run full retrieval + generation smoke validation after Ollama is serving:

```bash
ollama serve
python3 scripts/smoke_test.py --with-generation
```

If you already have a different local chat model installed, override it at runtime. For example:

```bash
RAG_CHAT_MODEL=qwen3:8b python3 scripts/smoke_test.py --with-generation
```

If you want to test the UI with the same corpus, either upload those files in Streamlit or index them directly:

```bash
python3 scripts/build_index.py tests/fixtures/sample_docs
```
