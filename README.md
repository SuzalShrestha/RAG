# Cloud RAG

This repository now defaults to a cloud-first MVP configuration intended to keep laptops cool:

- local PDF parsing and chunking only
- hosted dense retrieval with Pinecone integrated embeddings
- hosted sparse retrieval with Pinecone sparse indexes
- hosted reranking with Pinecone
- Groq answer generation
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

4. Start the app:

```bash
streamlit run app/ui/streamlit_app.py
```

The default path requires `RAG_PINECONE_API_KEY` and `RAG_GROQ_API_KEY` in `.env`. Ollama is only needed if you manually switch retrieval back to `local`.

## CLI

Build or refresh the index from the command line:

```bash
python3 scripts/build_index.py path/to/docs
```

Rebuild indexes from the stored chunk corpus without rereading raw files:

```bash
python3 scripts/build_index.py --from-db
```

Run retrieval baselines against a JSONL file:

```bash
python3 scripts/run_eval.py --dataset data/eval/sample_questions.jsonl --mode all
```

Also run answer-quality checks for rows that include `answer_contains` or `expected_refusal`:

```bash
python3 scripts/run_eval.py --dataset data/eval/sample_questions.jsonl --mode hybrid_rerank --with-generation
```

Each evaluation row should include:

```json
{"question":"What is the refund policy?","gold_chunk_ids":["chunk-123","chunk-456"]}
```

or:

```json
{"question":"What is the refund policy?","gold_filenames":["policy.md"],"answer_contains":["30 calendar days"]}
```

## Cloud Defaults

- `RAG_RETRIEVAL_PROVIDER=pinecone`
- `RAG_RERANK_PROVIDER=pinecone`
- `RAG_LLM_PROVIDER=groq`
- `RAG_ANSWER_MODE=llm`
- `RAG_ENABLE_DENSE_RETRIEVAL=true`
- `RAG_ENABLE_SPARSE_RETRIEVAL=true`
- `RAG_ENABLE_RERANKER=true`

The default Pinecone setup uses:

- dense index `rag-dense` with `multilingual-e5-large`
- sparse index `rag-sparse` with `pinecone-sparse-english-v0`
- rerank model `bge-reranker-v2-m3`

If you want to fall back to local retrieval later, switch:

```bash
RAG_RETRIEVAL_PROVIDER=local
RAG_RERANK_PROVIDER=local
RAG_ENABLE_DENSE_RETRIEVAL=false
RAG_ENABLE_RERANKER=false
RAG_ANSWER_MODE=extractive
```

Local dense retrieval still uses Ollama if you re-enable it.

## Smoke Test

The repo includes a tiny sample corpus under `tests/fixtures/sample_docs/`.

Run retrieval-only smoke validation:

```bash
python3 scripts/smoke_test.py
```

Run full answer-format smoke validation:

```bash
python3 scripts/smoke_test.py --with-generation
```

The smoke test intentionally forces local sparse retrieval so it can run offline. If you want answer validation with Groq, set the API key and optionally override the model at runtime. For example:

```bash
RAG_LLM_PROVIDER=groq RAG_CHAT_MODEL=llama-3.1-8b-instant python3 scripts/smoke_test.py --with-generation
```

If you want to test the UI with the same corpus, either upload those files in Streamlit or index them directly:

```bash
python3 scripts/build_index.py tests/fixtures/sample_docs
```
