from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import RAGPipeline
from app.config import Settings, get_settings
from app.utils.runtime_checks import groq_api_key_is_configured, ollama_is_running


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_fixture_paths() -> List[Path]:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "sample_docs"
    return sorted(path for path in fixture_dir.iterdir() if path.is_file())


def isolated_settings(base_settings: Settings, root: Path) -> Settings:
    payload = base_settings.model_dump()
    payload.update(
        retrieval_provider="local",
        rerank_provider="local",
        enable_dense_retrieval=False,
        enable_sparse_retrieval=True,
        enable_reranker=False,
        raw_data_dir=root / "raw",
        processed_data_dir=root / "processed",
        eval_data_dir=root / "eval",
        faiss_dir=root / "faiss",
        bm25_dir=root / "bm25",
        metadata_db_path=root / "data" / "metadata.db",
    )
    settings = Settings(_env_file=None, **payload)
    settings.ensure_directories()
    return settings


def verify_retrieval(pipeline: RAGPipeline, cases: Iterable[dict], top_k: int) -> List[str]:
    failures = []
    for case in cases:
        retrieved = pipeline.retrieve(case["question"])[:top_k]
        retrieved_filenames = [chunk.metadata.get("filename") for chunk in retrieved]
        expected_filenames = set(case.get("gold_filenames", []))
        if expected_filenames and not expected_filenames.intersection(retrieved_filenames):
            failures.append(
                "Retrieval miss for question '{question}'. Expected one of {expected}, got {actual}".format(
                    question=case["question"],
                    expected=sorted(expected_filenames),
                    actual=retrieved_filenames,
                )
            )
    return failures


def verify_generation(pipeline: RAGPipeline, cases: Iterable[dict]) -> List[str]:
    failures = []
    for case in cases:
        result = pipeline.answer(case["question"])
        answer_text = result.answer_markdown.lower()
        missing_phrases = [
            phrase for phrase in case.get("answer_contains", [])
            if phrase.lower() not in answer_text
        ]
        if missing_phrases:
            failures.append(
                "Generation miss for question '{question}'. Missing phrases: {phrases}. Answer: {answer}".format(
                    question=case["question"],
                    phrases=missing_phrases,
                    answer=result.answer_markdown,
                )
            )
        if not result.abstained and "[C" not in result.answer_markdown:
            failures.append(
                "Answer for question '{question}' did not include citations.".format(
                    question=case["question"],
                )
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bundled smoke test against the sample corpus.")
    parser.add_argument(
        "--dataset",
        default="data/eval/sample_questions.jsonl",
        help="JSONL file describing smoke-test questions",
    )
    parser.add_argument(
        "--docs",
        nargs="*",
        help="Optional explicit document paths. Defaults to bundled sample docs.",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also run answer generation checks if the configured LLM provider is available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieved chunks to inspect for retrieval validation.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    cases = load_jsonl(dataset_path)
    doc_paths = [Path(path).expanduser().resolve() for path in args.docs] if args.docs else default_fixture_paths()

    if not doc_paths:
        raise ValueError("No smoke-test documents were found.")

    settings = get_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = RAGPipeline(settings=isolated_settings(settings, Path(temp_dir)))
        summary = pipeline.index_paths(doc_paths)

        print("Indexed files: {count}".format(count=summary.files_indexed))
        print("Indexed chunks: {count}".format(count=summary.chunks_indexed))

        retrieval_failures = verify_retrieval(pipeline, cases, args.top_k)
        for failure in retrieval_failures:
            print("FAIL:", failure)

        generation_requested = args.with_generation
        generation_failures = []

        if generation_requested:
            if settings.answer_uses_groq() and not groq_api_key_is_configured(settings.groq_api_key):
                raise RuntimeError("Groq API key is missing. Set `RAG_GROQ_API_KEY` in `.env` and try again.")
            if settings.answer_uses_ollama() and not ollama_is_running(settings.ollama_base_url):
                raise RuntimeError(
                    "Ollama is not reachable at {url}. Start `ollama serve` and try again.".format(
                        url=settings.ollama_base_url,
                    )
                )
            generation_failures = verify_generation(pipeline, cases)
            for failure in generation_failures:
                print("FAIL:", failure)

        if retrieval_failures or generation_failures:
            raise SystemExit(1)

        print("PASS: retrieval smoke test")
        if generation_requested:
            print("PASS: answer smoke test")
        else:
            print("Answer checks were skipped. Re-run with --with-generation to validate answer formatting.")


if __name__ == "__main__":
    main()
