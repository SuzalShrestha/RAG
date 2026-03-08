from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import RAGPipeline
from app.utils.models import AnswerResult, RetrievedChunk


RETRIEVAL_MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unique_preserving_order(values: Sequence[str]) -> List[str]:
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def gold_targets(example: dict) -> Tuple[Optional[str], List[str]]:
    chunk_ids = [str(value) for value in example.get("gold_chunk_ids", []) if value]
    if chunk_ids:
        return "chunk_id", unique_preserving_order(chunk_ids)

    filenames = [str(value) for value in example.get("gold_filenames", []) if value]
    if filenames:
        return "filename", unique_preserving_order(filenames)

    return None, []


def chunk_target(chunk: RetrievedChunk, target_type: str) -> Optional[str]:
    if target_type == "chunk_id":
        return chunk.chunk_id
    if target_type == "filename":
        filename = chunk.metadata.get("filename")
        return str(filename) if filename else None
    raise ValueError("Unsupported target type: {target_type}".format(target_type=target_type))


def predicted_targets(
    chunks: Sequence[RetrievedChunk],
    target_type: str,
    top_k: int,
) -> List[str]:
    values = []
    for chunk in chunks[:top_k]:
        value = chunk_target(chunk, target_type)
        if value:
            values.append(value)
    return unique_preserving_order(values)


def reciprocal_rank(gold_ids: Sequence[str], predicted_ids: Sequence[str]) -> float:
    for index, predicted_id in enumerate(predicted_ids, start=1):
        if predicted_id in gold_ids:
            return 1.0 / float(index)
    return 0.0


def ndcg_at_k(gold_ids: Sequence[str], predicted_ids: Sequence[str], top_k: int) -> float:
    if not gold_ids:
        return 0.0

    dcg = 0.0
    for index, predicted_id in enumerate(predicted_ids[:top_k], start=1):
        if predicted_id in gold_ids:
            dcg += 1.0 / math.log2(index + 1.0)

    ideal_hits = min(len(gold_ids), top_k)
    ideal_dcg = sum(1.0 / math.log2(index + 1.0) for index in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_retrieval(
    examples: Sequence[dict],
    retrieve_fn: Callable[[str], List[RetrievedChunk]],
    top_k: int,
) -> Tuple[dict, List[dict]]:
    evaluated_examples = 0
    hit_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    failures = []

    for example in examples:
        target_type, gold_ids = gold_targets(example)
        if not gold_ids:
            continue

        evaluated_examples += 1
        retrieved = retrieve_fn(example["question"])
        predicted_ids = predicted_targets(retrieved, target_type, top_k)
        gold_set = set(gold_ids)
        intersection = gold_set.intersection(predicted_ids)

        hit = 1.0 if intersection else 0.0
        recall = len(intersection) / float(len(gold_set))
        mrr = reciprocal_rank(gold_ids, predicted_ids)
        ndcg = ndcg_at_k(gold_ids, predicted_ids, top_k)

        hit_sum += hit
        recall_sum += recall
        mrr_sum += mrr
        ndcg_sum += ndcg

        if not intersection:
            failures.append(
                {
                    "question": example["question"],
                    "gold_targets": gold_ids,
                    "predicted_targets": predicted_ids,
                    "target_type": target_type,
                }
            )

    count = float(evaluated_examples or 1)
    metrics = {
        "evaluated_examples": evaluated_examples,
        "HitRate@{k}".format(k=top_k): hit_sum / count if evaluated_examples else 0.0,
        "Recall@{k}".format(k=top_k): recall_sum / count if evaluated_examples else 0.0,
        "MRR@{k}".format(k=top_k): mrr_sum / count if evaluated_examples else 0.0,
        "nDCG@{k}".format(k=top_k): ndcg_sum / count if evaluated_examples else 0.0,
    }
    return metrics, failures


def evaluate_generation(
    examples: Sequence[dict],
    answer_fn: Callable[[str], AnswerResult],
) -> Tuple[dict, List[dict]]:
    checked_examples = 0
    answer_contains_cases = 0
    answer_contains_passes = 0
    refusal_cases = 0
    refusal_passes = 0
    citation_cases = 0
    citation_passes = 0
    failures = []

    for example in examples:
        expected_phrases = [str(value) for value in example.get("answer_contains", []) if value]
        expected_refusal = example.get("expected_refusal")
        should_check = bool(expected_phrases) or expected_refusal is not None
        if not should_check:
            continue

        checked_examples += 1
        result = answer_fn(example["question"])
        answer_text = result.answer_markdown.lower()
        citations_present = "[c" in answer_text

        failed_checks = []
        if expected_phrases:
            answer_contains_cases += 1
            missing_phrases = [
                phrase for phrase in expected_phrases if phrase.lower() not in answer_text
            ]
            if not missing_phrases:
                answer_contains_passes += 1
            else:
                failed_checks.append({"missing_phrases": missing_phrases})

        if expected_refusal is not None:
            refusal_cases += 1
            refusal_pass = bool(result.abstained) is bool(expected_refusal)
            if refusal_pass:
                refusal_passes += 1
            else:
                failed_checks.append(
                    {
                        "expected_refusal": bool(expected_refusal),
                        "actual_refusal": bool(result.abstained),
                    }
                )

        if not result.abstained:
            citation_cases += 1
            if citations_present:
                citation_passes += 1
            else:
                failed_checks.append({"missing_citations": True})

        if failed_checks:
            failures.append(
                {
                    "question": example["question"],
                    "answer_markdown": result.answer_markdown,
                    "checks": failed_checks,
                }
            )

    def rate(passes: int, cases: int) -> float:
        return passes / float(cases) if cases else 0.0

    metrics = {
        "checked_examples": checked_examples,
        "answer_contains_cases": answer_contains_cases,
        "answer_contains_pass_rate": rate(answer_contains_passes, answer_contains_cases),
        "refusal_cases": refusal_cases,
        "refusal_accuracy": rate(refusal_passes, refusal_cases),
        "citation_cases": citation_cases,
        "citation_coverage": rate(citation_passes, citation_cases),
    }
    return metrics, failures


def build_retrieval_strategies(pipeline: RAGPipeline, top_k: int) -> Dict[str, Callable[[str], List[RetrievedChunk]]]:
    return {
        "dense": lambda question: pipeline.retrieve_dense(question)[:top_k],
        "sparse": lambda question: pipeline.retrieve_sparse(question)[:top_k],
        "hybrid": lambda question: pipeline.retrieve(question)[:top_k],
        "hybrid_rerank": lambda question: pipeline.rerank_chunks(
            question,
            pipeline.retrieve(question),
            top_n=top_k,
        ),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(title)
    for key, value in metrics.items():
        if isinstance(value, float):
            print("  {key}: {value:.3f}".format(key=key, value=value))
        else:
            print("  {key}: {value}".format(key=key, value=value))


def print_failures(label: str, failures: Sequence[dict], limit: int) -> None:
    if not failures:
        print("  {label}: 0".format(label=label))
        return

    print("  {label}: {count}".format(label=label, count=len(failures)))
    for failure in failures[:limit]:
        print("   - {question}".format(question=failure["question"]))


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval and answer-quality evaluation against a JSONL dataset.")
    parser.add_argument(
        "--dataset",
        default="data/eval/sample_questions.jsonl",
        help="Path to a JSONL file with question plus gold_chunk_ids or gold_filenames",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many retrieved chunks to score")
    parser.add_argument(
        "--mode",
        choices=tuple(RETRIEVAL_MODES) + ("all",),
        default="all",
        help="Which retrieval baseline to evaluate",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also run answer-quality checks for rows with answer_contains or expected_refusal",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for a JSON report with metrics and failing examples",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=5,
        help="How many failing questions to print per section",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found: {path}".format(path=dataset_path))

    examples = load_jsonl(dataset_path)
    if not examples:
        raise ValueError("Dataset is empty: {path}".format(path=dataset_path))

    pipeline = RAGPipeline()
    strategies = build_retrieval_strategies(pipeline, args.top_k)
    selected_modes = list(RETRIEVAL_MODES) if args.mode == "all" else [args.mode]

    report = {
        "dataset": str(dataset_path),
        "top_k": args.top_k,
        "retrieval": {},
        "generation": None,
    }

    print("Dataset: {path}".format(path=dataset_path))
    print("Rows: {count}".format(count=len(examples)))

    for mode in selected_modes:
        print("")
        metrics, failures = evaluate_retrieval(examples, strategies[mode], args.top_k)
        print_metrics("Retrieval baseline: {mode}".format(mode=mode), metrics)
        print_failures("Failures", failures, args.failure_limit)
        report["retrieval"][mode] = {
            "metrics": metrics,
            "failures": failures,
        }

    if args.with_generation:
        print("")
        metrics, failures = evaluate_generation(examples, pipeline.answer)
        print_metrics("Generation checks", metrics)
        print_failures("Failures", failures, args.failure_limit)
        report["generation"] = {
            "metrics": metrics,
            "failures": failures,
        }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        write_report(output_path, report)
        print("")
        print("Wrote JSON report to {path}".format(path=output_path))


if __name__ == "__main__":
    main()
