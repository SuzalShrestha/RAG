from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import RAGPipeline


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def reciprocal_rank(gold_ids, predicted_ids):
    for index, chunk_id in enumerate(predicted_ids, start=1):
        if chunk_id in gold_ids:
            return 1.0 / float(index)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against a JSONL dataset.")
    parser.add_argument(
        "--dataset",
        default="data/eval/questions.jsonl",
        help="Path to a JSONL file with question and gold_chunk_ids keys",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many retrieved chunks to score")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found: {path}".format(path=dataset_path))

    pipeline = RAGPipeline()
    examples = list(load_jsonl(dataset_path))
    if not examples:
        raise ValueError("Dataset is empty: {path}".format(path=dataset_path))

    hits = 0
    recall_sum = 0.0
    mrr_sum = 0.0

    for example in examples:
        gold_ids = set(example.get("gold_chunk_ids", []))
        retrieved = pipeline.retrieve(example["question"])[: args.top_k]
        predicted_ids = [chunk.chunk_id for chunk in retrieved]

        if gold_ids.intersection(predicted_ids):
            hits += 1

        if gold_ids:
            recall_sum += len(gold_ids.intersection(predicted_ids)) / float(len(gold_ids))
            mrr_sum += reciprocal_rank(gold_ids, predicted_ids)

    count = float(len(examples))
    print("Examples: {count}".format(count=len(examples)))
    print("HitRate@{k}: {score:.3f}".format(k=args.top_k, score=hits / count))
    print("Recall@{k}: {score:.3f}".format(k=args.top_k, score=recall_sum / count))
    print("MRR@{k}: {score:.3f}".format(k=args.top_k, score=mrr_sum / count))


if __name__ == "__main__":
    main()
