from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import RAGPipeline


def iter_input_paths(raw_paths):
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
                    yield child
        else:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local RAG indexes.")
    parser.add_argument("paths", nargs="+", help="Files or directories to index")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    summary = pipeline.index_paths(iter_input_paths(args.paths))
    print("Indexed files: {count}".format(count=summary.files_indexed))
    print("New chunks: {count}".format(count=summary.chunks_indexed))
    print("Total documents: {count}".format(count=summary.total_documents))
    print("Total chunks: {count}".format(count=summary.total_chunks))
    if summary.failed_files:
        print("Skipped files: {count}".format(count=len(summary.failed_files)))
        for failure in summary.failed_files:
            print(" - {failure}".format(failure=failure))


if __name__ == "__main__":
    main()
