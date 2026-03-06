from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.utils.models import AnswerResult, Citation, RetrievedChunk


SYSTEM_PROMPT = """You are a retrieval-augmented assistant.
Answer only from the supplied context.
Every factual claim must include an inline citation like [C1] or [C2].
If the context is insufficient, say: I don't know based on the uploaded documents.
Do not cite chunks that were not provided.
"""


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Question:\n{question}\n\nContext:\n{context}\n\n"
                    "Write a concise Markdown answer with inline citations.",
                ),
            ]
        )

    def answer(self, question: str, chunks: List[RetrievedChunk]) -> AnswerResult:
        if not chunks:
            return AnswerResult(
                answer_markdown="I don't know based on the uploaded documents.",
                abstained=True,
            )

        context = self._format_context(chunks)
        message = self.prompt.invoke({"question": question, "context": context})
        response = self.llm.invoke(message)
        answer_text = str(response.content).strip()
        abstained = "i don't know based on the uploaded documents" in answer_text.lower()
        if not abstained and "[C" not in answer_text:
            labels = ", ".join("[{label}]".format(label=citation.label) for citation in self._build_citations(chunks))
            answer_text = "{answer}\n\nSources: {labels}".format(answer=answer_text, labels=labels)

        return AnswerResult(
            answer_markdown=answer_text,
            citations=self._build_citations(chunks),
            used_chunks=chunks,
            abstained=abstained,
            raw_response=response,
        )

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        formatted = []
        for index, chunk in enumerate(chunks, start=1):
            label = "C{index}".format(index=index)
            chunk.metadata["citation_label"] = label
            page_number = chunk.metadata.get("page_number")
            section_heading = chunk.metadata.get("section_heading")
            source_hint = "page={page}".format(page=page_number) if page_number else "section={section}".format(
                section=section_heading or "n/a"
            )
            formatted.append(
                "[{label}] file={filename} {source_hint}\n{content}".format(
                    label=label,
                    filename=chunk.metadata.get("filename", "unknown"),
                    source_hint=source_hint,
                    content=chunk.text,
                )
            )
        return "\n\n".join(formatted)

    def _build_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        citations = []
        for index, chunk in enumerate(chunks, start=1):
            citations.append(
                Citation(
                    label="C{index}".format(index=index),
                    chunk_id=chunk.chunk_id,
                    filename=str(chunk.metadata.get("filename", "unknown")),
                    page_number=chunk.metadata.get("page_number"),
                    section_heading=chunk.metadata.get("section_heading"),
                    excerpt=chunk.text[:280],
                )
            )
        return citations
