"""End-to-end Financial Document RAG Pipeline with citation grounding."""

from __future__ import annotations

import time
from pathlib import Path

from multi_modal_financial.agent.router import QueryRouter
from multi_modal_financial.grounding.citation import CitationGrounder
from multi_modal_financial.grounding.verifier import GroundingVerifier
from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.parsing.extractor import FinancialDocumentParser
from multi_modal_financial.retrieval.fusion import HybridRetriever
from multi_modal_financial.retrieval.reranker import FinancialReranker
from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    Document,
    DocumentMetadata,
    ScoredChunk,
)


class FinancialRAGPipeline:
    """End-to-end multi-modal financial RAG system."""

    def __init__(
        self,
        hybrid_index: HybridIndex | None = None,
        retriever: HybridRetriever | None = None,
        reranker: FinancialReranker | None = None,
        verifier: GroundingVerifier | None = None,
        router: QueryRouter | None = None,
    ):
        self.index = hybrid_index or HybridIndex()
        self.retriever = retriever or HybridRetriever(self.index)
        self.reranker = reranker or FinancialReranker()
        self.verifier = verifier or GroundingVerifier()
        self.router = router or QueryRouter()
        self.parser = FinancialDocumentParser()

    def ingest_text(
        self,
        text: str,
        doc_id: str,
        filename: str = "document.txt",
        ticker: str | None = None,
        period: str | None = None,
        year: int | None = None,
    ) -> Document:
        """Parse raw text and index the generated chunks."""
        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            ticker=ticker,
            period=period,
            year=year,
        )
        doc = self.parser.parse_text(text, metadata=meta)
        self.index.index_document(doc)
        return doc

    def ingest_file(self, file_path: str | Path, ticker: str | None = None) -> Document:
        """Ingest either a PDF or text file."""
        path = Path(file_path)
        doc_id = path.stem

        if path.suffix.lower() == ".pdf":
            doc = self.parser.parse_pdf(path)
            if ticker:
                doc.metadata.ticker = ticker
            self.index.index_document(doc)
            return doc

        content = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(content, doc_id=doc_id, filename=path.name, ticker=ticker)

    def ingest_directory(
        self,
        directory_path: str | Path,
        pattern: str = "*.*",
        default_ticker: str | None = None,
    ) -> list[Document]:
        """Batch ingest all financial documents in a folder matching the glob pattern."""
        p = Path(directory_path)
        if not p.is_dir():
            return []

        ingested_docs: list[Document] = []
        for file in p.glob(pattern):
            if file.suffix.lower() in {".txt", ".md", ".pdf", ".csv"}:
                try:
                    doc = self.ingest_file(file, ticker=default_ticker)
                    ingested_docs.append(doc)
                except Exception:
                    continue

        return ingested_docs

    def query(
        self,
        query_input: str | AgentQuery,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> AgentResponse:
        """Run full financial RAG cycle: routing -> retrieval -> reranking -> synthesis -> grounding."""
        start_time = time.perf_counter()

        if isinstance(query_input, str):
            agent_query = self.router.build_agent_query(query_input, top_k=top_k)
        else:
            agent_query = query_input

        # 1. Retrieve candidates
        scored_candidates = self.retriever.retrieve(agent_query, top_k=agent_query.top_k * 2)

        # 2. Rerank candidates
        if use_reranker:
            final_candidates = self.reranker.rerank(
                agent_query.query_str, scored_candidates, top_k=agent_query.top_k
            )
        else:
            final_candidates = scored_candidates[: agent_query.top_k]

        retrieved_chunks = [sc.chunk for sc in final_candidates]

        # 3. Synthesize answer with source attribution
        answer = self._synthesize_answer(agent_query.query_str, final_candidates)

        # 4. Extract citations and ground claims
        _, citations = CitationGrounder.extract_citations_from_text(answer, retrieved_chunks)
        if not citations and retrieved_chunks:
            # Auto-ground if explicit brackets were omitted in synthesis
            citations = CitationGrounder.auto_ground_claim(answer, retrieved_chunks)

        # 5. Verify factual grounding and check for unsupported numbers
        grounding_verdicts = self.verifier.verify_response(answer, retrieved_chunks)

        # Calculate confidence
        if grounding_verdicts:
            overall_conf = sum(v.support_score for v in grounding_verdicts) / len(
                grounding_verdicts
            )
        else:
            overall_conf = 0.5 if retrieved_chunks else 0.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            query=agent_query.query_str,
            answer=answer,
            citations=citations,
            grounding_verdicts=grounding_verdicts,
            retrieved_chunks=final_candidates,
            execution_time_ms=round(elapsed_ms, 2),
            overall_confidence=round(overall_conf, 4),
            metadata={
                "intent": agent_query.intent,
                "ticker": agent_query.ticker_filter,
                "period": agent_query.period_filter,
                "year": agent_query.year_filter,
                "candidate_count": len(final_candidates),
            },
        )

    def _synthesize_answer(self, query_str: str, scored_chunks: list[ScoredChunk]) -> str:
        """Deterministic context synthesizer mapping retrieved passages and tables to response."""
        if not scored_chunks:
            return "No relevant financial disclosures found matching query criteria."

        sections: list[str] = []
        for idx, item in enumerate(scored_chunks):
            ref_idx = idx + 1
            chunk = item.chunk
            if chunk.table_data:
                tbl = chunk.table_data
                header_info = f"Table '{tbl.title or 'Financial Data'}'"
                sections.append(f"According to {header_info} [{ref_idx}], {chunk.content.strip()}")
            elif chunk.figure_data:
                fig = chunk.figure_data
                sections.append(
                    f"Figure '{fig.caption or 'Chart'}' [{ref_idx}] illustrates: {fig.summary_text}"
                )
            else:
                sections.append(f"From disclosure [{ref_idx}]: {chunk.content.strip()}")

        return " ".join(sections)
