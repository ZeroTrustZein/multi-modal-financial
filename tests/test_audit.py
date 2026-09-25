"""Tests for grounding auditor, provenance reporting, and compliance ledgers."""

from pathlib import Path

from multi_modal_financial.grounding.audit import GroundingAuditor
from multi_modal_financial.types import (
    AgentResponse,
    CacheHitType,
    Citation,
    GroundingStatus,
    GroundingVerdict,
    ModalType,
    RerankExplanation,
)


class TestGroundingAuditor:
    """Unit tests for GroundingAuditor."""

    def test_audit_clean_response(self):
        auditor = GroundingAuditor(max_allowed_hallucination_rate=0.05)
        resp = AgentResponse(
            query="What was revenue?",
            answer="Revenue was $1,250 million [1].",
            citations=[
                Citation(
                    citation_id="cite_1",
                    chunk_id="c1",
                    doc_id="d1",
                    page_number=1,
                    quote="Revenue reached $1,250 million",
                    modal_type=ModalType.TEXT,
                    confidence=0.95,
                )
            ],
            grounding_verdicts=[
                GroundingVerdict(
                    claim="Revenue was $1,250 million",
                    citations=[],
                    is_supported=True,
                    support_score=0.95,
                    status=GroundingStatus.FULLY_SUPPORTED,
                    unsupported_numbers=[],
                )
            ],
            retrieved_chunks=[],
            execution_time_ms=15.0,
            overall_confidence=0.95,
        )

        record = auditor.audit_response(resp)
        assert record["status"] == "PASS"
        assert record["total_claims"] == 1
        assert record["supported_claims"] == 1
        assert len(record["unsupported_numbers"]) == 0

    def test_audit_hallucinated_response(self):
        auditor = GroundingAuditor(max_allowed_hallucination_rate=0.05)
        resp = AgentResponse(
            query="What was revenue and profit?",
            answer="Revenue was $9999M and profit was $8888M.",
            citations=[],
            grounding_verdicts=[
                GroundingVerdict(
                    claim="Revenue was $9999M",
                    citations=[],
                    is_supported=False,
                    support_score=0.0,
                    status=GroundingStatus.UNSUPPORTED,
                    unsupported_numbers=["9999"],
                ),
                GroundingVerdict(
                    claim="profit was $8888M",
                    citations=[],
                    is_supported=False,
                    support_score=0.0,
                    status=GroundingStatus.UNSUPPORTED,
                    unsupported_numbers=["8888"],
                ),
            ],
            retrieved_chunks=[],
            execution_time_ms=10.0,
            overall_confidence=0.0,
        )

        record = auditor.audit_response(resp)
        assert record["status"] == "FAIL"
        assert record["unsupported_claims"] == 2
        assert set(record["unsupported_numbers"]) == {"9999", "8888"}

    def test_audit_batch_report_and_export(self, tmp_path: Path):
        auditor = GroundingAuditor(max_allowed_hallucination_rate=0.10)

        # 1 valid response
        resp_clean = AgentResponse(
            query="Apple Q3 rev",
            answer="Apple revenue was $94.9 billion.",
            citations=[],
            grounding_verdicts=[
                GroundingVerdict(
                    claim="Apple revenue was $94.9 billion",
                    citations=[],
                    is_supported=True,
                    support_score=0.92,
                    status=GroundingStatus.FULLY_SUPPORTED,
                    unsupported_numbers=[],
                )
            ],
            retrieved_chunks=[],
            execution_time_ms=20.0,
            overall_confidence=0.92,
        )

        # 1 partially flawed response
        resp_warn = AgentResponse(
            query="Microsoft guidance",
            answer="Guidance implies 15% margin.",
            citations=[],
            grounding_verdicts=[
                GroundingVerdict(
                    claim="Guidance implies 15% margin",
                    citations=[],
                    is_supported=False,
                    support_score=0.4,
                    status=GroundingStatus.PARTIALLY_SUPPORTED,
                    unsupported_numbers=[],
                )
            ],
            retrieved_chunks=[],
            execution_time_ms=18.0,
            overall_confidence=0.4,
        )

        report = auditor.audit_batch([resp_clean, resp_warn])
        assert report.total_queries == 2
        assert report.total_claims == 2
        assert report.supported_claims == 1
        assert report.partially_supported_claims == 1
        assert report.hallucination_rate == 0.0  # 0 unsupported, 1 partially supported
        assert report.compliance_status in {"PASS", "WARN"}

        # Export test
        json_out = tmp_path / "audit.json"
        md_out = tmp_path / "audit.md"

        auditor.export_report(report, json_out)
        auditor.export_report(report, md_out)

        assert json_out.exists()
        assert md_out.exists()
        assert "Financial RAG Grounding & Factual Audit Report" in md_out.read_text(
            encoding="utf-8"
        )

    def test_audit_with_cache_and_reranker_telemetry(self):
        auditor = GroundingAuditor()
        resp1 = AgentResponse(
            query="Apple Q3 net sales",
            answer="Net sales was $94.9B",
            citations=[],
            grounding_verdicts=[
                GroundingVerdict(
                    claim="Net sales was $94.9B",
                    citations=[],
                    is_supported=True,
                    support_score=0.95,
                    status=GroundingStatus.FULLY_SUPPORTED,
                )
            ],
            retrieved_chunks=[],
            cache_hit=True,
            cache_type=CacheHitType.SEMANTIC,
            cache_similarity=0.91,
            rerank_explanations=[
                RerankExplanation(
                    chunk_id="c1",
                    initial_rank=3,
                    final_rank=1,
                    initial_score=0.4,
                    final_score=0.85,
                    reasons=["High concept overlap"],
                )
            ],
        )

        record = auditor.audit_response(resp1)
        assert record["cache_hit"] is True
        assert record["cache_type"] == "semantic"
        assert record["rerank_count"] == 1

        report = auditor.audit_batch([resp1])
        assert report.cache_hits == 1
        assert report.semantic_cache_hits == 1
        assert report.exact_cache_hits == 0
        assert report.reranked_queries == 1
        md = report.to_markdown()
        assert "**Cache Hits**: 1" in md
        assert "**Reranked Queries**: 1" in md
