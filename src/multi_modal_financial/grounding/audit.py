"""Grounding audit, provenance ledger, and hallucination compliance reporting subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

from multi_modal_financial.types import AgentResponse, GroundingStatus, GroundingVerdict


@dataclass
class GroundingAuditReport:
    """Comprehensive compliance and factual grounding audit report."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_queries: int = 0
    total_claims: int = 0
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    hallucination_rate: float = 0.0
    mean_confidence: float = 0.0
    compliance_status: str = "PASS"  # PASS, WARN, FAIL
    flagged_numbers: list[str] = field(default_factory=list)
    query_verdicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# Financial RAG Grounding & Factual Audit Report",
            f"**Audit Timestamp**: {self.timestamp}",
            f"**Compliance Status**: `{self.compliance_status}`",
            "",
            "## Summary Metrics",
            f"- **Total Queries Audited**: {self.total_queries}",
            f"- **Total Claims Evaluated**: {self.total_claims}",
            f"- **Supported Claims**: {self.supported_claims} ({(self.supported_claims / max(self.total_claims, 1))*100:.1f}%)",
            f"- **Partially Supported Claims**: {self.partially_supported_claims}",
            f"- **Unsupported Claims**: {self.unsupported_claims}",
            f"- **Hallucination Rate**: {self.hallucination_rate * 100:.2f}%",
            f"- **Mean Support Confidence**: {self.mean_confidence:.4f}",
            "",
        ]

        if self.flagged_numbers:
            lines.extend([
                "## Flagged Unsupported Figures",
                ", ".join(f"`{num}`" for num in self.flagged_numbers),
                "",
            ])

        lines.append("## Query Details")
        lines.append("| Query | Claims | Supported | Hallucinated Figures | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        for q in self.query_verdicts:
            q_text = q.get("query", "")[:50].replace("|", "\\|")
            q_claims = q.get("total_claims", 0)
            q_supp = q.get("supported_claims", 0)
            q_bad_nums = ", ".join(q.get("unsupported_numbers", [])) or "None"
            q_status = q.get("status", "OK")
            lines.append(f"| {q_text} | {q_claims} | {q_supp} | {q_bad_nums} | {q_status} |")

        return "\n".join(lines)


class GroundingAuditor:
    """Audits RAG pipelines for hallucinations and generates compliance ledgers."""

    def __init__(self, max_allowed_hallucination_rate: float = 0.05, min_confidence: float = 0.60):
        self.max_allowed_hallucination_rate = max_allowed_hallucination_rate
        self.min_confidence = min_confidence

    def audit_response(self, response: AgentResponse) -> dict:
        """Audit an individual AgentResponse for factual integrity."""
        verdicts: list[GroundingVerdict] = response.grounding_verdicts
        total_claims = len(verdicts)
        supported = sum(1 for v in verdicts if v.is_supported)
        unsupported_nums = list(
            {num for v in verdicts for num in v.unsupported_numbers}
        )

        has_hallucinated_nums = len(unsupported_nums) > 0
        unsupported_claims = sum(1 for v in verdicts if v.status == GroundingStatus.UNSUPPORTED)
        partially_supported = sum(1 for v in verdicts if v.status == GroundingStatus.PARTIALLY_SUPPORTED)

        status = "FAIL" if has_hallucinated_nums or (total_claims > 0 and supported == 0) else "PASS"

        return {
            "query": response.query,
            "total_claims": total_claims,
            "supported_claims": supported,
            "partially_supported": partially_supported,
            "unsupported_claims": unsupported_claims,
            "unsupported_numbers": unsupported_nums,
            "confidence": response.overall_confidence,
            "status": status,
        }

    def audit_batch(self, responses: list[AgentResponse]) -> GroundingAuditReport:
        """Generate full audit report across a batch of responses."""
        total_queries = len(responses)
        total_claims = 0
        supported_claims = 0
        partially_supported = 0
        unsupported_claims = 0
        all_flagged_nums: list[str] = []
        confidences: list[float] = []
        query_records: list[dict] = []

        for resp in responses:
            record = self.audit_response(resp)
            query_records.append(record)

            total_claims += record["total_claims"]
            supported_claims += record["supported_claims"]
            partially_supported += record["partially_supported"]
            unsupported_claims += record["unsupported_claims"]
            all_flagged_nums.extend(record["unsupported_numbers"])
            confidences.append(resp.overall_confidence)

        unique_flagged = sorted(set(all_flagged_nums))
        hallucination_rate = (
            (unsupported_claims / total_claims) if total_claims > 0 else 0.0
        )
        mean_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        if hallucination_rate <= self.max_allowed_hallucination_rate and not unique_flagged:
            compliance = "PASS"
        elif hallucination_rate <= self.max_allowed_hallucination_rate * 2.0:
            compliance = "WARN"
        else:
            compliance = "FAIL"

        return GroundingAuditReport(
            total_queries=total_queries,
            total_claims=total_claims,
            supported_claims=supported_claims,
            partially_supported_claims=partially_supported,
            unsupported_claims=unsupported_claims,
            hallucination_rate=round(hallucination_rate, 4),
            mean_confidence=round(mean_conf, 4),
            compliance_status=compliance,
            flagged_numbers=unique_flagged,
            query_verdicts=query_records,
        )

    def export_report(self, report: GroundingAuditReport, output_path: str | Path) -> Path:
        """Export audit report to file (.json or .md)."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.suffix.lower() == ".json":
            target.write_text(report.to_json(), encoding="utf-8")
        else:
            target.write_text(report.to_markdown(), encoding="utf-8")

        return target
