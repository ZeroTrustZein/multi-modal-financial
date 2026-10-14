"""Model Context Protocol (MCP) server exposing multi-modal financial RAG capabilities."""

from __future__ import annotations

import json
import sys
from typing import Any

from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.types import AgentQuery, RetrievalStrategy

SERVER_NAME = "multi-modal-financial-mcp"
SERVER_VERSION = "0.1.0"

SUPPORTED_TOOLS = [
    {
        "name": "query_financial_filings",
        "description": "Execute hybrid sparse-dense RAG query over financial disclosures with citation grounding and neural reranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Financial or accounting question."},
                "ticker": {"type": "string", "description": "Optional ticker symbol filter."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of candidate chunks.",
                    "default": 3,
                },
                "strategy": {
                    "type": "string",
                    "enum": ["hybrid_rrf", "hybrid_convex", "dense", "sparse"],
                    "default": "hybrid_rrf",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "compute_financial_ratios",
        "description": "Calculate GAAP profitability, liquidity, and solvency ratios for a company filing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Company ticker symbol."},
                "year": {"type": "integer", "default": 2025},
                "period": {"type": "string", "default": "Q3"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_filing_periods",
        "description": "Perform horizontal YoY or QoQ financial variance analysis across two reporting periods.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Company ticker symbol."},
                "base_period": {"type": "string", "default": "Q2"},
                "compare_period": {"type": "string", "default": "Q3"},
                "year": {"type": "integer", "default": 2025},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "audit_factual_grounding",
        "description": "Audit claims against ingested filing ground truth to detect hallucinations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of factual financial assertions.",
                }
            },
            "required": ["claims"],
        },
    },
]


class FinancialMcpServer:
    """Stdio Model Context Protocol (MCP) server for financial RAG engine."""

    def __init__(self, orchestrator: FinancialPipelineOrchestrator | None = None):
        self.orchestrator = orchestrator or FinancialPipelineOrchestrator(
            config=OrchestratorConfig(enable_semantic_cache=True)
        )
        self.generator = SyntheticFilingGenerator()
        self._ensure_default_corpus()

    def _ensure_default_corpus(self) -> None:
        """Seed index with synthetic sample documents if empty."""
        if self.orchestrator.index.total_chunks == 0:
            doc_q2 = self.generator.generate_filing_document(ticker="AAPL", year=2025, period="Q2")
            doc_q3 = self.generator.generate_filing_document(ticker="AAPL", year=2025, period="Q3")
            self.orchestrator.index.index_document(doc_q2)
            self.orchestrator.index.index_document(doc_q3)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch and execute an MCP tool invocation."""
        if name == "query_financial_filings":
            query_str = arguments["query"]
            ticker = arguments.get("ticker")
            top_k = arguments.get("top_k", 3)
            strat_name = arguments.get("strategy", "hybrid_rrf")

            strategy = RetrievalStrategy.HYBRID_RRF
            if strat_name == "dense":
                strategy = RetrievalStrategy.DENSE
            elif strat_name == "sparse":
                strategy = RetrievalStrategy.SPARSE
            elif strat_name == "hybrid_convex":
                strategy = RetrievalStrategy.HYBRID_CONVEX

            agent_q = AgentQuery(
                query_str=query_str,
                ticker_filter=ticker,
                top_k=top_k,
                retrieval_strategy=strategy,
            )
            resp = self.orchestrator.run_query(agent_q)
            return {
                "answer": resp.answer,
                "grounded": resp.is_fully_grounded(),
                "confidence": resp.overall_confidence,
                "citations": [c.model_dump() for c in resp.citations],
                "cache_hit": resp.cache_hit,
            }

        elif name == "compute_financial_ratios":
            ticker = arguments["ticker"].upper()
            year = arguments.get("year", 2025)
            period = arguments.get("period", "Q3")

            matching_doc = None
            for d in self.orchestrator.index.documents.values():
                if (
                    d.metadata.ticker == ticker
                    and d.metadata.year == year
                    and d.metadata.period == period
                ):
                    matching_doc = d
                    break

            if not matching_doc:
                matching_doc = self.generator.generate_filing_document(
                    ticker=ticker, year=year, period=period
                )
                self.orchestrator.index.index_document(matching_doc)

            ratios = self.orchestrator.analyze_document_ratios(matching_doc.doc_id)
            return ratios.to_dict()

        elif name == "compare_filing_periods":
            ticker = arguments["ticker"].upper()
            base_p = arguments.get("base_period", "Q2")
            comp_p = arguments.get("compare_period", "Q3")
            year = arguments.get("year", 2025)

            def get_or_create(p: str):
                for d in self.orchestrator.index.documents.values():
                    if (
                        d.metadata.ticker == ticker
                        and d.metadata.year == year
                        and d.metadata.period == p
                    ):
                        return d
                new_d = self.generator.generate_filing_document(ticker=ticker, year=year, period=p)
                self.orchestrator.index.index_document(new_d)
                return new_d

            doc_base = get_or_create(base_p)
            doc_comp = get_or_create(comp_p)

            variances = self.orchestrator.compare_documents(doc_base.doc_id, doc_comp.doc_id)
            return {
                "ticker": ticker,
                "comparison": f"{base_p} vs {comp_p} {year}",
                "variances": [v.to_dict() for v in variances],
            }

        elif name == "audit_factual_grounding":
            claims = arguments.get("claims", [])
            report = self.orchestrator.audit_queries(queries=claims)
            return {
                "passed": report.compliance_status == "PASS",
                "total_claims": report.total_claims,
                "hallucination_rate": report.hallucination_rate,
                "unsupported_claims": report.unsupported_claims,
            }

        raise ValueError(f"Unknown MCP tool: {name}")

    def handle_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process incoming JSON-RPC 2.0 MCP message."""
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {}},
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": SUPPORTED_TOOLS},
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            try:
                content = self.execute_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(content, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }

        elif method == "notifications/initialized":
            return None

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def run_stdio_server(self) -> None:
        """Run standard I/O loop reading JSON-RPC lines from stdin and writing to stdout."""
        for line in sys.stdin:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                req = json.loads(line_str)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
