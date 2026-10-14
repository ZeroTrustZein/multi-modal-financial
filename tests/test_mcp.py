"""Tests for Financial MCP Server and JSON-RPC tool endpoints."""

from __future__ import annotations

import json

from multi_modal_financial.mcp.server import FinancialMcpServer


class TestFinancialMcpServer:
    def test_mcp_initialize(self) -> None:
        server = FinancialMcpServer()
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = server.handle_request(req)
        assert resp is not None
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "multi-modal-financial-mcp"
        assert "tools" in resp["result"]["capabilities"]

    def test_mcp_list_tools(self) -> None:
        server = FinancialMcpServer()
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = server.handle_request(req)
        assert resp is not None
        assert "tools" in resp["result"]
        tool_names = [t["name"] for t in resp["result"]["tools"]]
        assert "query_financial_filings" in tool_names
        assert "compute_financial_ratios" in tool_names
        assert "compare_filing_periods" in tool_names
        assert "audit_factual_grounding" in tool_names

    def test_mcp_call_query_tool(self) -> None:
        server = FinancialMcpServer()
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "query_financial_filings",
                "arguments": {"query": "What were net sales?", "ticker": "AAPL"},
            },
        }
        resp = server.handle_request(req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        content_text = resp["result"]["content"][0]["text"]
        payload = json.loads(content_text)
        assert "answer" in payload
        assert "citations" in payload

    def test_mcp_call_ratios_tool(self) -> None:
        server = FinancialMcpServer()
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "compute_financial_ratios",
                "arguments": {"ticker": "AAPL", "year": 2025, "period": "Q3"},
            },
        }
        resp = server.handle_request(req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        content_text = resp["result"]["content"][0]["text"]
        payload = json.loads(content_text)
        assert "gross_margin_pct" in payload
        assert "operating_margin_pct" in payload

    def test_mcp_call_compare_tool(self) -> None:
        server = FinancialMcpServer()
        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "compare_filing_periods",
                "arguments": {
                    "ticker": "AAPL",
                    "base_period": "Q2",
                    "compare_period": "Q3",
                    "year": 2025,
                },
            },
        }
        resp = server.handle_request(req)
        assert resp is not None
        assert resp["result"]["isError"] is False
        content_text = resp["result"]["content"][0]["text"]
        payload = json.loads(content_text)
        assert "variances" in payload
        assert payload["ticker"] == "AAPL"

    def test_mcp_call_unknown_tool(self) -> None:
        server = FinancialMcpServer()
        req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}},
        }
        resp = server.handle_request(req)
        assert resp is not None
        assert resp["result"]["isError"] is True
