"""CLI entrypoint for running the MCP server."""

from multi_modal_financial.mcp.server import FinancialMcpServer

if __name__ == "__main__":
    server = FinancialMcpServer()
    server.run_stdio_server()
