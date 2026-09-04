"""Tests for financial data validation and statement reconciliation."""


from multi_modal_financial.data.validator import (
    FinancialTableValidator,
    IssueSeverity,
    StatementReconciler,
)
from multi_modal_financial.types import FinancialStatementType, TableData


class TestFinancialTableValidator:
    """Unit tests for FinancialTableValidator."""

    def test_validate_valid_table_structure(self):
        tbl = TableData(
            table_id="tbl_ok",
            headers=["Metric", "2024", "2025"],
            rows=[
                ["Revenue", "$1,000", "$1,200"],
                ["Operating Income", "$200", "$250"],
            ],
        )
        report = FinancialTableValidator.validate_table_structure(tbl)
        assert report.is_valid
        assert len(report.errors()) == 0

    def test_validate_ragged_table_columns(self):
        tbl = TableData(
            table_id="tbl_bad",
            headers=["Metric", "2024", "2025"],
            rows=[
                ["Revenue", "$1,000", "$1,200"],
                ["Missing Col", "$200"],  # only 2 columns instead of 3
            ],
        )
        report = FinancialTableValidator.validate_table_structure(tbl)
        assert not report.is_valid
        assert len(report.errors()) == 1
        assert report.errors()[0].row_idx == 1

    def test_validate_missing_headers(self):
        tbl = TableData(
            table_id="tbl_no_header",
            headers=[],
            rows=[["Revenue", "100"]],
        )
        report = FinancialTableValidator.validate_table_structure(tbl)
        assert not report.is_valid
        assert any("missing headers" in e.message.lower() for e in report.errors())

    def test_validate_numeric_bounds_margin_exceeded(self):
        tbl = TableData(
            table_id="tbl_bounds",
            headers=["Metric", "Value"],
            rows=[
                ["Gross Margin", "145.0%"],
                ["Total Revenue", "-$500"],
            ],
        )
        issues = FinancialTableValidator.validate_numeric_bounds(tbl)
        assert len(issues) >= 2
        # One warning for margin > 100%, one error for negative revenue
        warnings = [i for i in issues if i.severity == IssueSeverity.WARNING]
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert len(warnings) == 1
        assert "exceeds 100%" in warnings[0].message
        assert len(errors) == 1
        assert "negative balance" in errors[0].message


class TestStatementReconciler:
    """Unit tests for StatementReconciler."""

    def test_reconcile_balanced_balance_sheet(self):
        tbl = TableData(
            table_id="bs_bal",
            headers=["Line Item", "As of Dec 31, 2025"],
            rows=[
                ["Total Assets", "$10,000"],
                ["Total Liabilities", "$6,000"],
                ["Total Stockholders' Equity", "$4,000"],
            ],
            statement_type=FinancialStatementType.BALANCE_SHEET,
        )
        result = StatementReconciler.reconcile_balance_sheet(tbl, col_idx=1)
        assert result.is_balanced
        assert result.expected_value == 10000.0
        assert result.actual_value == 10000.0
        assert result.difference == 0.0

    def test_reconcile_unbalanced_balance_sheet(self):
        tbl = TableData(
            table_id="bs_unbal",
            headers=["Line Item", "As of Dec 31, 2025"],
            rows=[
                ["Total Assets", "$10,000"],
                ["Total Liabilities", "$5,000"],
                ["Total Stockholders' Equity", "$4,000"],
            ],
            statement_type=FinancialStatementType.BALANCE_SHEET,
        )
        result = StatementReconciler.reconcile_balance_sheet(tbl, col_idx=1)
        assert not result.is_balanced
        assert result.expected_value == 10000.0
        assert result.actual_value == 9000.0
        assert result.difference == 1000.0

    def test_reconcile_income_statement(self):
        tbl = TableData(
            table_id="is_bal",
            headers=["Line Item", "Q3 2025"],
            rows=[
                ["Total Revenue", "$5,000"],
                ["Cost of Goods Sold", "$3,000"],
                ["Gross Profit", "$2,000"],
            ],
            statement_type=FinancialStatementType.INCOME_STATEMENT,
        )
        result = StatementReconciler.reconcile_income_statement(tbl, col_idx=1)
        assert result.is_balanced
        assert result.expected_value == 2000.0
        assert result.actual_value == 2000.0
        assert result.difference == 0.0
