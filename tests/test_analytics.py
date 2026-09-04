"""Tests for financial ratio calculator and multi-period comparator."""


from multi_modal_financial.analytics.comparator import PeriodComparator
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator
from multi_modal_financial.types import (
    FinancialMetric,
    FinancialStatementType,
    TableData,
)


class TestFinancialRatioCalculator:
    """Unit tests for FinancialRatioCalculator."""

    def test_compute_margins(self):
        margins = FinancialRatioCalculator.compute_margins(
            revenue=1000.0, gross_profit=450.0, operating_income=250.0, net_income=180.0
        )
        assert margins["gross_margin"] == 45.0
        assert margins["operating_margin"] == 25.0
        assert margins["net_margin"] == 18.0

    def test_compute_liquidity(self):
        liq = FinancialRatioCalculator.compute_liquidity(
            current_assets=500.0,
            current_liabilities=250.0,
            cash_and_equivalents=100.0,
            marketable_securities=50.0,
            accounts_receivable=75.0,
        )
        assert liq["current_ratio"] == 2.0
        assert liq["quick_ratio"] == 0.9  # (100 + 50 + 75) / 250 = 225 / 250 = 0.9

    def test_compute_solvency(self):
        solv = FinancialRatioCalculator.compute_solvency(
            total_debt=400.0, total_equity=800.0, total_assets=1600.0
        )
        assert solv["debt_to_equity"] == 0.5
        assert solv["debt_to_assets"] == 0.25

    def test_compute_from_table(self):
        tbl = TableData(
            table_id="is_1",
            headers=["Metric", "2025"],
            rows=[
                ["Total Revenue", "$1,000"],
                ["Gross Profit", "$400"],
                ["Operating Income", "$200"],
                ["Net Income", "$150"],
            ],
            statement_type=FinancialStatementType.INCOME_STATEMENT,
        )
        ratios = FinancialRatioCalculator.compute_from_table(tbl, col_idx=1)
        assert ratios.gross_margin_pct == 40.0
        assert ratios.operating_margin_pct == 20.0
        assert ratios.net_margin_pct == 15.0

    def test_compute_from_document(self):
        from multi_modal_financial.types import Chunk, Document, DocumentMetadata, ModalType

        tbl = TableData(
            title="Statement of Operations",
            headers=["Metric", "2025"],
            rows=[
                ["Total Revenue", "$2,000"],
                ["Gross Profit", "$1,000"],
                ["Operating Income", "$500"],
                ["Net Income", "$400"],
            ],
            statement_type=FinancialStatementType.INCOME_STATEMENT,
        )
        chunk = Chunk(
            chunk_id="chk_1",
            doc_id="doc_1",
            content=tbl.to_markdown(),
            modal_type=ModalType.TABLE,
            table_data=tbl,
        )
        doc = Document(
            doc_id="doc_1",
            metadata=DocumentMetadata(doc_id="doc_1", filename="doc1.txt"),
            chunks=[chunk],
        )
        ratios = FinancialRatioCalculator.compute_from_document(doc)
        assert ratios.gross_margin_pct == 50.0
        assert ratios.operating_margin_pct == 25.0
        assert ratios.net_margin_pct == 20.0


class TestPeriodComparator:
    """Unit tests for PeriodComparator."""

    def test_calculate_variance_growth(self):
        var = PeriodComparator.calculate_variance("Revenue", base_val=1000.0, compare_val=1250.0)
        assert var.absolute_change == 250.0
        assert var.percentage_change == 25.0
        assert var.trend == "UP"

    def test_calculate_variance_decline(self):
        var = PeriodComparator.calculate_variance("Net Income", base_val=200.0, compare_val=150.0)
        assert var.absolute_change == -50.0
        assert var.percentage_change == -25.0
        assert var.trend == "DOWN"

    def test_calculate_variance_flat(self):
        var = PeriodComparator.calculate_variance("Operating Margin", base_val=15.0, compare_val=15.0)
        assert var.absolute_change == 0.0
        assert var.percentage_change == 0.0
        assert var.trend == "FLAT"

    def test_compare_metrics(self):
        base_list = [
            FinancialMetric(name="Revenue", raw_value="100.0", value=100.0, unit="USD"),
            FinancialMetric(name="OpEx", raw_value="30.0", value=30.0, unit="USD"),
        ]
        comp_list = [
            FinancialMetric(name="Revenue", raw_value="115.0", value=115.0, unit="USD"),
            FinancialMetric(name="OpEx", raw_value="27.0", value=27.0, unit="USD"),
        ]
        variances = PeriodComparator.compare_metrics(base_list, comp_list)
        assert len(variances) == 2
        rev_var = next(v for v in variances if v.metric_name == "Revenue")
        assert rev_var.percentage_change == 15.0
        assert rev_var.trend == "UP"

        opex_var = next(v for v in variances if v.metric_name == "OpEx")
        assert opex_var.percentage_change == -10.0
        assert opex_var.trend == "DOWN"

    def test_compare_table_columns(self):
        tbl = TableData(
            table_id="tbl_comp",
            headers=["Line Item", "Q3 2025", "Q3 2024"],
            rows=[
                ["Revenue", "$1,200", "$1,000"],
                ["Gross Profit", "$600", "$500"],
            ],
        )
        variances = PeriodComparator.compare_table_columns(tbl, base_col=2, compare_col=1)
        assert len(variances) == 2
        assert variances[0].percentage_change == 20.0
        assert variances[0].trend == "UP"
