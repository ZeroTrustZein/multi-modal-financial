"""Financial table validation and accounting statement reconciliation subsystem."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from multi_modal_financial.types import FinancialStatementType, TableData


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Individual data anomaly or integrity issue detected in financial data."""

    severity: IssueSeverity
    message: str
    row_idx: int | None = None
    col_name: str | None = None
    field_value: Any = None


@dataclass
class ValidationReport:
    """Structured report of table integrity validation."""

    is_valid: bool
    table_title: str
    total_issues: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]


@dataclass
class ReconciliationResult:
    """Outcome of accounting identity verification."""

    statement_type: FinancialStatementType
    equation: str
    expected_value: float
    actual_value: float
    difference: float
    is_balanced: bool
    details: dict[str, float] = field(default_factory=dict)


class FinancialTableValidator:
    """Validates structural and numeric integrity of financial tables."""

    @classmethod
    def validate_table_structure(cls, table: TableData) -> ValidationReport:
        """Inspect table headers, column alignments, and row integrity."""
        issues: list[ValidationIssue] = []

        if not table.headers:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    message="Table is missing headers.",
                )
            )
            return ValidationReport(
                is_valid=False,
                table_title=table.title or "",
                total_issues=1,
                issues=issues,
            )

        expected_cols = len(table.headers)
        if expected_cols < 2:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    message=f"Table only has {expected_cols} column; financial tables typically require at least label and value columns.",
                )
            )

        # Check rows column alignment
        for r_idx, row in enumerate(table.rows):
            if len(row) != expected_cols:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        message=f"Row {r_idx} column count ({len(row)}) does not match header count ({expected_cols}).",
                        row_idx=r_idx,
                    )
                )

        # Check for empty table content
        if not table.rows:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    message="Table has no data rows.",
                )
            )

        # Check for non-empty cell contents
        empty_rows = 0
        for row in table.rows:
            if all(not str(cell).strip() for cell in row):
                empty_rows += 1
        if empty_rows > 0:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    message=f"Table has {empty_rows} completely empty rows.",
                )
            )

        is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return ValidationReport(
            is_valid=is_valid,
            table_title=table.title or "",
            total_issues=len(issues),
            issues=issues,
        )

    @classmethod
    def validate_numeric_bounds(cls, table: TableData) -> list[ValidationIssue]:
        """Verify financial metric boundaries (e.g. margins <= 100%, non-negative revenues)."""
        issues: list[ValidationIssue] = []

        metrics = table.extract_metrics()
        for metric in metrics:
            val = metric.value
            if val is None:
                continue
            name_lower = metric.name.lower()
            is_pct = metric.scale == "percent" or "%" in metric.raw_value

            if is_pct:
                # Margins typically shouldn't exceed 100% (gross/operating/net)
                if ("margin" in name_lower or "tax rate" in name_lower) and val > 100.0:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            message=f"Margin metric '{metric.name}' value {val}% exceeds 100%.",
                            field_value=val,
                        )
                    )
            else:
                # Non-negative metrics in standard GAAP: Revenue, Total Assets, Cash
                if any(
                    k in name_lower
                    for k in [
                        "total revenue",
                        "net sales",
                        "total assets",
                        "cash and cash equivalents",
                    ]
                ):
                    if val < 0:
                        issues.append(
                            ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                message=f"Metric '{metric.name}' has invalid negative balance {val}.",
                                field_value=val,
                            )
                        )

        return issues


class StatementReconciler:
    """Reconciles standard accounting identities across financial table metrics."""

    @staticmethod
    def _find_row_value(table: TableData, col_idx: int, *keywords: str) -> float | None:
        """Find numerical value in a given column prioritizing exact label match over substring."""
        # Pass 1: exact matches
        for kw in keywords:
            kw_clean = kw.lower().strip()
            for row in table.rows:
                if not row:
                    continue
                if row[0].lower().strip() == kw_clean:
                    if col_idx < len(row):
                        from multi_modal_financial.types import FinancialMetric

                        metric = FinancialMetric.from_raw(name=row[0], raw_value=row[col_idx])
                        if metric.value is not None and not math.isnan(metric.value):
                            return metric.value

        # Pass 2: substring matches excluding 'current' for total assets/liabilities
        for kw in keywords:
            kw_clean = kw.lower().strip()
            for row in table.rows:
                if not row:
                    continue
                row_label = row[0].lower().strip()
                if kw_clean in ["total assets", "assets"] and "current" in row_label:
                    continue
                if kw_clean in ["total liabilities", "liabilities"] and "current" in row_label:
                    continue
                if kw_clean in row_label:
                    if col_idx < len(row):
                        from multi_modal_financial.types import FinancialMetric

                        metric = FinancialMetric.from_raw(name=row[0], raw_value=row[col_idx])
                        if metric.value is not None and not math.isnan(metric.value):
                            return metric.value
        return None

    @classmethod
    def reconcile_balance_sheet(
        cls, table: TableData, col_idx: int = 1, tolerance: float = 1.0
    ) -> ReconciliationResult:
        """Verify: Total Assets == Total Liabilities + Total Stockholders' Equity."""
        total_assets = cls._find_row_value(table, col_idx, "total assets", "assets")
        total_liabilities = cls._find_row_value(table, col_idx, "total liabilities", "liabilities")
        equity = cls._find_row_value(
            table,
            col_idx,
            "total stockholders' equity",
            "total shareholders' equity",
            "total equity",
            "equity",
        )
        liab_and_equity = cls._find_row_value(
            table,
            col_idx,
            "total liabilities and stockholders' equity",
            "total liabilities and equity",
        )

        details: dict[str, float] = {}
        if total_assets is not None:
            details["total_assets"] = total_assets
        if total_liabilities is not None:
            details["total_liabilities"] = total_liabilities
        if equity is not None:
            details["stockholders_equity"] = equity
        if liab_and_equity is not None:
            details["total_liabilities_and_equity"] = liab_and_equity

        if total_assets is None:
            return ReconciliationResult(
                statement_type=FinancialStatementType.BALANCE_SHEET,
                equation="Assets = Liabilities + Equity",
                expected_value=0.0,
                actual_value=0.0,
                difference=0.0,
                is_balanced=False,
                details=details,
            )

        # If explicit liabilities and equity sum row is present
        if liab_and_equity is not None:
            diff = abs(total_assets - liab_and_equity)
            return ReconciliationResult(
                statement_type=FinancialStatementType.BALANCE_SHEET,
                equation="Total Assets == Total Liabilities and Equity",
                expected_value=total_assets,
                actual_value=liab_and_equity,
                difference=round(diff, 4),
                is_balanced=diff <= tolerance,
                details=details,
            )

        # Otherwise sum individual liabilities and equity
        if total_liabilities is not None and equity is not None:
            computed_sum = total_liabilities + equity
            diff = abs(total_assets - computed_sum)
            return ReconciliationResult(
                statement_type=FinancialStatementType.BALANCE_SHEET,
                equation="Total Assets == Total Liabilities + Stockholders' Equity",
                expected_value=total_assets,
                actual_value=computed_sum,
                difference=round(diff, 4),
                is_balanced=diff <= tolerance,
                details=details,
            )

        return ReconciliationResult(
            statement_type=FinancialStatementType.BALANCE_SHEET,
            equation="Assets = Liabilities + Equity",
            expected_value=total_assets,
            actual_value=0.0,
            difference=total_assets,
            is_balanced=False,
            details=details,
        )

    @classmethod
    def reconcile_income_statement(
        cls, table: TableData, col_idx: int = 1, tolerance: float = 1.0
    ) -> ReconciliationResult:
        """Verify: Gross Profit == Total Revenue - Cost of Revenue / COGS."""
        revenue = cls._find_row_value(
            table, col_idx, "total revenue", "revenue", "net sales", "sales"
        )
        cogs = cls._find_row_value(
            table, col_idx, "cost of revenue", "cost of goods sold", "cogs", "cost of sales"
        )
        gross_profit = cls._find_row_value(table, col_idx, "gross profit", "gross margin")

        details: dict[str, float] = {}
        if revenue is not None:
            details["revenue"] = revenue
        if cogs is not None:
            details["cogs"] = cogs
        if gross_profit is not None:
            details["gross_profit"] = gross_profit

        if revenue is not None and cogs is not None and gross_profit is not None:
            expected_gp = revenue - abs(cogs)
            diff = abs(gross_profit - expected_gp)
            return ReconciliationResult(
                statement_type=FinancialStatementType.INCOME_STATEMENT,
                equation="Gross Profit == Revenue - COGS",
                expected_value=expected_gp,
                actual_value=gross_profit,
                difference=round(diff, 4),
                is_balanced=diff <= tolerance,
                details=details,
            )

        return ReconciliationResult(
            statement_type=FinancialStatementType.INCOME_STATEMENT,
            equation="Gross Profit == Revenue - COGS",
            expected_value=0.0,
            actual_value=0.0,
            difference=0.0,
            is_balanced=False,
            details=details,
        )
