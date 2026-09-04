"""Multi-period and multi-entity financial comparative reasoning subsystem."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from multi_modal_financial.types import Document, FinancialMetric, TableData


@dataclass
class VarianceResult:
    """Quantitative variance and trend between two financial observations."""

    metric_name: str
    base_value: float
    compare_value: float
    absolute_change: float
    percentage_change: float | None
    trend: str  # "UP", "DOWN", "FLAT"
    unit: str = "USD"

    def to_dict(self) -> dict:
        return asdict(self)


class PeriodComparator:
    """Compares financial metrics across fiscal periods, quarters, or entities."""

    @staticmethod
    def calculate_variance(
        name: str,
        base_val: float,
        compare_val: float,
        unit: str = "USD",
        tolerance: float = 1e-4,
    ) -> VarianceResult:
        """Compute variance and percentage change with handling for zero/negative base."""
        diff = compare_val - base_val
        if abs(diff) < tolerance:
            pct_change: float | None = 0.0
            trend = "FLAT"
        elif abs(base_val) > 1e-6:
            pct_change = round((diff / abs(base_val)) * 100.0, 2)
            trend = "UP" if diff > 0 else "DOWN"
        else:
            pct_change = None
            trend = "UP" if diff > 0 else "DOWN"

        return VarianceResult(
            metric_name=name,
            base_value=round(base_val, 4),
            compare_value=round(compare_val, 4),
            absolute_change=round(diff, 4),
            percentage_change=pct_change,
            trend=trend,
            unit=unit,
        )

    @classmethod
    def compare_metrics(
        cls,
        base_metrics: list[FinancialMetric],
        compare_metrics: list[FinancialMetric],
    ) -> list[VarianceResult]:
        """Align and compare two metric lists by metric name."""
        base_map = {
            m.name.lower().strip(): m
            for m in base_metrics
            if m.value is not None and not math.isnan(m.value)
        }
        results: list[VarianceResult] = []

        for m_comp in compare_metrics:
            if m_comp.value is None or math.isnan(m_comp.value):
                continue
            key = m_comp.name.lower().strip()
            if key in base_map:
                m_base = base_map[key]
                if m_base.value is None or math.isnan(m_base.value):
                    continue
                unit = (
                    "%" if (m_comp.scale == "percent" or "%" in m_comp.raw_value) else m_comp.unit
                )
                var = cls.calculate_variance(
                    name=m_comp.name,
                    base_val=m_base.value,
                    compare_val=m_comp.value,
                    unit=unit,
                )
                results.append(var)

        return results

    @classmethod
    def compare_table_columns(
        cls,
        table: TableData,
        base_col: int = 2,
        compare_col: int = 1,
    ) -> list[VarianceResult]:
        """Compare two columns in the same table (e.g. Prior Period vs Current Period)."""
        variances: list[VarianceResult] = []
        if len(table.headers) <= max(base_col, compare_col):
            return []

        for row in table.rows:
            if not row or len(row) <= max(base_col, compare_col):
                continue
            name = row[0].strip()
            base_metric = FinancialMetric.from_raw(name=name, raw_value=row[base_col])
            comp_metric = FinancialMetric.from_raw(name=name, raw_value=row[compare_col])

            if (
                base_metric.value is not None
                and comp_metric.value is not None
                and not math.isnan(base_metric.value)
                and not math.isnan(comp_metric.value)
            ):
                unit = (
                    "%"
                    if (comp_metric.scale == "percent" or "%" in comp_metric.raw_value)
                    else str(table.unit or "USD")
                )
                var = cls.calculate_variance(
                    name=name,
                    base_val=base_metric.value,
                    compare_val=comp_metric.value,
                    unit=unit,
                )
                variances.append(var)

        return variances

    @classmethod
    def compare_documents(
        cls,
        doc_base: Document,
        doc_compare: Document,
    ) -> list[VarianceResult]:
        """Extract and compare all matching metrics across two documents."""
        base_metrics = doc_base.get_all_metrics()
        compare_metrics = doc_compare.get_all_metrics()
        return cls.compare_metrics(base_metrics, compare_metrics)
