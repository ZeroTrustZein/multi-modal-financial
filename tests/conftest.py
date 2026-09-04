"""Pytest fixtures for financial RAG test suite."""

import pytest

from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.types import (
    Chunk,
    Document,
    DocumentMetadata,
    ModalType,
    TableData,
)


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="sec_aapl_10k_2025",
        filename="aapl-2025.txt",
        ticker="AAPL",
        period="FY",
        year=2025,
        doc_type="10-K",
        page_count=12,
    )


@pytest.fixture
def sample_table() -> TableData:
    return TableData(
        title="Consolidated Statements of Operations",
        headers=["Category", "2024", "2025"],
        rows=[
            ["Total Net Sales", "$383,285", "$391,035"],
            ["Cost of Sales", "$214,137", "$210,352"],
            ["Operating Income", "$114,301", "$123,216"],
        ],
        unit="USD",
        scale="millions",
    )


@pytest.fixture
def sample_document(sample_metadata: DocumentMetadata, sample_table: TableData) -> Document:
    c1 = Chunk(
        chunk_id="aapl_c1",
        doc_id=sample_metadata.doc_id,
        modal_type=ModalType.TEXT,
        content="Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, and wearables.",
        page_number=1,
    )
    c2 = Chunk(
        chunk_id="aapl_c2",
        doc_id=sample_metadata.doc_id,
        modal_type=ModalType.TABLE,
        content=sample_table.to_markdown(),
        page_number=2,
        table_data=sample_table,
    )
    c3 = Chunk(
        chunk_id="aapl_c3",
        doc_id=sample_metadata.doc_id,
        modal_type=ModalType.METRIC,
        content="Diluted EPS: $6.08 in fiscal 2025 compared to $5.67 in fiscal 2024.",
        page_number=3,
    )
    return Document(
        doc_id=sample_metadata.doc_id,
        metadata=sample_metadata,
        chunks=[c1, c2, c3],
    )


@pytest.fixture
def populated_pipeline(sample_document: Document) -> FinancialRAGPipeline:
    pipeline = FinancialRAGPipeline()
    pipeline.index.index_document(sample_document)
    return pipeline
