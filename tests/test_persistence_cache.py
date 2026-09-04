"""Tests for index persistence, embedding cache, and query cache subsystem."""

from pathlib import Path

import numpy as np
import pytest

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.storage.cache import EmbeddingCache, QueryCache
from multi_modal_financial.storage.persistence import IndexPersistence
from multi_modal_financial.types import (
    AgentResponse,
    Chunk,
    Document,
    DocumentMetadata,
    ModalType,
)


class TestIndexPersistence:
    """Unit tests for saving, loading, and serializing HybridIndex."""

    @pytest.fixture
    def populated_index(self) -> HybridIndex:
        idx = HybridIndex(dimension=64)
        doc1 = Document(
            doc_id="doc_aapl",
            metadata=DocumentMetadata(
                doc_id="doc_aapl",
                filename="doc_aapl.txt",
                ticker="AAPL",
                year=2025,
                period="Q3",
            ),
            chunks=[
                Chunk(
                    chunk_id="chunk_aapl_1",
                    doc_id="doc_aapl",
                    content="Apple reported quarterly revenue of $94.9 billion, up 6% YoY.",
                    modal_type=ModalType.TEXT,
                ),
                Chunk(
                    chunk_id="chunk_aapl_2",
                    doc_id="doc_aapl",
                    content="Services gross margin reached 74.0%, setting an all-time record.",
                    modal_type=ModalType.TEXT,
                ),
            ],
        )
        doc2 = Document(
            doc_id="doc_msft",
            metadata=DocumentMetadata(
                doc_id="doc_msft",
                filename="doc_msft.txt",
                ticker="MSFT",
                year=2025,
                period="Q3",
            ),
            chunks=[
                Chunk(
                    chunk_id="chunk_msft_1",
                    doc_id="doc_msft",
                    content="Microsoft Cloud revenue surged to $38.9 billion, driven by AI demand.",
                    modal_type=ModalType.TEXT,
                ),
            ],
        )
        idx.index_document(doc1)
        idx.index_document(doc2)
        return idx

    def test_save_and_load_directory(self, populated_index: HybridIndex, tmp_path: Path):
        save_dir = tmp_path / "test_persist_dir"
        persisted_path = IndexPersistence.save(populated_index, save_dir, compress=False)
        assert persisted_path.exists()
        assert (persisted_path / "manifest.json").exists()
        assert (persisted_path / "vectors.npz").exists()
        assert (persisted_path / "bm25.json").exists()

        restored_index = IndexPersistence.load(save_dir)
        assert restored_index.total_documents() == 2
        assert restored_index.total_chunks() == 3

        # Test BM25 and vector search on restored index
        res_bm25 = restored_index.bm25.search("Apple revenue", top_k=2)
        assert len(res_bm25) > 0
        assert res_bm25[0][0] == "chunk_aapl_1"

        res_vec = restored_index.vector.search("Microsoft Cloud AI", top_k=2)
        assert len(res_vec) > 0
        assert res_vec[0][0] == "chunk_msft_1"

    def test_save_and_load_zip_archive(self, populated_index: HybridIndex, tmp_path: Path):
        zip_path = tmp_path / "index_archive.zip"
        persisted_zip = IndexPersistence.save(populated_index, zip_path, compress=True)
        assert persisted_zip.exists()
        assert persisted_zip.suffix == ".zip"

        restored_index = IndexPersistence.load(persisted_zip)
        assert restored_index.total_documents() == 2
        assert restored_index.total_chunks() == 3

    def test_export_and_import_corpus(self, populated_index: HybridIndex, tmp_path: Path):
        export_file = tmp_path / "corpus_export.json"
        IndexPersistence.export_corpus(populated_index, export_file)
        assert export_file.exists()

        imported_index = IndexPersistence.import_corpus(export_file)
        assert imported_index.total_documents() == 2
        assert imported_index.total_chunks() == 3
        # Should have functioning search indices
        assert imported_index.bm25.corpus_size == 3


class TestCaches:
    """Unit tests for EmbeddingCache and QueryCache."""

    def test_embedding_cache_put_get(self):
        cache = EmbeddingCache(max_size=2)
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)

        cache.put("text 1", vec1)
        cache.put("text 2", vec2)

        cached1 = cache.get("text 1")
        assert cached1 is not None
        assert np.allclose(cached1, vec1)

        # Trigger LRU eviction by adding third item
        vec3 = np.array([0.5, 0.5], dtype=np.float32)
        cache.put("text 3", vec3)

        assert len(cache._cache) == 2
        # text 2 was least recently used because text 1 was accessed via get
        assert cache.get("text 2") is None
        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["hits"] >= 1

    def test_embedding_cache_get_or_compute(self):
        cache = EmbeddingCache(max_size=10)
        computations = 0

        def dummy_embed(texts: list[str]) -> np.ndarray:
            nonlocal computations
            computations += len(texts)
            return np.ones((len(texts), 4), dtype=np.float32)

        res1 = cache.get_or_compute(["apple", "banana"], dummy_embed)
        assert computations == 2
        assert res1.shape == (2, 4)

        # Call again with one cached and one new
        res2 = cache.get_or_compute(["apple", "cherry"], dummy_embed)
        assert computations == 3  # only cherry was computed
        assert res2.shape == (2, 4)

    def test_query_cache(self):
        cache = QueryCache(max_size=2, ttl_seconds=100.0)
        key1 = QueryCache.make_key("revenue in Q3", ticker="AAPL", top_k=3, alpha=0.5)
        resp1 = AgentResponse(
            query="revenue in Q3",
            answer="Revenue was $94.9B",
            citations=[],
            grounding_verdicts=[],
            retrieved_chunks=[],
            execution_time_ms=12.0,
            overall_confidence=0.95,
        )

        cache.put(key1, resp1)
        cached = cache.get(key1)
        assert cached is not None
        assert cached.answer == resp1.answer

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["size"] == 1
