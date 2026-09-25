"""Tests for index persistence, embedding cache, and query cache subsystem."""

from pathlib import Path

import numpy as np
import pytest

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.interfaces import SemanticCacheProtocol
from multi_modal_financial.storage.cache import EmbeddingCache, QueryCache, SemanticCache
from multi_modal_financial.storage.persistence import IndexManifest, IndexPersistence
from multi_modal_financial.types import (
    AgentResponse,
    CacheEvictionPolicy,
    CacheHitType,
    Chunk,
    Document,
    DocumentMetadata,
    ModalType,
    SemanticCacheConfig,
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

    def test_manifest_serialization(self):
        manifest = IndexManifest(
            total_documents=10,
            total_chunks=50,
            reranker_strategy="hybrid",
            semantic_cache_entries=5,
        )
        d = manifest.to_dict()
        assert d["reranker_strategy"] == "hybrid"
        assert d["semantic_cache_entries"] == 5

        # Unknown field tolerance
        d["unknown_future_field"] = "foo"
        rebuilt = IndexManifest.from_dict(d)
        assert rebuilt.total_documents == 10
        assert rebuilt.reranker_strategy == "hybrid"


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


class TestSemanticCache:
    """Unit tests for vector similarity matching, policies, and TTL in SemanticCache."""

    @pytest.fixture
    def sample_response(self) -> AgentResponse:
        return AgentResponse(
            query="Net sales 2025",
            answer="Net sales were $391,035 million.",
            citations=[],
            grounding_verdicts=[],
            retrieved_chunks=[],
            execution_time_ms=5.0,
            overall_confidence=0.98,
        )

    def test_protocol_conformance(self) -> None:
        cache = SemanticCache()
        assert isinstance(cache, SemanticCacheProtocol)

    def test_exact_hit_fast_path(self, sample_response: AgentResponse) -> None:
        cache = SemanticCache()
        cache.put("Net Sales for AAPL 2025", sample_response)

        # Lookup with identical/normalized string
        res = cache.get("  net sales for aapl 2025  ")
        assert res is not None
        assert res.hit is True
        assert res.hit_type == CacheHitType.EXACT
        assert res.similarity == 1.0
        assert res.response is not None
        assert res.response.answer == sample_response.answer
        assert res.lookup_latency_ms >= 0.0

    def test_semantic_hit_cosine(self, sample_response: AgentResponse) -> None:
        config = SemanticCacheConfig(similarity_threshold=0.75)
        cache = SemanticCache(config=config)
        cache.put("What were total net sales in 2025?", sample_response)

        # Semantically overlapping query
        res = cache.get("total net sales 2025")
        assert res is not None
        assert res.hit is True
        assert res.hit_type in (CacheHitType.EXACT, CacheHitType.SEMANTIC)
        assert res.similarity >= 0.75
        assert res.response is not None

    def test_miss_below_threshold(self, sample_response: AgentResponse) -> None:
        config = SemanticCacheConfig(similarity_threshold=0.95)
        cache = SemanticCache(config=config)
        cache.put("Apple fiscal 2025 iPhone sales", sample_response)

        # Unrelated query
        res = cache.get("Weather forecast in Miami Florida")
        assert res is not None
        assert res.hit is False
        assert res.hit_type == CacheHitType.NONE

    def test_lru_eviction(self, sample_response: AgentResponse) -> None:
        config = SemanticCacheConfig(max_entries=2, eviction_policy=CacheEvictionPolicy.LRU)
        cache = SemanticCache(config=config)

        cache.put("q1", sample_response)
        cache.put("q2", sample_response)

        # Touch q1 to make q2 the least recently used
        cache.get("q1")

        # Put third entry to force eviction
        cache.put("q3", sample_response)

        stats = cache.stats()
        assert stats.entry_count == 2
        assert stats.evictions == 1
        # q2 should have been evicted
        res_q2 = cache.get("q2")
        res_q1 = cache.get("q1")
        assert res_q2 is not None and res_q2.hit is False
        assert res_q1 is not None and res_q1.hit is True

    def test_lfu_eviction(self, sample_response: AgentResponse) -> None:
        config = SemanticCacheConfig(max_entries=2, eviction_policy=CacheEvictionPolicy.LFU)
        cache = SemanticCache(config=config)

        cache.put("query_a", sample_response)
        cache.put("query_b", sample_response)

        # Access query_b multiple times to increase its frequency
        cache.get("query_b")
        cache.get("query_b")

        # Put query_c to trigger eviction
        cache.put("query_c", sample_response)

        assert cache.stats().evictions == 1
        # query_a had access_count 1, query_b had 3 -> query_a evicted
        res_qa = cache.get("query_a")
        res_qb = cache.get("query_b")
        assert res_qa is not None and res_qa.hit is False
        assert res_qb is not None and res_qb.hit is True

    def test_fifo_eviction(self, sample_response: AgentResponse) -> None:
        import time

        config = SemanticCacheConfig(max_entries=2, eviction_policy=CacheEvictionPolicy.FIFO)
        cache = SemanticCache(config=config)

        cache.put("first_in", sample_response)
        time.sleep(0.01)
        cache.put("second_in", sample_response)

        # Access first_in heavily (in FIFO this should NOT prevent eviction)
        cache.get("first_in")
        cache.get("first_in")

        # Put third_in to trigger eviction
        cache.put("third_in", sample_response)

        assert cache.stats().evictions == 1
        res_first = cache.get("first_in")
        res_second = cache.get("second_in")
        assert res_first is not None and res_first.hit is False
        assert res_second is not None and res_second.hit is True

    def test_ttl_expiration(self, sample_response: AgentResponse) -> None:
        import time

        config = SemanticCacheConfig(ttl_seconds=0.02)
        cache = SemanticCache(config=config)
        cache.put("short_lived", sample_response)

        # Immediately available
        res_early = cache.get("short_lived")
        assert res_early is not None and res_early.hit is True

        # Wait for TTL to elapse
        time.sleep(0.03)
        res_late = cache.get("short_lived")
        assert res_late is not None and res_late.hit is False

    def test_clear_and_stats_telemetry(self, sample_response: AgentResponse) -> None:
        cache = SemanticCache()
        cache.put("apple", sample_response)
        cache.get("apple")
        cache.get("nonexistent")

        st = cache.stats()
        assert st.total_queries == 2
        assert st.exact_hits == 1
        assert st.misses == 1
        assert st.hit_rate == 0.5
        assert st.entry_count == 1

        cache.clear()
        st_after = cache.stats()
        assert st_after.total_queries == 0
        assert st_after.entry_count == 0
        assert st_after.hit_rate == 0.0

    def test_save_and_load_semantic_cache(
        self, sample_response: AgentResponse, tmp_path: Path
    ) -> None:
        cache = SemanticCache()
        cache.put("Apple fiscal 2025 revenue", sample_response)
        cache.put("Microsoft Cloud revenue 2025", sample_response)

        cache_file = tmp_path / "semantic_cache.json"
        saved_path = IndexPersistence.save_semantic_cache(cache, cache_file)
        assert saved_path.exists()

        restored_cache = IndexPersistence.load_semantic_cache(saved_path)
        assert len(restored_cache._entries) == 2
        lookup = restored_cache.get("Apple fiscal 2025 revenue")
        assert lookup is not None
        assert lookup.hit is True
        assert lookup.hit_type == CacheHitType.EXACT

    def test_semantic_cache_disabled(self, sample_response: AgentResponse) -> None:
        config = SemanticCacheConfig(enabled=False)
        cache = SemanticCache(config=config)
        cache.put("Query when disabled", sample_response)
        assert len(cache._entries) == 0
        assert cache.get("Query when disabled") is None

    def test_semantic_cache_distance_metrics(self, sample_response: AgentResponse) -> None:
        # 1. Euclidean distance
        cfg_euc = SemanticCacheConfig(distance_metric="euclidean", similarity_threshold=0.5)
        cache_euc = SemanticCache(config=cfg_euc)
        sim_euc = cache_euc._compute_similarity([1.0, 0.0], [0.0, 1.0])
        assert 0.0 < sim_euc < 1.0

        # 2. Dot product
        cfg_dot = SemanticCacheConfig(distance_metric="dot", similarity_threshold=0.5)
        cache_dot = SemanticCache(config=cfg_dot)
        sim_dot = cache_dot._compute_similarity([0.5, 0.5], [0.5, 0.5])
        assert sim_dot == 0.5

        # 3. Cosine distance with near-zero norm
        cfg_cos = SemanticCacheConfig(distance_metric="cosine")
        cache_cos = SemanticCache(config=cfg_cos)
        sim_zero = cache_cos._compute_similarity([0.0, 0.0], [1.0, 1.0])
        assert sim_zero == 0.0

    def test_semantic_cache_expired_purge_on_put(self, sample_response: AgentResponse) -> None:
        import time

        config = SemanticCacheConfig(max_entries=2, ttl_seconds=0.02)
        cache = SemanticCache(config=config)
        cache.put("q_old1", sample_response)
        cache.put("q_old2", sample_response)
        assert len(cache._entries) == 2

        time.sleep(0.03)
        # Put third entry when previous 2 entries are expired
        cache.put("q_new", sample_response)
        # Expired entries should be purged, leaving only q_new
        assert len(cache._entries) == 1
        assert "q_new" in [e.query for e in cache._entries.values()]

    def test_semantic_cache_update_existing_entry(self, sample_response: AgentResponse) -> None:
        cache = SemanticCache()
        cache.put("update_me", sample_response)
        assert cache.stats().entry_count == 1

        updated_resp = AgentResponse(
            query="update_me",
            answer="Updated Answer",
            citations=[],
            grounding_verdicts=[],
            retrieved_chunks=[],
            execution_time_ms=5.0,
            overall_confidence=0.99,
        )
        cache.put("update_me", updated_resp)
        assert cache.stats().entry_count == 1
        res = cache.get("update_me")
        assert res is not None and res.response is not None
        assert res.response.answer == "Updated Answer"

    def test_semantic_cache_missing_vector_skip(self, sample_response: AgentResponse) -> None:
        cache = SemanticCache()
        cache.put("entry_with_no_vec", sample_response)
        # Manually clear vector to simulate missing vector entry
        for e in cache._entries.values():
            e.query_vector = []

        res = cache.get("unseen query that scans")
        assert res is not None and res.hit is False

    def test_persistence_semantic_cache_errors(self, tmp_path: Path) -> None:
        import pytest

        # Missing file
        missing_file = tmp_path / "missing_cache.json"
        with pytest.raises(FileNotFoundError, match="Persisted cache not found"):
            IndexPersistence.load_semantic_cache(missing_file)

        # Subdirectory auto-creation in save_semantic_cache
        deep_file = tmp_path / "nested" / "dir" / "cache.json"
        cache = SemanticCache()
        saved = IndexPersistence.save_semantic_cache(cache, deep_file)
        assert saved.exists()
