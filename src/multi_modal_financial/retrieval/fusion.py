"""Retrieval fusion algorithms: Reciprocal Rank Fusion (RRF) and Convex Blending."""

from __future__ import annotations

import math
import time

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.types import (
    AgentQuery,
    RetrievalBenchmarkResult,
    ScoredChunk,
)


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Compute weighted Reciprocal Rank Fusion (RRF) across arbitrary ranked candidate lists."""
    rrf_scores: dict[str, float] = {}

    w_list = weights or [1.0] * len(ranked_lists)
    if len(w_list) < len(ranked_lists):
        w_list.extend([1.0] * (len(ranked_lists) - len(w_list)))

    for list_idx, candidate_list in enumerate(ranked_lists):
        w = w_list[list_idx]
        for rank, (doc_id, _) in enumerate(candidate_list):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (w / (k + rank + 1))

    # Sort descending by fused score
    sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_candidates


def min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalize score list to [0, 1]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def z_score_normalize(scores: list[float]) -> list[float]:
    """Standardize scores via zero-mean unit-variance."""
    if not scores:
        return []
    n = len(scores)
    mean_val = sum(scores) / n
    variance = sum((s - mean_val) ** 2 for s in scores) / max(1, n - 1)
    std_val = math.sqrt(variance)
    if std_val < 1e-9:
        return [0.0 for _ in scores]
    return [(s - mean_val) / std_val for s in scores]


class HybridRetriever:
    """Combines BM25 and dense vector results into unified ranked candidate chunks."""

    def __init__(self, index: HybridIndex, rrf_k: int = 60):
        self.index = index
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: AgentQuery,
        use_rrf: bool = True,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Execute hybrid search using RRF or convex alpha weighting with metadata constraints."""
        k = top_k or query.top_k
        fetch_limit = max(k * 4, 30)

        # 1. Sparse BM25 search
        sparse_hits = self.index.bm25.search(query.query_str, top_k=fetch_limit)
        sparse_dict = dict(sparse_hits)

        # 2. Dense vector search
        dense_hits = self.index.vector.search(query.query_str, top_k=fetch_limit)
        dense_dict = dict(dense_hits)

        # Filter candidates if metadata filtering is requested
        has_filter = bool(
            query.ticker_filter
            or query.period_filter
            or query.year_filter
            or query.doc_type_filter
            or query.modal_filter
        )

        allowed_ids: set[str] | None = None
        if has_filter:
            allowed_ids = set(
                self.index.filter_chunk_ids(
                    ticker=query.ticker_filter,
                    period=query.period_filter,
                    year=query.year_filter,
                    modal_type=query.modal_filter,
                    doc_type=query.doc_type_filter,
                )
            )
            sparse_hits = [h for h in sparse_hits if h[0] in allowed_ids]
            dense_hits = [h for h in dense_hits if h[0] in allowed_ids]

        combined_scored: list[ScoredChunk] = []

        if use_rrf:
            fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self.rrf_k)
            for rank, (cid, rrf_score) in enumerate(fused[:k]):
                chunk = self.index.get_chunk(cid)
                if not chunk:
                    continue
                combined_scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=round(rrf_score, 6),
                        dense_score=round(dense_dict.get(cid, 0.0), 4),
                        sparse_score=round(sparse_dict.get(cid, 0.0), 4),
                        rank=rank + 1,
                    )
                )
        else:
            # Convex combination: alpha * dense_norm + (1 - alpha) * sparse_norm
            all_cids = list(set(list(sparse_dict.keys()) + list(dense_dict.keys())))
            if allowed_ids is not None:
                all_cids = [cid for cid in all_cids if cid in allowed_ids]

            if all_cids:
                s_scores = [sparse_dict.get(cid, 0.0) for cid in all_cids]
                d_scores = [dense_dict.get(cid, 0.0) for cid in all_cids]

                norm_s = min_max_normalize(s_scores)
                norm_d = min_max_normalize(d_scores)

                combined_pairs = []
                alpha = query.alpha
                for i, cid in enumerate(all_cids):
                    final_s = (alpha * norm_d[i]) + ((1.0 - alpha) * norm_s[i])
                    combined_pairs.append((cid, final_s, d_scores[i], s_scores[i]))

                combined_pairs.sort(key=lambda x: x[1], reverse=True)
                for rank, (cid, score, d_sc, s_sc) in enumerate(combined_pairs[:k]):
                    chunk = self.index.get_chunk(cid)
                    if not chunk:
                        continue
                    combined_scored.append(
                        ScoredChunk(
                            chunk=chunk,
                            score=round(score, 6),
                            dense_score=round(d_sc, 4),
                            sparse_score=round(s_sc, 4),
                            rank=rank + 1,
                        )
                    )

        return combined_scored

    def benchmark(
        self,
        test_queries: list[tuple[AgentQuery, list[str]]],
        k: int = 5,
        use_rrf: bool = True,
    ) -> RetrievalBenchmarkResult:
        """Evaluate retrieval performance across a set of labeled evaluation queries."""
        if not test_queries:
            return RetrievalBenchmarkResult(
                method_name="Hybrid RRF" if use_rrf else "Convex Alpha Blend",
                recall_at_1=0.0,
                recall_at_k=0.0,
                mrr=0.0,
                ndcg=0.0,
                query_count=0,
                avg_latency_ms=0.0,
            )

        total_r1 = 0.0
        total_rk = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0
        latencies: list[float] = []

        for query, relevant_chunk_ids in test_queries:
            start_t = time.perf_counter()
            retrieved = self.retrieve(query, use_rrf=use_rrf, top_k=k)
            latency = (time.perf_counter() - start_t) * 1000.0
            latencies.append(latency)

            retrieved_ids = [sc.chunk.chunk_id for sc in retrieved]
            rel_set = set(relevant_chunk_ids)

            # Recall@1
            if retrieved_ids and retrieved_ids[0] in rel_set:
                total_r1 += 1.0

            # Recall@K
            hits_in_k = len(set(retrieved_ids).intersection(rel_set))
            total_rk += hits_in_k / max(1, len(rel_set))

            # MRR
            rr = 0.0
            for rank_idx, r_id in enumerate(retrieved_ids):
                if r_id in rel_set:
                    rr = 1.0 / (rank_idx + 1)
                    break
            total_mrr += rr

            # NDCG@K
            dcg = 0.0
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_set), k)))
            for rank_idx, r_id in enumerate(retrieved_ids):
                if r_id in rel_set:
                    dcg += 1.0 / math.log2(rank_idx + 2)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            total_ndcg += ndcg

        n = len(test_queries)
        method_name = "Hybrid RRF" if use_rrf else "Convex Alpha Blend"

        return RetrievalBenchmarkResult(
            method_name=method_name,
            recall_at_1=round(total_r1 / n, 4),
            recall_at_k=round(total_rk / n, 4),
            mrr=round(total_mrr / n, 4),
            ndcg=round(total_ndcg / n, 4),
            query_count=n,
            avg_latency_ms=round(sum(latencies) / n, 2),
        )
