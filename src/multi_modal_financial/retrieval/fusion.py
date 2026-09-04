"""Retrieval fusion algorithms: Reciprocal Rank Fusion (RRF) and Convex Blending."""

from __future__ import annotations

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.types import AgentQuery, ScoredChunk


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across two ranked candidate lists."""
    rrf_scores: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, (doc_id, _) in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort descending
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
        """Execute hybrid search using RRF or convex alpha weighting."""
        k = top_k or query.top_k
        fetch_limit = max(k * 3, 20)

        # 1. Sparse BM25 search
        sparse_hits = self.index.bm25.search(query.query_str, top_k=fetch_limit)
        sparse_dict = dict(sparse_hits)

        # 2. Dense vector search
        dense_hits = self.index.vector.search(query.query_str, top_k=fetch_limit)
        dense_dict = dict(dense_hits)

        # Filter candidates if metadata filtering is requested
        allowed_ids = set(
            self.index.filter_chunk_ids(
                ticker=query.ticker_filter,
                period=query.period_filter,
                year=query.year_filter,
            )
        )
        if query.ticker_filter or query.period_filter or query.year_filter:
            sparse_hits = [h for h in sparse_hits if h[0] in allowed_ids]
            dense_hits = [h for h in dense_hits if h[0] in allowed_ids]

        combined_scored: list[ScoredChunk] = []

        if use_rrf:
            fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self.rrf_k)
            for rank, (cid, score) in enumerate(fused[:k]):
                chunk = self.index.get_chunk(cid)
                if not chunk:
                    continue
                combined_scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=score,
                        dense_score=dense_dict.get(cid, 0.0),
                        sparse_score=sparse_dict.get(cid, 0.0),
                        rank=rank + 1,
                    )
                )
        else:
            # Convex combination: alpha * dense_norm + (1 - alpha) * sparse_norm
            all_cids = list(set(list(sparse_dict.keys()) + list(dense_dict.keys())))
            if allowed_ids:
                all_cids = [cid for cid in all_cids if cid in allowed_ids]

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
                        score=score,
                        dense_score=d_sc,
                        sparse_score=s_sc,
                        rank=rank + 1,
                    )
                )

        return combined_scored
