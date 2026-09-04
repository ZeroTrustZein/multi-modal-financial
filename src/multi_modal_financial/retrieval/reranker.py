"""Financial context reranker for second-stage candidate refinement."""

from __future__ import annotations

import re

from multi_modal_financial.types import ModalType, ScoredChunk


class FinancialReranker:
    """Reranks initial candidate chunks based on financial entity and numerical salience."""

    def __init__(self, table_boost: float = 0.2, metric_boost: float = 0.25):
        self.table_boost = table_boost
        self.metric_boost = metric_boost

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidates with domain-specific financial bonuses."""
        if not candidates:
            return []

        q_terms = set(re.findall(r"[a-zA-Z0-9_\-]+", query.lower()))
        q_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))

        reranked: list[ScoredChunk] = []

        for item in candidates:
            chunk = item.chunk
            content_lower = chunk.content.lower()

            # Base score carryover
            score = item.score

            # Term overlap bonus
            c_terms = set(re.findall(r"[a-zA-Z0-9_\-]+", content_lower))
            overlap_ratio = len(q_terms.intersection(c_terms)) / max(len(q_terms), 1)
            score += 0.3 * overlap_ratio

            # Exact number match bonus
            if q_numbers:
                c_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", chunk.content))
                num_overlap = len(q_numbers.intersection(c_numbers)) / len(q_numbers)
                score += 0.35 * num_overlap

            # Modality boosts: if financial table or metric matches query keywords
            if chunk.modal_type == ModalType.TABLE:
                score += self.table_boost
            elif chunk.modal_type == ModalType.METRIC:
                score += self.metric_boost

            reranked.append(
                ScoredChunk(
                    chunk=chunk,
                    score=round(score, 6),
                    dense_score=item.dense_score,
                    sparse_score=item.sparse_score,
                    rank=0,
                )
            )

        # Sort descending
        reranked.sort(key=lambda x: x.score, reverse=True)
        limit = top_k if top_k is not None else len(reranked)

        for rank, r_item in enumerate(reranked[:limit]):
            r_item.rank = rank + 1

        return reranked[:limit]
