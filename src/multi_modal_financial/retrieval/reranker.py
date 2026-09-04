"""Financial context reranker for second-stage candidate refinement."""

from __future__ import annotations

import re

from multi_modal_financial.types import AgentQuery, ModalType, ScoredChunk


class FinancialReranker:
    """Reranks initial candidate chunks based on financial entity, statement, and numerical salience."""

    YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")
    PERIOD_PATTERN = re.compile(r"\b(Q[1-4]|FY|FY\d{2,4})\b", re.IGNORECASE)

    def __init__(
        self,
        table_boost: float = 0.2,
        metric_boost: float = 0.25,
        figure_boost: float = 0.15,
        entity_boost: float = 0.2,
    ):
        self.table_boost = table_boost
        self.metric_boost = metric_boost
        self.figure_boost = figure_boost
        self.entity_boost = entity_boost

    def rerank(
        self,
        query: str | AgentQuery,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidates with domain-specific financial entity and modality bonuses."""
        if not candidates:
            return []

        query_str = query.query_str if isinstance(query, AgentQuery) else str(query)
        q_lower = query_str.lower()
        q_terms = set(re.findall(r"[a-zA-Z0-9_\-]+", q_lower))
        q_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", query_str))
        q_years = set(self.YEAR_PATTERN.findall(query_str))
        q_periods = {p.upper() for p in self.PERIOD_PATTERN.findall(query_str)}

        reranked: list[ScoredChunk] = []

        for item in candidates:
            chunk = item.chunk
            content_lower = chunk.content.lower()

            # Base score carryover
            base_score = item.score
            modality_bonus = 0.0
            entity_score = 0.0

            # 1. Lexical term overlap
            c_terms = set(re.findall(r"[a-zA-Z0-9_\-]+", content_lower))
            overlap_ratio = len(q_terms.intersection(c_terms)) / max(len(q_terms), 1)
            term_score = 0.3 * overlap_ratio

            # 2. Number alignment bonus
            num_score = 0.0
            if q_numbers:
                c_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", chunk.content))
                clean_q_nums = {n.replace(",", "") for n in q_numbers}
                clean_c_nums = {n.replace(",", "") for n in c_numbers}
                num_overlap = len(clean_q_nums.intersection(clean_c_nums)) / len(clean_q_nums)
                num_score = 0.35 * num_overlap

            # 3. Year and Period temporal alignment
            if q_years:
                c_years = set(self.YEAR_PATTERN.findall(chunk.content))
                if q_years.intersection(c_years):
                    entity_score += self.entity_boost

            if q_periods:
                c_periods = {p.upper() for p in self.PERIOD_PATTERN.findall(chunk.content)}
                if q_periods.intersection(c_periods):
                    entity_score += self.entity_boost

            # 4. Modality-specific bonuses
            if chunk.modal_type == ModalType.TABLE:
                modality_bonus += self.table_boost
                if any(
                    w in q_lower for w in ["table", "compare", "margin", "statement", "breakdown"]
                ):
                    modality_bonus += 0.1
            elif chunk.modal_type == ModalType.METRIC:
                modality_bonus += self.metric_boost
            elif chunk.modal_type == ModalType.FIGURE:
                modality_bonus += self.figure_boost
                if any(w in q_lower for w in ["chart", "figure", "graph", "trend", "trajectory"]):
                    modality_bonus += 0.1

            final_rerank_score = base_score + term_score + num_score + entity_score + modality_bonus

            reranked.append(
                ScoredChunk(
                    chunk=chunk,
                    score=round(final_rerank_score, 6),
                    dense_score=item.dense_score,
                    sparse_score=item.sparse_score,
                    rank=0,
                    rerank_score=round(final_rerank_score, 6),
                    modality_bonus=round(modality_bonus, 4),
                )
            )

        # Sort descending by composite score
        reranked.sort(key=lambda x: x.score, reverse=True)
        limit = top_k if top_k is not None else len(reranked)

        for rank_idx, r_item in enumerate(reranked[:limit]):
            r_item.rank = rank_idx + 1

        return reranked[:limit]
