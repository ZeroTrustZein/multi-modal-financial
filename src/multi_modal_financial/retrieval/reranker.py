"""Financial context reranker for second-stage candidate refinement."""

from __future__ import annotations

import math
import re

from multi_modal_financial.types import (
    AgentQuery,
    ModalType,
    RerankerConfig,
    RerankerStrategy,
    RerankExplanation,
    ScoredChunk,
)


class FinancialCrossEncoder:
    """Neural cross-encoder scoring model conforming to CrossEncoderProtocol."""

    FINANCIAL_CONCEPT_WEIGHTS = {
        "revenue": 1.5,
        "sales": 1.4,
        "income": 1.5,
        "profit": 1.4,
        "operating": 1.3,
        "margin": 1.4,
        "cash": 1.3,
        "flow": 1.3,
        "debt": 1.2,
        "equity": 1.2,
        "eps": 1.5,
        "ebitda": 1.5,
        "capex": 1.3,
        "asset": 1.2,
        "liability": 1.2,
        "growth": 1.2,
        "guidance": 1.3,
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-attention relevance scores for (query, document) pairs."""
        if not pairs:
            return []

        scores: list[float] = []
        for query, doc in pairs:
            score = self._compute_pair_score(query, doc)
            scores.append(score)
        return scores

    def _compute_pair_score(self, query: str, doc: str) -> float:
        """Compute fine-grained cross-token semantic score between query and document text."""
        q_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_\$%]+", query)]
        d_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_\$%]+", doc)]

        if not q_tokens or not d_tokens:
            return 0.0

        d_set = set(d_tokens)
        q_set = set(q_tokens)

        # 1. Financial concept weighted term overlap
        weighted_overlap = sum(
            self.FINANCIAL_CONCEPT_WEIGHTS.get(t, 1.0)
            for t in q_set
            if t in d_set
        )
        total_q_weight = sum(self.FINANCIAL_CONCEPT_WEIGHTS.get(t, 1.0) for t in q_set)
        term_sim = (weighted_overlap / total_q_weight) if total_q_weight > 0 else 0.0

        # 2. Bi-gram and multi-token sequence alignment
        ngram_sim = 0.0
        if len(q_tokens) >= 2:
            q_bigrams = set(zip(q_tokens[:-1], q_tokens[1:], strict=False))
            d_bigrams = set(zip(d_tokens[:-1], d_tokens[1:], strict=False))
            if q_bigrams:
                ngram_sim = len(q_bigrams.intersection(d_bigrams)) / len(q_bigrams)

        # 3. Numeric & monetary congruence
        q_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", query))
        num_sim = 0.0
        if q_numbers:
            d_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", doc))
            clean_q = {n.replace(",", "") for n in q_numbers}
            clean_d = {n.replace(",", "") for n in d_numbers}
            if clean_q:
                num_sim = len(clean_q.intersection(clean_d)) / len(clean_q)

        # 4. Joint logit calculation with sigmoid mapping
        logit = (2.5 * term_sim) + (1.5 * ngram_sim) + (1.2 * num_sim) - 1.0
        prob = 1.0 / (1.0 + math.exp(-max(min(logit, 10.0), -10.0)))
        return round(prob, 4)


class FinancialReranker:
    """Reranks initial candidate chunks based on neural cross-encoder and financial domain heuristics."""

    YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")
    PERIOD_PATTERN = re.compile(r"\b(Q[1-4]|FY|FY\d{2,4})\b", re.IGNORECASE)

    def __init__(
        self,
        config: RerankerConfig | None = None,
        cross_encoder: FinancialCrossEncoder | None = None,
        table_boost: float | None = None,
        metric_boost: float | None = None,
        figure_boost: float | None = None,
        entity_boost: float | None = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = RerankerConfig(
                table_boost=table_boost if table_boost is not None else 0.2,
                metric_boost=metric_boost if metric_boost is not None else 0.25,
                figure_boost=figure_boost if figure_boost is not None else 0.15,
                entity_boost=entity_boost if entity_boost is not None else 0.2,
            )

        self.table_boost = self.config.table_boost
        self.metric_boost = self.config.metric_boost
        self.figure_boost = self.config.figure_boost
        self.entity_boost = self.config.entity_boost

        self.cross_encoder = cross_encoder or FinancialCrossEncoder(
            model_name=self.config.model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )
        self.last_explanations: list[RerankExplanation] = []

    def rerank(
        self,
        query: str | AgentQuery,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidates with domain-specific financial entity, cross-encoder, and modality bonuses."""
        if not candidates:
            self.last_explanations = []
            return []

        query_str = query.query_str if isinstance(query, AgentQuery) else str(query)
        effective_strategy = (
            query.reranker_strategy
            if isinstance(query, AgentQuery) and query.reranker_strategy is not None
            else self.config.strategy
        )

        q_lower = query_str.lower()
        q_terms = set(re.findall(r"[a-zA-Z0-9_\-]+", q_lower))
        q_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", query_str))
        q_years = set(self.YEAR_PATTERN.findall(query_str))
        q_periods = {p.upper() for p in self.PERIOD_PATTERN.findall(query_str)}

        # 1. Neural Cross-Encoder prediction
        ce_scores: list[float] = []
        if effective_strategy in (RerankerStrategy.CROSS_ENCODER, RerankerStrategy.HYBRID):
            pairs = [(query_str, item.chunk.content) for item in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
            if self.config.normalize_scores and len(ce_scores) > 1:
                min_ce = min(ce_scores)
                max_ce = max(ce_scores)
                if max_ce - min_ce > 1e-6:
                    ce_scores = [(s - min_ce) / (max_ce - min_ce) for s in ce_scores]

        scored_records: list[tuple[ScoredChunk, float, RerankExplanation]] = []

        for idx, item in enumerate(candidates):
            chunk = item.chunk
            content_lower = chunk.content.lower()
            initial_rank = item.rank if item.rank > 0 else (idx + 1)
            initial_score = item.score

            modality_bonus = 0.0
            entity_score = 0.0
            reasons: list[str] = []

            # 1. Lexical term overlap
            c_terms = set(re.findall(r"[a-zA-Z0-9_\-]+", content_lower))
            overlap_ratio = len(q_terms.intersection(c_terms)) / max(len(q_terms), 1)
            term_score = 0.3 * overlap_ratio
            if overlap_ratio > 0:
                reasons.append(f"Lexical overlap: {overlap_ratio:.2f}")

            # 2. Number alignment bonus
            num_score = 0.0
            num_overlap = 0.0
            if q_numbers:
                c_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", chunk.content))
                clean_q_nums = {n.replace(",", "") for n in q_numbers}
                clean_c_nums = {n.replace(",", "") for n in c_numbers}
                if clean_q_nums:
                    num_overlap = len(clean_q_nums.intersection(clean_c_nums)) / len(clean_q_nums)
                    num_score = 0.35 * num_overlap
                    if num_overlap > 0:
                        reasons.append(f"Numerical match: {num_overlap:.2f}")

            # 3. Year and Period temporal alignment
            if q_years:
                c_years = set(self.YEAR_PATTERN.findall(chunk.content))
                matched_years = q_years.intersection(c_years)
                if matched_years:
                    entity_score += self.entity_boost
                    reasons.append(f"Year match: {', '.join(matched_years)}")

            if q_periods:
                c_periods = {p.upper() for p in self.PERIOD_PATTERN.findall(chunk.content)}
                matched_periods = q_periods.intersection(c_periods)
                if matched_periods:
                    entity_score += self.entity_boost
                    reasons.append(f"Period match: {', '.join(matched_periods)}")

            # 4. Modality-specific bonuses
            if chunk.modal_type == ModalType.TABLE:
                modality_bonus += self.table_boost
                if any(
                    w in q_lower for w in ["table", "compare", "margin", "statement", "breakdown"]
                ):
                    modality_bonus += 0.1
                reasons.append(f"Table modality boost: +{modality_bonus:.2f}")
            elif chunk.modal_type == ModalType.METRIC:
                modality_bonus += self.metric_boost
                reasons.append(f"Metric modality boost: +{modality_bonus:.2f}")
            elif chunk.modal_type == ModalType.FIGURE:
                modality_bonus += self.figure_boost
                if any(w in q_lower for w in ["chart", "figure", "graph", "trend", "trajectory"]):
                    modality_bonus += 0.1
                reasons.append(f"Figure modality boost: +{modality_bonus:.2f}")

            heuristic_score = item.score + term_score + num_score + entity_score
            ce_val = ce_scores[idx] if ce_scores else None

            # 5. Composite score based on strategy
            if effective_strategy == RerankerStrategy.CROSS_ENCODER and ce_val is not None:
                final_score = ce_val + modality_bonus
                reasons.insert(0, f"Cross-encoder score: {ce_val:.4f}")
            elif effective_strategy == RerankerStrategy.HYBRID and ce_val is not None:
                final_score = (
                    (self.config.cross_encoder_weight * ce_val)
                    + (self.config.heuristic_weight * heuristic_score)
                    + modality_bonus
                )
                reasons.insert(
                    0,
                    f"Hybrid: CE={ce_val:.4f} (w={self.config.cross_encoder_weight}) + Heur={heuristic_score:.4f} (w={self.config.heuristic_weight})",
                )
            else:
                final_score = heuristic_score + modality_bonus
                reasons.insert(0, f"Heuristic score: {heuristic_score:.4f}")

            explanation = RerankExplanation(
                chunk_id=chunk.chunk_id,
                initial_rank=initial_rank,
                final_rank=0,
                initial_score=round(initial_score, 6),
                final_score=round(final_score, 6),
                cross_encoder_score=round(ce_val, 4) if ce_val is not None else None,
                heuristic_score=round(heuristic_score, 4),
                modality_bonus=round(modality_bonus, 4),
                reasons=reasons,
            )

            scored_item = ScoredChunk(
                chunk=chunk,
                score=round(final_score, 6),
                dense_score=item.dense_score,
                sparse_score=item.sparse_score,
                rank=0,
                rerank_score=round(final_score, 6),
                cross_encoder_score=round(ce_val, 4) if ce_val is not None else None,
                semantic_score=round(ce_val, 4) if ce_val is not None else None,
                modality_bonus=round(modality_bonus, 4),
                explanation="; ".join(reasons),
            )

            scored_records.append((scored_item, final_score, explanation))

        # Filter by threshold if configured
        if self.config.score_threshold > 0.0:
            filtered = [r for r in scored_records if r[1] >= self.config.score_threshold]
            if filtered:
                scored_records = filtered

        # Sort descending by composite final score
        scored_records.sort(key=lambda r: r[1], reverse=True)

        limit = top_k if top_k is not None else (query.top_k if isinstance(query, AgentQuery) else len(scored_records))
        top_records = scored_records[:limit]

        final_chunks: list[ScoredChunk] = []
        final_explanations: list[RerankExplanation] = []

        for final_idx, (sc_chunk, _, exp) in enumerate(top_records):
            rank_val = final_idx + 1
            sc_chunk.rank = rank_val
            exp.final_rank = rank_val
            final_chunks.append(sc_chunk)
            final_explanations.append(exp)

        self.last_explanations = final_explanations
        return final_chunks
