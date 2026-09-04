"""BM25 Okapi Sparse Search Engine with financial tokenization."""

from __future__ import annotations

import math
import re
from collections import Counter


class BM25Index:
    """Okapi BM25 implementation tuned for financial vocabulary and numerical terms."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_ids: list[str] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize preserving financial tickers, numbers, percentages, and currencies."""
        # Lowercase everything except ticker symbols or uppercase acronyms
        # Regex captures:
        # - percentages: 12.5%
        # - currencies: $1,234.50 or $12M
        # - numbers: 100,000 or 45.8
        # - alphanumeric tokens & acronyms: Q3, FY2025, EBITDA, AAPL
        pattern = r"\$?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?%?|[a-zA-Z0-9_\-]+"
        tokens = re.findall(pattern, text.lower())
        return [t for t in tokens if len(t) > 1 or t.isdigit()]

    def fit(self, doc_ids: list[str], documents: list[str]) -> BM25Index:
        """Fit index on collection of document texts."""
        self.doc_ids = list(doc_ids)
        self.corpus_size = len(documents)
        self.doc_tokens = []
        self.doc_len = []
        self.doc_freqs = Counter()

        total_length = 0
        for text in documents:
            tokens = self.tokenize(text)
            self.doc_tokens.append(tokens)
            tok_len = len(tokens)
            self.doc_len.append(tok_len)
            total_length += tok_len

            # Unique terms in doc for DF
            unique_terms = set(tokens)
            for t in unique_terms:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0
        self._compute_idf()
        return self

    def _compute_idf(self) -> None:
        """Compute IDF with standard Okapi smoothing."""
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF formulation: log((N - n + 0.5) / (n + 0.5) + 1.0)
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(idf_val, 1e-4)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search query against BM25 index and return (doc_id, score) pairs."""
        if self.corpus_size == 0:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scores: list[float] = [0.0] * self.corpus_size

        for term in q_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                doc_len = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                numerator = tf * (self.k1 + 1.0)
                scores[doc_idx] += idf * (numerator / denom)

        # Rank documents
        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
        results = [
            (self.doc_ids[i], scores[i])
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]
        return results
